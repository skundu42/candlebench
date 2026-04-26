use anyhow::{bail, Context, Result};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use humansize::{format_size, DECIMAL};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Terminal,
};
use std::io::{self, Stdout};
use std::path::Path;
use std::time::Duration;

use crate::gguf::GgufSummary;
use crate::inspect::InspectSummary;
use crate::util::{format_number, format_shape};

enum ModelSummary {
    SafeTensors(InspectSummary),
    Gguf(GgufSummary),
}

pub fn run(path: Option<&Path>) -> Result<()> {
    let summary = match path {
        Some(path) => Some(load_summary(path)?),
        None => None,
    };

    let mut terminal = setup_terminal()?;
    let result = run_loop(&mut terminal, summary.as_ref());
    restore_terminal(&mut terminal)?;
    result
}

fn load_summary(path: &Path) -> Result<ModelSummary> {
    if !path.exists() {
        bail!("file does not exist: {}", path.display());
    }

    match path.extension().and_then(|ext| ext.to_str()) {
        Some("gguf") => Ok(ModelSummary::Gguf(crate::gguf::inspect_gguf(path)?)),
        Some("safetensors") => Ok(ModelSummary::SafeTensors(
            crate::inspect::inspect_safetensors(path)?,
        )),
        _ => bail!("TUI supports .safetensors and .gguf files"),
    }
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode().context("failed to enable terminal raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    Terminal::new(CrosstermBackend::new(stdout)).context("failed to initialize terminal")
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().context("failed to disable terminal raw mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .context("failed to leave alternate screen")?;
    terminal.show_cursor().context("failed to show cursor")
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    summary: Option<&ModelSummary>,
) -> Result<()> {
    let mut scroll = 0usize;
    loop {
        terminal.draw(|frame| draw(frame, summary, scroll))?;

        if event::poll(Duration::from_millis(200))? {
            match event::read()? {
                Event::Key(key) if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) => break,
                Event::Key(key) if matches!(key.code, KeyCode::Down | KeyCode::Char('j')) => {
                    scroll = scroll.saturating_add(1);
                }
                Event::Key(key) if matches!(key.code, KeyCode::Up | KeyCode::Char('k')) => {
                    scroll = scroll.saturating_sub(1);
                }
                Event::Key(key) if matches!(key.code, KeyCode::PageDown) => {
                    scroll = scroll.saturating_add(10);
                }
                Event::Key(key) if matches!(key.code, KeyCode::PageUp) => {
                    scroll = scroll.saturating_sub(10);
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn draw(frame: &mut ratatui::Frame<'_>, summary: Option<&ModelSummary>, scroll: usize) {
    let area = frame.area();
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Min(8),
            Constraint::Length(3),
        ])
        .split(area);

    let title = Line::from(vec![
        Span::styled(
            "Candlebench",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::raw("q/Esc quit, j/k scroll"),
    ]);

    let stats = match summary {
        Some(summary) => summary_stats(summary),
        None => vec![
            Line::from("Open a file with: candlebench tui ./model.safetensors"),
            Line::from("Supported files: .safetensors, .gguf"),
        ],
    };
    let header = Paragraph::new(stats)
        .block(Block::default().title(title).borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    frame.render_widget(header, root[0]);

    match summary {
        Some(summary) => draw_summary(frame, root[1], summary, scroll),
        None => {
            let commands = vec![
                ListItem::new("inspect <path> -- inspect SafeTensors or GGUF metadata"),
                ListItem::new("download --repo <id> <file> -- download from Hugging Face Hub"),
                ListItem::new("tokenize <tokenizer.json> <text> -- inspect tokenizer output"),
                ListItem::new("embed --text <text> -- run BERT-style embedding inference"),
                ListItem::new("similarity --text <a> --text <b> -- compare embeddings"),
                ListItem::new("bench-embed --text <text> -- benchmark embedding throughput"),
                ListItem::new("bench-matmul --backend auto -- benchmark Candle matmul"),
            ];
            frame.render_widget(
                List::new(commands).block(Block::default().title("Commands").borders(Borders::ALL)),
                root[1],
            );
        }
    }

    frame.render_widget(
        Paragraph::new("Use --json on non-TUI commands for machine-readable output.")
            .block(Block::default().borders(Borders::ALL)),
        root[2],
    );
}

fn draw_summary(
    frame: &mut ratatui::Frame<'_>,
    area: ratatui::layout::Rect,
    summary: &ModelSummary,
    scroll: usize,
) {
    let rows = match summary {
        ModelSummary::SafeTensors(summary) => summary
            .tensors
            .iter()
            .skip(scroll)
            .take(area.height.saturating_sub(2) as usize)
            .map(|row| {
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{:<42}", truncate(&row.name, 40)), Style::default()),
                    Span::styled(
                        format!("{:<8}", row.dtype),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::raw(format!(
                        " {:<22} {:>12} {:>10}",
                        truncate(&format_shape(&row.shape), 22),
                        format_number(row.parameters),
                        format_size(row.bytes, DECIMAL)
                    )),
                ]))
            })
            .collect::<Vec<_>>(),
        ModelSummary::Gguf(summary) => summary
            .metadata
            .iter()
            .skip(scroll)
            .take(area.height.saturating_sub(2) as usize)
            .map(|row| {
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{:<34}", truncate(&row.key, 32)), Style::default()),
                    Span::styled(
                        format!("{:<14}", truncate(&row.value_type, 14)),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::raw(truncate(&row.value, 80)),
                ]))
            })
            .collect::<Vec<_>>(),
    };

    let title = match summary {
        ModelSummary::SafeTensors(_) => "Largest Tensors",
        ModelSummary::Gguf(_) => "GGUF Metadata",
    };
    frame.render_widget(
        List::new(rows).block(Block::default().title(title).borders(Borders::ALL)),
        area,
    );
}

fn summary_stats(summary: &ModelSummary) -> Vec<Line<'static>> {
    match summary {
        ModelSummary::SafeTensors(summary) => vec![
            Line::from(format!("file: {}", summary.path)),
            Line::from(format!(
                "format: SafeTensors | size: {} | tensors: {}",
                format_size(summary.file_size_bytes, DECIMAL),
                summary.tensor_count
            )),
            Line::from(format!(
                "tensor data: {} | rough parameters: {}",
                format_size(summary.total_tensor_bytes, DECIMAL),
                format_number(summary.total_parameters)
            )),
        ],
        ModelSummary::Gguf(summary) => vec![
            Line::from(format!("file: {}", summary.path)),
            Line::from(format!(
                "format: GGUF {} | size: {} | tensors: {} | metadata: {}",
                summary.version,
                format_size(summary.file_size_bytes, DECIMAL),
                summary.tensor_count,
                summary.metadata_count
            )),
            Line::from(format!(
                "tensor data: {} | rough parameters: {}",
                format_size(summary.total_tensor_bytes, DECIMAL),
                format_number(summary.total_parameters)
            )),
        ],
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        value.to_string()
    } else {
        let mut out = value.chars().take(max_chars).collect::<String>();
        out.push_str("...");
        out
    }
}
