pub fn num_elements(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

pub fn format_shape(shape: &[usize]) -> String {
    if shape.is_empty() {
        "[]".to_string()
    } else {
        format!(
            "[{}]",
            shape
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

pub fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();

    for (i, ch) in s.chars().rev().enumerate() {
        if i != 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }

    out.chars().rev().collect()
}
