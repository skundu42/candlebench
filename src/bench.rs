use anyhow::{bail, Result};
use candle_core::Tensor;
use std::hint::black_box;
use std::time::{Duration, Instant};

use crate::backend::{device_for_backend, device_label, Backend};

pub fn run_matmul(size: usize, iters: usize, backend: Backend) -> Result<()> {
    if size == 0 {
        bail!("--size must be greater than 0");
    }

    if iters == 0 {
        bail!("--iters must be greater than 0");
    }

    let device = device_for_backend(backend)?;

    println!("Candle matmul benchmark");
    println!("=======================");
    println!("device: {}", device_label(&device));
    println!("matrix: [{size} x {size}]");
    println!("iters: {iters}");
    println!();

    let a = Tensor::randn(0f32, 1f32, (size, size), &device)?;
    let b = Tensor::randn(0f32, 1f32, (size, size), &device)?;

    // Warmup.
    let warmup = a.matmul(&b)?;
    black_box(warmup.dims());

    let mut elapsed = Duration::ZERO;

    for _ in 0..iters {
        let start = Instant::now();
        let c = a.matmul(&b)?;
        black_box(c.dims());
        elapsed += start.elapsed();
    }

    let avg = elapsed / iters as u32;
    let avg_secs = avg.as_secs_f64();

    // A dense square matmul is roughly 2N^3 floating point operations.
    let flops = 2.0 * (size as f64).powi(3);
    let gflops = (flops / avg_secs) / 1e9;

    println!("total elapsed: {:.3}s", elapsed.as_secs_f64());
    println!("avg / iter: {:.3}ms", avg_secs * 1000.0);
    println!("rough throughput: {:.2} GFLOP/s", gflops);

    Ok(())
}
