use anyhow::{bail, Result};
use candle_core::Device;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
    Auto,
}

impl Backend {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "auto" => Ok(Self::Auto),
            _ => bail!("backend must be one of: cpu, metal, auto"),
        }
    }
}

pub fn device_for_backend(backend: Backend) -> Result<Device> {
    match backend {
        Backend::Cpu => Ok(Device::Cpu),
        Backend::Metal => Device::new_metal(0).map_err(Into::into),
        Backend::Auto => Device::metal_if_available(0).map_err(Into::into),
    }
}

pub fn device_label(device: &Device) -> &'static str {
    if device.is_metal() {
        "Metal"
    } else if device.is_cuda() {
        "CUDA"
    } else {
        "CPU"
    }
}
