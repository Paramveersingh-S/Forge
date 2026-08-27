//! Forge Core — Rust acceleration layer for LLM fine-tuning.
//!
//! This crate provides three performance-critical subsystems:
//!
//! 1. **Layer Streaming I/O** (`stream`): Memory-mapped safetensors with async
//!    DMA scheduling and pinned buffer management. Streams one decoder layer at
//!    a time from host RAM (or NVMe) into a GPU VRAM buffer pool, bounding peak
//!    VRAM to a single layer rather than the whole model.
//!
//! 2. **Safetensors Parser** (`safetensors`): Zero-copy deserialization of
//!    safetensors files, reading tensor metadata without loading weight data.
//!
//! 3. **Crypto Engine** (`crypto`): ed25519 detached signatures for adapter
//!    weights and SHA-256 content-addressable checksums.
//!
//! All public APIs are exposed to Python via PyO3.

use pyo3::prelude::*;

pub mod crypto;
pub mod safetensors;
pub mod stream;

/// Register the forge_core Python module.
#[pymodule]
fn forge_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize Rust logging → Python logging bridge
    pyo3_log::init();

    // Register submodules
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // Stream submodule
    let stream_mod = PyModule::new_bound(m.py(), "stream")?;
    stream::register_module(&stream_mod)?;
    m.add_submodule(&stream_mod)?;

    // Safetensors submodule
    let st_mod = PyModule::new_bound(m.py(), "safetensors")?;
    safetensors::register_module(&st_mod)?;
    m.add_submodule(&st_mod)?;

    // Crypto submodule
    let crypto_mod = PyModule::new_bound(m.py(), "crypto")?;
    crypto::register_module(&crypto_mod)?;
    m.add_submodule(&crypto_mod)?;

    Ok(())
}

/// Return the forge-core version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
