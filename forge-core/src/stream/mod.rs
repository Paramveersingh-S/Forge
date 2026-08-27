//! Layer streaming I/O engine.
//!
//! Streams frozen base-model weights from host memory (RAM or NVMe) into GPU
//! VRAM one decoder layer at a time. The design ensures peak VRAM is bounded
//! by a single layer rather than the entire model.
//!
//! ## Architecture
//!
//! ```text
//! ┌─ Host ─────────────────────────────────┐
//! │  mmap'd safetensors (per-layer shards)  │
//! │        ↓ memcpy (pinned)                │
//! │  Pinned staging buffer (page-locked)    │
//! └────────┬───────────────────────────────┘
//!          │  cudaMemcpyAsync (CUDA stream)
//!          ▼
//! ┌─ Device ───────────────────────────────┐
//! │  Buffer A ←── compute this layer       │
//! │  Buffer B ←── prefetch next layer      │
//! │  LoRA adapters (resident)              │
//! └────────────────────────────────────────┘
//! ```

pub mod buffer_pool;
pub mod dma;
pub mod mmap;

use pyo3::prelude::*;

/// Configuration for the layer streaming engine.
#[pyclass]
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Path to the directory containing per-layer safetensors shards.
    #[pyo3(get, set)]
    pub shard_dir: String,

    /// Number of VRAM buffers in the pool (2 = double-buffered).
    #[pyo3(get, set)]
    pub num_buffers: usize,

    /// Whether to use page-locked (pinned) host memory.
    #[pyo3(get, set)]
    pub pin_memory: bool,

    /// Stream source tier: "ram" or "disk".
    #[pyo3(get, set)]
    pub source_tier: String,

    /// Total number of decoder layers in the model.
    #[pyo3(get, set)]
    pub num_layers: usize,

    /// Size in bytes of the largest single layer.
    #[pyo3(get, set)]
    pub max_layer_bytes: usize,
}

#[pymethods]
impl StreamConfig {
    #[new]
    #[pyo3(signature = (shard_dir, num_layers, max_layer_bytes, num_buffers=2, pin_memory=true, source_tier="ram".to_string()))]
    fn new(
        shard_dir: String,
        num_layers: usize,
        max_layer_bytes: usize,
        num_buffers: usize,
        pin_memory: bool,
        source_tier: String,
    ) -> Self {
        Self {
            shard_dir,
            num_buffers,
            pin_memory,
            source_tier,
            num_layers,
            max_layer_bytes,
        }
    }
}

/// Layer streaming engine state.
#[pyclass]
pub struct StreamEngine {
    config: StreamConfig,
    // In a full implementation, this would hold:
    // - mmap'd file handles
    // - pinned host buffers
    // - CUDA stream handles
    // - buffer pool state
}

#[pymethods]
impl StreamEngine {
    /// Create a new streaming engine from config.
    #[new]
    fn new(config: StreamConfig) -> PyResult<Self> {
        log::info!(
            "Initializing layer streaming engine: {} layers, {} buffers, tier={}",
            config.num_layers,
            config.num_buffers,
            config.source_tier
        );
        Ok(Self { config })
    }

    /// Estimate the host RAM required for the full model store.
    fn estimate_host_ram_bytes(&self) -> usize {
        self.config.num_layers * self.config.max_layer_bytes
    }

    /// Estimate the VRAM required for the buffer pool.
    fn estimate_vram_bytes(&self) -> usize {
        self.config.num_buffers * self.config.max_layer_bytes
    }

    /// Return the stream config.
    fn get_config(&self) -> StreamConfig {
        self.config.clone()
    }

    /// Asynchronously (or synchronously for now) transfer layer data into a PyTorch pinned host buffer.
    ///
    /// This method bridges the Python tensor to the Rust core via a raw pointer.
    /// In a production system, this would trigger an async DMA copy from the mmap'd
    /// safetensors shard directly into the pinned host memory.
    #[pyo3(signature = (ptr, size_bytes, layer_idx))]
    fn transfer_to_ptr(&self, ptr: usize, size_bytes: usize, layer_idx: usize) -> PyResult<()> {
        log::debug!(
            "Rust FFI transfer: loading layer {} into host ptr {:#x} ({} bytes)",
            layer_idx,
            ptr,
            size_bytes
        );

        let path = std::path::PathBuf::from(&self.config.shard_dir)
            .join(format!("layer_{}.safetensors", layer_idx));

        // Use unsafe Rust to copy data into the provided tensor memory.
        // This memory must be page-locked (pinned) by PyTorch on the caller side.
        let dest_ptr = ptr as *mut u8;
        if path.exists() {
            let file = std::fs::File::open(&path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))? };
            unsafe {
                std::ptr::copy_nonoverlapping(mmap.as_ptr(), dest_ptr, std::cmp::min(size_bytes, mmap.len()));
            }
        } else {
            unsafe {
                std::ptr::write_bytes(dest_ptr, 0, size_bytes);
            }
        }

        Ok(())
    }
}

/// Register this module's classes and functions with the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StreamConfig>()?;
    m.add_class::<StreamEngine>()?;
    Ok(())
}
