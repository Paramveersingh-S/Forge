//! Memory-mapped safetensors file access.
//!
//! Uses `memmap2` for zero-copy access to per-layer shard files.
//! Each decoder layer is stored as a separate safetensors file
//! to enable independent mmap'ing and streaming.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MmapError {
    #[error("shard directory not found: {0}")]
    ShardDirNotFound(PathBuf),
    #[error("layer shard not found: layer {layer} at {path}")]
    ShardNotFound { layer: usize, path: PathBuf },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A memory-mapped view of a single layer's safetensors shard.
pub struct LayerMmap {
    /// The memory-mapped region.
    _mmap: Mmap,
    /// Layer index.
    pub layer_idx: usize,
    /// Size in bytes.
    pub size_bytes: usize,
}

impl LayerMmap {
    /// Open and mmap a single layer shard file.
    pub fn open(path: &Path, layer_idx: usize) -> Result<Self, MmapError> {
        let file = File::open(path).map_err(|_| MmapError::ShardNotFound {
            layer: layer_idx,
            path: path.to_path_buf(),
        })?;
        let mmap = unsafe { Mmap::map(&file)? };
        let size_bytes = mmap.len();
        Ok(Self {
            _mmap: mmap,
            layer_idx,
            size_bytes,
        })
    }

    /// Get a read-only slice of the mmap'd data.
    pub fn as_bytes(&self) -> &[u8] {
        &self._mmap
    }
}

/// Manager for all layer shard mmaps.
pub struct ShardStore {
    shard_dir: PathBuf,
    layers: HashMap<usize, LayerMmap>,
}

impl ShardStore {
    /// Open all layer shards in a directory.
    pub fn open(shard_dir: &Path, num_layers: usize) -> Result<Self, MmapError> {
        if !shard_dir.exists() {
            return Err(MmapError::ShardDirNotFound(shard_dir.to_path_buf()));
        }

        let mut layers = HashMap::with_capacity(num_layers);
        for i in 0..num_layers {
            let shard_path = shard_dir.join(format!("layer_{:04}.safetensors", i));
            let layer = LayerMmap::open(&shard_path, i)?;
            layers.insert(i, layer);
        }

        Ok(Self {
            shard_dir: shard_dir.to_path_buf(),
            layers,
        })
    }

    /// Get the total size of all shards.
    pub fn total_bytes(&self) -> usize {
        self.layers.values().map(|l| l.size_bytes).sum()
    }

    /// Get a specific layer's data.
    pub fn get_layer(&self, idx: usize) -> Option<&LayerMmap> {
        self.layers.get(&idx)
    }

    /// Get the shard directory path.
    pub fn shard_dir(&self) -> &Path {
        &self.shard_dir
    }
}
