//! Zero-copy safetensors parser.
//!
//! Reads the safetensors header (JSON metadata) without loading weight
//! data into memory. Used by the layer streaming engine to discover
//! tensor shapes, dtypes, and byte offsets before any I/O.

use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SafetensorsError {
    #[error("invalid header: {0}")]
    InvalidHeader(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug, Clone, Deserialize)]
#[pyclass]
pub struct TensorInfo {
    /// Data type (e.g. "F32", "BF16", "F16").
    #[pyo3(get)]
    pub dtype: String,
    /// Shape dimensions.
    #[pyo3(get)]
    pub shape: Vec<usize>,
    /// Byte offset range [start, end) in the data section.
    #[pyo3(get)]
    pub data_offsets: (usize, usize),
}

/// Parse safetensors header metadata without loading weight data.
#[pyfunction]
pub fn parse_header(path: &str) -> PyResult<HashMap<String, TensorInfo>> {
    let header = read_header(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(header)
}

/// Read the JSON header from a safetensors file.
fn read_header(path: &Path) -> Result<HashMap<String, TensorInfo>, SafetensorsError> {
    let mut file = File::open(path)?;

    // First 8 bytes: header size as u64 little-endian
    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf) as usize;

    if header_size > 100_000_000 {
        return Err(SafetensorsError::InvalidHeader(format!(
            "header size {} exceeds 100 MB limit",
            header_size
        )));
    }

    // Read the JSON header
    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf)?;

    let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_buf)?;

    let mut tensors = HashMap::new();
    for (name, value) in raw {
        // Skip the "__metadata__" key
        if name == "__metadata__" {
            continue;
        }
        let info: TensorInfo = serde_json::from_value(value)?;
        tensors.insert(name, info);
    }

    Ok(tensors)
}

/// Compute the total size in bytes of all tensors in a safetensors file.
#[pyfunction]
pub fn file_tensor_bytes(path: &str) -> PyResult<usize> {
    let header = read_header(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let total: usize = header
        .values()
        .map(|t| t.data_offsets.1 - t.data_offsets.0)
        .sum();
    Ok(total)
}

/// Register this module's functions with the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TensorInfo>()?;
    m.add_function(wrap_pyfunction!(parse_header, m)?)?;
    m.add_function(wrap_pyfunction!(file_tensor_bytes, m)?)?;
    Ok(())
}
