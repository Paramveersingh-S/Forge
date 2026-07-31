//! VRAM buffer pool management.
//!
//! Pre-allocates a fixed number of GPU VRAM buffers at engine init time,
//! each sized to hold one decoder layer. This avoids runtime allocation
//! jitter and ensures the VRAM ceiling is deterministic.

/// Metadata for a pre-allocated VRAM buffer.
#[derive(Debug, Clone)]
pub struct VramBuffer {
    /// Buffer index in the pool.
    pub index: usize,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Whether this buffer is currently allocated on device.
    pub is_allocated: bool,
}

/// Pool of pre-allocated VRAM buffers.
pub struct BufferPool {
    buffers: Vec<VramBuffer>,
    buffer_size: usize,
}

impl BufferPool {
    /// Create a new buffer pool with `count` buffers of `size` bytes each.
    pub fn new(count: usize, size: usize) -> Self {
        let buffers = (0..count)
            .map(|i| VramBuffer {
                index: i,
                size_bytes: size,
                is_allocated: false,
            })
            .collect();

        Self {
            buffers,
            buffer_size: size,
        }
    }

    /// Total VRAM required for the pool.
    pub fn total_vram_bytes(&self) -> usize {
        self.buffers.len() * self.buffer_size
    }

    /// Number of buffers.
    pub fn count(&self) -> usize {
        self.buffers.len()
    }

    /// Per-buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Mark a buffer as allocated on device.
    pub fn mark_allocated(&mut self, index: usize) {
        if index < self.buffers.len() {
            self.buffers[index].is_allocated = true;
        }
    }

    /// Mark a buffer as freed.
    pub fn mark_freed(&mut self, index: usize) {
        if index < self.buffers.len() {
            self.buffers[index].is_allocated = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = BufferPool::new(2, 1024 * 1024); // 2 x 1MB
        assert_eq!(pool.count(), 2);
        assert_eq!(pool.buffer_size(), 1024 * 1024);
        assert_eq!(pool.total_vram_bytes(), 2 * 1024 * 1024);
    }

    #[test]
    fn test_pool_allocation() {
        let mut pool = BufferPool::new(2, 512);
        assert!(!pool.buffers[0].is_allocated);
        pool.mark_allocated(0);
        assert!(pool.buffers[0].is_allocated);
        pool.mark_freed(0);
        assert!(!pool.buffers[0].is_allocated);
    }
}
