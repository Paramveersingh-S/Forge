//! Async DMA scheduler for host-to-device weight transfers.
//!
//! Coordinates the double-buffered prefetch: while the GPU computes on
//! buffer A, the DMA engine asynchronously copies the next layer into
//! buffer B. Uses CUDA streams under the hood (via PyTorch's stream API
//! on the Python side); this module provides the scheduling logic.

use std::sync::atomic::{AtomicUsize, Ordering};

/// DMA transfer state for a single buffer slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    /// Buffer is empty and available for a new transfer.
    Free,
    /// A DMA transfer is in progress to fill this buffer.
    Transferring,
    /// Buffer contains valid layer data, ready for compute.
    Ready,
    /// GPU is actively computing on this buffer.
    InUse,
}

/// A slot in the double-buffer pool.
#[derive(Debug)]
pub struct BufferSlot {
    /// Which layer is currently loaded (or being loaded) in this slot.
    pub layer_idx: Option<usize>,
    /// Current state of this buffer.
    pub state: BufferState,
    /// Size of the buffer in bytes.
    pub size_bytes: usize,
}

/// Round-robin DMA scheduler for N buffer slots.
pub struct DmaScheduler {
    /// The buffer slots.
    slots: Vec<BufferSlot>,
    /// Index of the next slot to use for a new transfer.
    next_slot: AtomicUsize,
}

impl DmaScheduler {
    /// Create a new scheduler with `num_slots` buffer slots.
    pub fn new(num_slots: usize, buffer_size: usize) -> Self {
        let slots = (0..num_slots)
            .map(|_| BufferSlot {
                layer_idx: None,
                state: BufferState::Free,
                size_bytes: buffer_size,
            })
            .collect();

        Self {
            slots,
            next_slot: AtomicUsize::new(0),
        }
    }

    /// Get the next free slot for a DMA transfer. Returns None if all busy.
    pub fn acquire_slot(&mut self) -> Option<usize> {
        let start = self.next_slot.load(Ordering::Relaxed);
        let n = self.slots.len();

        for offset in 0..n {
            let idx = (start + offset) % n;
            if self.slots[idx].state == BufferState::Free {
                self.slots[idx].state = BufferState::Transferring;
                self.next_slot.store((idx + 1) % n, Ordering::Relaxed);
                return Some(idx);
            }
        }
        None
    }

    /// Mark a slot's transfer as complete (Ready for compute).
    pub fn mark_ready(&mut self, slot_idx: usize, layer_idx: usize) {
        self.slots[slot_idx].state = BufferState::Ready;
        self.slots[slot_idx].layer_idx = Some(layer_idx);
    }

    /// Mark a slot as in-use (GPU is computing on it).
    pub fn mark_in_use(&mut self, slot_idx: usize) {
        self.slots[slot_idx].state = BufferState::InUse;
    }

    /// Release a slot after compute is done.
    pub fn release(&mut self, slot_idx: usize) {
        self.slots[slot_idx].state = BufferState::Free;
        self.slots[slot_idx].layer_idx = None;
    }

    /// Get the number of slots.
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Check if a specific layer is already loaded in any slot.
    pub fn find_layer(&self, layer_idx: usize) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| s.layer_idx == Some(layer_idx) && s.state == BufferState::Ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_acquire_release() {
        let mut sched = DmaScheduler::new(2, 1024);

        // Acquire first slot
        let slot0 = sched.acquire_slot().unwrap();
        assert_eq!(slot0, 0);

        // Acquire second slot
        let slot1 = sched.acquire_slot().unwrap();
        assert_eq!(slot1, 1);

        // No more free slots
        assert!(sched.acquire_slot().is_none());

        // Release and re-acquire
        sched.release(slot0);
        let slot = sched.acquire_slot().unwrap();
        assert_eq!(slot, 0);
    }

    #[test]
    fn test_scheduler_find_layer() {
        let mut sched = DmaScheduler::new(2, 1024);

        let slot = sched.acquire_slot().unwrap();
        sched.mark_ready(slot, 5);

        assert_eq!(sched.find_layer(5), Some(0));
        assert_eq!(sched.find_layer(3), None);
    }
}
