//! One bounded, FIFO frame queue per camera. Never waits in the capture task.
//! On overflow we drop the oldest unprocessed frame to preserve live-video latency.
use crate::rtsp::Frame;
use std::{collections::VecDeque, sync::{Mutex, atomic::{AtomicU64, AtomicUsize, Ordering}}};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

struct QueueState {
    frames: VecDeque<Frame>,
    closed: bool,
}

pub struct FrameQueue {
    state: Mutex<QueueState>,
    notify: Notify,
    capacity: usize,
    depth: AtomicUsize,
    dropped: AtomicU64,
}

impl FrameQueue {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "frame queue capacity must be positive");
        Self {
            state: Mutex::new(QueueState { frames: VecDeque::with_capacity(capacity), closed: false }),
            notify: Notify::new(),
            capacity,
            depth: AtomicUsize::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    /// Nonblocking, FIFO insertion; full queue evicts the oldest frame.
    /// Only call from the producer / RTSP decoder task.
    pub fn push(&self, frame: Frame) -> bool {
        let mut state = self.state.lock().expect("frame queue poisoned");
        if state.closed { return false; }
        if state.frames.len() == self.capacity {
            state.frames.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        state.frames.push_back(frame);
        self.depth.store(state.frames.len(), Ordering::Relaxed);
        drop(state);
        self.notify.notify_one();
        true
    }

    /// Exactly one consumer per camera; returns frames in FIFO order.
    pub async fn pop(&self, stop: &CancellationToken) -> Option<Frame> {
        loop {
            // Register before checking the queue to avoid lost notifications.
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut state = self.state.lock().expect("frame queue poisoned");
                if let Some(frame) = state.frames.pop_front() {
                    self.depth.store(state.frames.len(), Ordering::Relaxed);
                    return Some(frame);
                }
                if state.closed { return None; }
            }
            tokio::select! {
                _ = stop.cancelled() => return None,
                _ = &mut notified => {},
            }
        }
    }

    pub fn close(&self) {
        self.state.lock().expect("frame queue poisoned").closed = true;
        self.notify.notify_waiters();
    }

    pub fn depth(&self) -> usize { self.depth.load(Ordering::Relaxed) }
    pub fn dropped(&self) -> u64 { self.dropped.load(Ordering::Relaxed) }
    pub fn capacity(&self) -> usize { self.capacity }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use image::RgbImage;
    fn frame(sequence: u64) -> Frame {
        Frame { image: RgbImage::new(1, 1), timestamp: Utc::now(), sequence }
    }
    #[tokio::test]
    async fn full_queue_evicts_oldest_but_preserves_fifo() {
        let q = FrameQueue::new(2);
        q.push(frame(1)); q.push(frame(2)); q.push(frame(3));
        let token = CancellationToken::new();
        assert_eq!(q.dropped(), 1);
        assert_eq!(q.pop(&token).await.unwrap().sequence, 2);
        assert_eq!(q.pop(&token).await.unwrap().sequence, 3);
        assert_eq!(q.depth(), 0);
    }
    #[tokio::test]
    async fn closed_queue_wakes_consumer() {
        let q = FrameQueue::new(2);
        q.close();
        assert!(q.pop(&CancellationToken::new()).await.is_none());
    }
}
