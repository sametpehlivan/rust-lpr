//! One camera worker process = one async RTSP reader + one dedicated,
//! persistent OS thread that owns detector, OCR and tracker. There is no
//! spawn_blocking per frame, no inference task pool and no session pool.
use crate::{
    config::CameraConfig, domain::TrackingEvent, frame_queue::FrameQueue,
    inference::InferenceEngine, rtsp::capture_loop, tracking::PlateTracker,
};
use anyhow::{Context, Result};
use chrono::Utc;
use std::{
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant}, thread,
};
use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct CameraCounters {
    received: Arc<AtomicU64>,
    ffmpeg_read_latest_ms: Arc<AtomicU64>,
    ffmpeg_read_max_ms: Arc<AtomicU64>,
    analyzed: AtomicU64,
    inference_latest_ms: AtomicU64,
    inference_max_ms: AtomicU64,
    frame_age_latest_ms: AtomicU64,
    events: AtomicU64,
    event_queue_dropped: AtomicU64,
}

pub struct CameraRuntime {
    stop: CancellationToken,
    queue: Arc<FrameQueue>,
    reader_task: JoinHandle<()>,
    inference_thread: thread::JoinHandle<()>,
}

fn emit(sender: &mpsc::Sender<TrackingEvent>, counters: &CameraCounters, completed: Vec<TrackingEvent>) {
    for event in completed {
        if let Err(error) = sender.try_send(event) {
            counters.event_queue_dropped.fetch_add(1, Ordering::Relaxed);
            tracing::error!(%error, "TCP tracking event dropped (queue full or closed)");
        } else {
            counters.events.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn start_camera(
    camera: CameraConfig,
    mut inference: InferenceEngine,
    events: mpsc::Sender<TrackingEvent>,
    ffmpeg: String,
    configured_queue_capacity: usize,
    decoder_threads: usize,
    filter_threads: usize,
) -> Result<CameraRuntime> {
    // Keep at most two full RGB frames in memory, regardless of old config
    // files that used queue=30. Preserve live-video freshness.
    let queue_capacity = configured_queue_capacity.clamp(1, 2);
    let queue = Arc::new(FrameQueue::new(queue_capacity));
    let stop = CancellationToken::new();
    let counters = Arc::new(CameraCounters::default());
    let handle = tokio::runtime::Handle::current();

    let consumer_queue = queue.clone();
    let consumer_stop = stop.clone();
    let consumer_counters = counters.clone();
    let consumer_camera = camera.clone();
    let inference_thread = thread::Builder::new()
        .name(format!("onnx-{}", camera.id))
        .spawn(move || {
            let mut tracker = PlateTracker::new(consumer_camera.clone());
            handle.block_on(async move {
                let mut idle = tokio::time::interval(Duration::from_secs(1));
                let mut metrics = tokio::time::interval_at(
                    tokio::time::Instant::now() + Duration::from_secs(10),
                    Duration::from_secs(10),
                );
                let mut last_log = Instant::now();
                let (mut previous_received, mut previous_analyzed, mut previous_dropped) = (0, 0, 0);
                loop {
                    tokio::select! {
                        biased;
                        _ = consumer_stop.cancelled() => break,
                        _ = idle.tick() => {
                            emit(&events, &consumer_counters, tracker.expire_idle(Utc::now()));
                        }
                        _ = metrics.tick() => {
                            let elapsed = last_log.elapsed().as_secs_f64().max(0.001);
                            last_log = Instant::now();
                            let received = consumer_counters.received.load(Ordering::Relaxed);
                            let analyzed = consumer_counters.analyzed.load(Ordering::Relaxed);
                            let dropped = consumer_queue.dropped();
                            tracing::info!(
                                camera=%consumer_camera.id,
                                pid=std::process::id(),
                                output_fps=(received-previous_received) as f64 / elapsed,
                                inference_fps=(analyzed-previous_analyzed) as f64 / elapsed,
                                queue_drops_10s=dropped-previous_dropped,
                                queue_depth=consumer_queue.depth(),
                                ffmpeg_output_read_ms=consumer_counters.ffmpeg_read_latest_ms.load(Ordering::Relaxed),
                                inference_ms=consumer_counters.inference_latest_ms.load(Ordering::Relaxed),
                                inference_max_ms=consumer_counters.inference_max_ms.load(Ordering::Relaxed),
                                queue_age_ms=consumer_counters.frame_age_latest_ms.load(Ordering::Relaxed),
                                tcp_queue_dropped=consumer_counters.event_queue_dropped.load(Ordering::Relaxed),
                                "camera pipeline performance"
                            );
                            (previous_received, previous_analyzed, previous_dropped) = (received, analyzed, dropped);
                        }
                        frame = consumer_queue.pop(&consumer_stop) => {
                            let Some(frame) = frame else { break; };
                            let age = Utc::now().signed_duration_since(frame.timestamp)
                                .num_milliseconds().max(0) as u64;
                            consumer_counters.frame_age_latest_ms.store(age, Ordering::Relaxed);
                            let started = Instant::now();
                            let result = inference.infer(&frame.image, consumer_camera.roi);
                            let took_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
                            consumer_counters.inference_latest_ms.store(took_ms, Ordering::Relaxed);
                            consumer_counters.inference_max_ms.fetch_max(took_ms, Ordering::Relaxed);
                            match result {
                                Ok(detections) => {
                                    consumer_counters.analyzed.fetch_add(1, Ordering::Relaxed);
                                    emit(&events, &consumer_counters, tracker.update(detections, frame.timestamp));
                                }
                                Err(error) => tracing::warn!(camera=%consumer_camera.id, %error, "ONNX inference failed"),
                            }
                        }
                    }
                }
                emit(&events, &consumer_counters, tracker.flush(Utc::now()));
                tracing::info!(camera=%consumer_camera.id, "inference thread finished");
            });
        })
        .context("cannot spawn dedicated camera inference thread")?;

    // Reading FFmpeg stdout is asynchronous and never waits for inference.
    let reader_queue = queue.clone();
    let reader_stop = stop.child_token();
    let reader_received = counters.received.clone();
    let reader_latest = counters.ffmpeg_read_latest_ms.clone();
    let reader_max = counters.ffmpeg_read_max_ms.clone();
    let reader_task = tokio::spawn(async move {
        capture_loop(
            camera, ffmpeg, reader_queue, reader_stop,
            reader_received, reader_latest, reader_max,
            decoder_threads, filter_threads,
        ).await;
    });

    Ok(CameraRuntime { stop, queue, reader_task, inference_thread })
}

impl CameraRuntime {
    pub async fn shutdown(self) {
        self.stop.cancel();
        self.queue.close();
        // Reader cancels its asynchronous pipe read, kills FFmpeg and reaps it.
        if let Err(error) = self.reader_task.await {
            tracing::warn!(%error, "RTSP reader task ended unexpectedly");
        }
        // A native inference call cannot safely be interrupted mid-kernel.
        // Join on a shutdown-only blocking thread to keep async runtime alive.
        let join = tokio::task::spawn_blocking(move || self.inference_thread.join());
        match join.await {
            Ok(Ok(())) => {},
            Ok(Err(_)) => tracing::error!("dedicated inference thread panicked"),
            Err(error) => tracing::error!(%error, "joining inference thread failed"),
        }
    }
}
