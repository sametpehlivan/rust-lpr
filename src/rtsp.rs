//! Low-memory RTSP decoding through FFmpeg CLI. One producer per camera
//! continuously reads complete raw frames and writes them to a bounded FIFO.
use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use image::RgbImage;
use std::{process::Stdio, sync::{Arc, atomic::{AtomicU64, Ordering}}, time::{Duration, Instant}};
use tokio::{io::AsyncReadExt, process::Command, time::timeout};
use tokio_util::sync::CancellationToken;
use crate::{config::CameraConfig, frame_queue::FrameQueue};

pub struct Frame {
    pub image: RgbImage,
    pub timestamp: DateTime<Utc>,
    pub sequence: u64,
}

pub async fn capture_loop(
    camera: CameraConfig,
    ffmpeg_path: String,
    queue: Arc<FrameQueue>,
    stop: CancellationToken,
    frames_received: Arc<AtomicU64>,
    read_latest_ms: Arc<AtomicU64>,
    read_max_ms: Arc<AtomicU64>,
    decoder_threads: usize,
    filter_threads: usize,
) {
    loop {
        if stop.is_cancelled() {break;}
        match capture_once(&camera, &ffmpeg_path, &queue, &stop, &frames_received, &read_latest_ms, &read_max_ms, decoder_threads, filter_threads).await {
            Ok(()) => tracing::info!(camera=%camera.id,"RTSP capture stopped"),
            Err(error) => tracing::warn!(camera=%camera.id,error=%error,"RTSP disconnected, reconnecting"),
        }
        tokio::select!{ _=stop.cancelled()=>break, _=tokio::time::sleep(Duration::from_secs(3))=>{} }
    }
}

async fn capture_once(
    camera: &CameraConfig,
    ffmpeg_path: &str,
    queue: &FrameQueue,
    stop: &CancellationToken,
    frames_received: &AtomicU64,
    read_latest_ms: &AtomicU64,
    read_max_ms: &AtomicU64,
    decoder_threads: usize,
    filter_threads: usize,
) -> Result<()> {
    let filters=format!("fps={},scale={}:{}:flags=bilinear",camera.fps,camera.width,camera.height);
    let decode_threads=decoder_threads.to_string();
    let filter_threads=filter_threads.to_string();
    // IMPORTANT: decoder thread setting is an input option (before -i),
    // filter_threads is a global FFmpeg option. 4 default auto-sized FFmpeg
    // processes + 4 ONNX session pairs can oversubscribe a small CPU.
    let mut child=Command::new(ffmpeg_path)
        .args(["-nostdin","-hide_banner","-loglevel","error",
            "-filter_threads",&filter_threads,"-rtsp_transport","tcp",
            "-timeout","5000000","-threads:v",&decode_threads,
            "-i",&camera.rtsp_url,"-an","-sn","-dn",
            "-vf",&filters,"-pix_fmt","rgb24", "-threads:v", "1", "-f","rawvideo","pipe:1"])
        .stdout(Stdio::piped()).stderr(Stdio::null()).kill_on_drop(true)
        .spawn().with_context(||format!("could not launch {ffmpeg_path}"))?;
    let mut stdout=child.stdout.take().context("FFmpeg stdout missing")?;
    let len=(camera.width as usize).checked_mul(camera.height as usize)
        .and_then(|v|v.checked_mul(3)).context("frame dimensions overflow")?;
    let mut failure=None;
    loop {
        let mut raw=vec![0u8;len];
        let began=Instant::now();
        let read=tokio::select! {
            _=stop.cancelled()=>break,
            result=timeout(Duration::from_secs(12),stdout.read_exact(&mut raw))=>result,
        };
        match read {
            Ok(Ok(_)) => {
                // Time until a full RGB frame arrived from FFmpeg stdout.
                // Includes waiting for next frame at the configured output FPS.
                let read_ms=began.elapsed().as_millis().min(u64::MAX as u128) as u64;
                read_latest_ms.store(read_ms,Ordering::Relaxed);
                read_max_ms.fetch_max(read_ms,Ordering::Relaxed);
                let seq=frames_received.fetch_add(1,Ordering::Relaxed)+1;
                let image=RgbImage::from_raw(camera.width,camera.height,raw)
                    .ok_or_else(||anyhow!("invalid RGB frame byte count"))?;
                if !queue.push(Frame {image,timestamp:Utc::now(),sequence:seq}) { break; }
            }
            Ok(Err(e)) => {failure=Some(anyhow!(e).context("FFmpeg stream ended"));break;}
            Err(_) => {failure=Some(anyhow!("FFmpeg produced no full frame in 12 seconds"));break;}
        }
    }
    let _=child.kill().await;
    let _=child.wait().await;
    if let Some(e)=failure {Err(e)} else {Ok(())}
}
