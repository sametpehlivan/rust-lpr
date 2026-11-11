mod api;
mod camera;
mod config;
mod domain;
mod frame_queue;
mod inference;
mod process_manager;
mod rtsp;
mod tracking;
mod transport;

use anyhow::{bail, Context, Result};
use config::{AppConfig, CameraConfig};
use inference::InferenceEngine;
use std::{path::Path, time::Duration};
use tokio::{io::{AsyncBufReadExt, BufReader}, net::TcpListener, sync::mpsc};
use tracing::info;

// Async I/O, TCP and control run on one Tokio scheduler thread per process.
// Each camera's synchronous inference runs on exactly one dedicated OS thread.
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let worker_mode = args.iter().any(|a| a == "--worker");
    let config_arg = if let Some(pos) = args.iter().position(|a| a == "--config") {
        args.get(pos + 1).cloned().context("--config needs a file path")?
    } else {
        args.iter().find(|a| !a.starts_with('-'))
            .cloned().unwrap_or_else(|| "config.json".to_owned())
    };
    let config_path = Path::new(&config_arg);
    // Load the project's .env, regardless of the caller's working directory.
    // Actual environment variables take priority over .env values.
    config::load_environment(config_path)?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "rustlpr=info,warn".into()))
        .init();
    let config = AppConfig::load(config_path)?;

    if worker_mode {
        run_worker(config).await
    } else {
        run_manager(config_path, config).await
    }
}

async fn run_manager(config_path: &Path, config: AppConfig) -> Result<()> {
    info!(cameras=config.cameras.len(), "NetLpr manager starting; process-per-camera");
    let manager = process_manager::CameraProcessManager::new(config_path.to_path_buf())?;
    for cam in config.cameras {
        let status = manager.start(cam).await?;
        info!(camera=%status.id, pid=?status.pid, "camera child started");
    }
    let listener = TcpListener::bind(&config.api_bind).await
        .with_context(|| format!("cannot bind management API on {}", config.api_bind))?;
    info!(address=%config.api_bind, "camera management API ready");
    let result = axum::serve(listener, api::router(manager.clone()))
        .with_graceful_shutdown(async { let _ = tokio::signal::ctrl_c().await; })
        .await;
    manager.stop_all_wait().await;
    result?;
    Ok(())
}

async fn run_worker(config: AppConfig) -> Result<()> {
    // Camera URI is sent over stdin so RTSP credentials are not command-line args.
    let mut reader = BufReader::new(tokio::io::stdin());
    let mut first_line = String::new();
    if reader.read_line(&mut first_line).await? == 0 {
        bail!("worker did not receive camera config from parent");
    }
    let cam: CameraConfig = serde_json::from_str(&first_line)
        .context("invalid camera config on worker stdin")?;
    cam.validate()?;
    let camera_id = cam.id.clone();

    // Absolutely one session pair per process. Ignore old worker_count:6.
    // Inference remains one detector + one OCR session per camera process.
    // Session tunables come from this camera's config, not a fixed global value.
    let inference = InferenceEngine::load(config.inference.clone(), &cam.onnx_sessions)?;
    let (events, receiver) = mpsc::channel(config.tcp_queue_capacity);
    let transport_task = tokio::spawn(transport::run_tcp_sender(config.tcp_destination.clone(), receiver));
    let camera_run = camera::start_camera(
        cam, inference, events.clone(), config.ffmpeg_path,
        config.frame_queue_capacity, config.ffmpeg_decoder_threads,
        config.ffmpeg_filter_threads,
    )?;
    info!(camera=%camera_id, pid=std::process::id(),
        decoder_threads=config.ffmpeg_decoder_threads, filter_threads=config.ffmpeg_filter_threads, onnx_sessions=2,
        "camera process running (dedicated inference thread)");

    // A parent's STOP or closed stdin terminates only this camera process.
    loop {
        let mut line = String::new();
        tokio::select! {
            result = reader.read_line(&mut line) => {
                match result {
                    Ok(0) => { info!(camera=%camera_id, "parent pipe closed"); break; }
                    Ok(_) if line.trim().eq_ignore_ascii_case("STOP") => break,
                    Ok(_) => {},
                    Err(error) => { tracing::warn!(%error, "parent pipe error"); break; }
                }
            }
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    camera_run.shutdown().await;
    drop(events);
    // Prevent a stopped camera from hanging indefinitely on a TCP server.
    if tokio::time::timeout(Duration::from_secs(5), transport_task).await.is_err() {
        tracing::warn!(camera=%camera_id, "TCP sender did not finish within 5 seconds");
    }
    info!(camera=%camera_id, "camera worker stopped");
    Ok(())
}
