//! Manager process: one OS child process per camera. Commands go over the
//! child's stdin; pixel buffers never cross process boundaries.
use crate::config::CameraConfig;
use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use std::{collections::HashMap, path::PathBuf, process::Stdio, sync::Arc, time::Duration};
use tokio::{io::AsyncWriteExt, process::{Child, Command}, sync::Mutex, time::timeout};
use tracing::{info, warn};

pub struct CameraProcessManager {
    config_path: PathBuf,
    executable: PathBuf,
    children: Mutex<HashMap<String, CameraChild>>,
}

struct CameraChild {
    config: CameraConfig,
    child: Child,
}

#[derive(Debug, Serialize)]
pub struct CameraProcessStatus {
    pub id: String,
    pub pid: Option<u32>,
    pub state: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

#[derive(Debug, Serialize)]
pub struct StopResult {
    pub id: String,
    pub pid: Option<u32>,
    pub status: &'static str,
    pub forced: bool,
}

impl CameraProcessManager {
    pub fn new(config_path: PathBuf) -> Result<Arc<Self>> {
        let config_path = std::fs::canonicalize(&config_path)
            .with_context(|| format!("cannot resolve config path: {}", config_path.display()))?;
        let executable = std::env::current_exe().context("cannot locate worker executable")?;
        Ok(Arc::new(Self {
            config_path,
            executable,
            children: Mutex::new(HashMap::new()),
        }))
    }

    pub async fn start(&self, camera: CameraConfig) -> Result<CameraProcessStatus> {
        camera.validate()?;
        let mut children = self.children.lock().await;
        if let Some(existing) = children.get_mut(&camera.id) {
            if existing.child.try_wait()?.is_none() {
                bail!("camera {} is already running (pid {:?})", camera.id, existing.child.id());
            }
        }
        // Drop a terminated child to permit restarts with the same camera ID.
        children.remove(&camera.id);

        let mut child = Command::new(&self.executable)
            .arg("--worker")
            .arg("--config").arg(&self.config_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .context("cannot spawn camera worker process")?;

        // Camera configuration is passed in a private pipe, not in process
        // command-line arguments (which could leak RTSP credentials).
        let line = serde_json::to_vec(&camera)?;
        let result = async {
            let stdin = child.stdin.as_mut().context("worker stdin unavailable")?;
            stdin.write_all(&line).await?;
            stdin.write_all(b"\n").await?;
            stdin.flush().await?;
            Ok::<_, anyhow::Error>(())
        }.await;
        if let Err(error) = result {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(error.context("cannot send camera configuration to worker"));
        }
        let status = CameraProcessStatus {
            id: camera.id.clone(),
            pid: child.id(),
            state: "starting".to_string(),
            width: camera.width,
            height: camera.height,
            fps: camera.fps,
        };
        info!(camera = %camera.id, pid = ?child.id(), "camera worker process launched");
        children.insert(camera.id.clone(), CameraChild { config: camera, child });
        Ok(status)
    }

    pub async fn list(&self) -> Result<Vec<CameraProcessStatus>> {
        let mut children = self.children.lock().await;
        let mut statuses = Vec::with_capacity(children.len());
        for entry in children.values_mut() {
            let exited = entry.child.try_wait()?;
            statuses.push(CameraProcessStatus {
                id: entry.config.id.clone(),
                pid: entry.child.id(),
                state: match exited {
                    None => "running".to_string(),
                    Some(code) => format!("exited({code})"),
                },
                width: entry.config.width,
                height: entry.config.height,
                fps: entry.config.fps,
            });
        }
        statuses.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(statuses)
    }

    pub async fn stop(&self, id: &str) -> Result<Option<StopResult>> {
        let entry = self.children.lock().await.remove(id);
        match entry {
            None => Ok(None),
            Some(entry) => Ok(Some(stop_child(entry).await)),
        }
    }

    pub async fn restart(&self, id: &str) -> Result<Option<CameraProcessStatus>> {
        let camera = {
            let children = self.children.lock().await;
            children.get(id).map(|p| p.config.clone())
        };
        match camera {
            None => Ok(None),
            Some(camera) => {
                self.stop(id).await?;
                Ok(Some(self.start(camera).await?))
            }
        }
    }

    pub async fn stop_all_wait(&self) {
        let entries: Vec<_> = {
            let mut children = self.children.lock().await;
            children.drain().map(|(_, entry)| entry).collect()
        };
        let handles: Vec<_> = entries.into_iter().map(|entry| {
            tokio::spawn(async move { stop_child(entry).await })
        }).collect();
        for handle in handles {
            if let Err(error) = handle.await {
                warn!(%error, "worker shutdown join failed");
            }
        }
    }
}

async fn stop_child(mut entry: CameraChild) -> StopResult {
    let id = entry.config.id.clone();
    let pid = entry.child.id();
    if let Ok(Some(code)) = entry.child.try_wait() {
        info!(camera = %id, ?pid, %code, "camera worker already exited");
        return StopResult { id, pid, status: "stopped", forced: false };
    }
    if let Some(mut stdin) = entry.child.stdin.take() {
        if let Err(error) = stdin.write_all(b"STOP\n").await {
            warn!(camera = %id, %error, "could not send graceful STOP");
        }
        // Closing the pipe is a secondary stop signal in the worker.
        drop(stdin);
    }
    match timeout(Duration::from_secs(15), entry.child.wait()).await {
        Ok(Ok(code)) => {
            info!(camera = %id, ?pid, %code, "camera worker stopped gracefully");
            return StopResult { id, pid, status: "stopped", forced: false };
        }
        Ok(Err(error)) => warn!(camera = %id, %error, "worker wait error"),
        Err(_) => warn!(camera = %id, ?pid, "worker ignored STOP; forcing termination"),
    }
    // Windows: kill descendants (in particular ffmpeg.exe) if a worker hangs.
    #[cfg(windows)]
    if let Some(pid) = pid {
        let _ = Command::new("taskkill.exe")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .output().await;
    }
    if let Ok(None) = entry.child.try_wait() {
        let _ = entry.child.kill().await;
    }
    let _ = entry.child.wait().await;
    StopResult { id, pid, status: "stopped", forced: true }
}
