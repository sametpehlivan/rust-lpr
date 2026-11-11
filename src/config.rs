use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, env, path::{Path, PathBuf}};

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub api_bind: String,
    #[serde(default = "default_ffmpeg")]
    pub ffmpeg_path: String,
    pub tcp_destination: String,
    #[serde(default = "default_decoder_threads")]
    pub ffmpeg_decoder_threads: usize,
    #[serde(default = "default_filter_threads")]
    pub ffmpeg_filter_threads: usize,
    #[serde(default = "default_tcp_queue")]
    pub tcp_queue_capacity: usize,
    #[serde(default = "default_frame_queue")]
    pub frame_queue_capacity: usize,
    pub inference: InferenceConfig,
    #[serde(default)]
    pub cameras: Vec<CameraConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
    pub plate_detector: String,
    pub plate_ocr: String,
    #[serde(default = "default_cpu")]
    pub device: String,
    #[serde(default = "default_one")]
    pub worker_count: usize,
    #[serde(default = "default_two")]
    pub intra_threads_per_worker: usize,
    #[serde(default = "default_plate_threshold")]
    pub plate_threshold: f32,
    #[serde(default = "default_ocr_threshold")]
    pub ocr_threshold: f32,
    #[serde(default = "default_ten")]
    pub max_plates_per_frame: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub id: String,
    pub rtsp_url: String,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default = "default_fps")]
    pub fps: u32,
    #[serde(default)]
    pub roi: Option<[f32; 4]>,
    #[serde(default)]
    pub polygon: Vec<[f32; 2]>,
    /// Per-camera detector and OCR session overrides. Missing values use
    /// the defaults that the previous low-resource multi-process build used.
    #[serde(default)]
    pub onnx_sessions: CameraSessionOverrides,
    #[serde(default = "default_seven")]
    pub max_track_age_frames: u32,
    #[serde(default = "default_five")]
    pub max_track_idle_seconds: u64,
}

/// ONNX Runtime thread count: 0 means ORT chooses the default automatically.
/// 1 means a single inference thread; >1 permits intra-op parallelism.
#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub intra_threads: usize,
    pub inter_threads: usize,
    pub parallel_execution: bool,
    pub optimization_level: String,
    pub intra_op_spinning: bool,
    pub inter_op_spinning: bool,
    pub memory_pattern: bool,
    pub cpu_mem_arena: bool,
}

impl SessionOptions {
    pub fn detector_default() -> Self {
        Self {
            intra_threads: 4, inter_threads: 1, parallel_execution: false,
            optimization_level: "all".to_owned(), intra_op_spinning: false,
            inter_op_spinning: false, memory_pattern: true, cpu_mem_arena: true,
        }
    }
    pub fn ocr_default() -> Self {
        Self { intra_threads: 1, ..Self::detector_default() }
    }
    fn apply(&mut self, overrides: &SessionOptionOverrides) {
        if let Some(v) = overrides.intra_threads { self.intra_threads = v; }
        if let Some(v) = overrides.inter_threads { self.inter_threads = v; }
        if let Some(v) = overrides.parallel_execution { self.parallel_execution = v; }
        if let Some(v) = &overrides.optimization_level { self.optimization_level = v.clone(); }
        if let Some(v) = overrides.intra_op_spinning { self.intra_op_spinning = v; }
        if let Some(v) = overrides.inter_op_spinning { self.inter_op_spinning = v; }
        if let Some(v) = overrides.memory_pattern { self.memory_pattern = v; }
        if let Some(v) = overrides.cpu_mem_arena { self.cpu_mem_arena = v; }
    }
    fn validate(&self, camera: &str, which: &str) -> Result<()> {
        if self.intra_threads > 256 || self.inter_threads == 0 || self.inter_threads > 256 {
            bail!("camera {camera} {which}: intra_threads must be 0..=256 and inter_threads must be 1..=256");
        }
        if !["all", "level3", "level2", "level1", "disabled"].contains(&self.optimization_level.to_ascii_lowercase().as_str()) {
            bail!("camera {camera} {which}: invalid optimization_level {:?}", self.optimization_level);
        }
        Ok(())
    }
}

/// Optional overrides: omitting a field retains the prior default.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionOptionOverrides {
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
    pub parallel_execution: Option<bool>,
    pub optimization_level: Option<String>,
    pub intra_op_spinning: Option<bool>,
    pub inter_op_spinning: Option<bool>,
    pub memory_pattern: Option<bool>,
    pub cpu_mem_arena: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraSessionOverrides {
    pub detector: SessionOptionOverrides,
    pub ocr: SessionOptionOverrides,
}

impl CameraSessionOverrides {
    pub fn resolved(&self) -> (SessionOptions, SessionOptions) {
        let mut detector = SessionOptions::detector_default();
        detector.apply(&self.detector);
        let mut ocr = SessionOptions::ocr_default();
        ocr.apply(&self.ocr);
        (detector, ocr)
    }
    pub fn validate(&self, camera: &str) -> Result<()> {
        let (detector, ocr) = self.resolved();
        detector.validate(camera, "detector")?;
        ocr.validate(camera, "ocr")?;
        Ok(())
    }
}

/// Load .env from the config file's directory. An explicitly provided
/// NETLPR_ENV_FILE overrides discovery; environment takes priority over .env.
pub fn load_environment(config_path: &Path) -> Result<()> {
    let env_file = if let Some(custom) = env::var_os("NETLPR_ENV_FILE") {
        PathBuf::from(custom)
    } else {
        let absolute_config = std::fs::canonicalize(config_path)
            .with_context(|| format!("cannot resolve config path: {}", config_path.display()))?;
        absolute_config.parent().context("config path has no directory")?.join(".env")
    };
    if env_file.is_file() {
        dotenvy::from_path(&env_file)
            .with_context(|| format!("cannot parse environment file: {}", env_file.display()))?;
    } else if env::var_os("NETLPR_ENV_FILE").is_some() {
        bail!("NETLPR_ENV_FILE not found: {}", env_file.display());
    }
    Ok(())
}

/// A simple command name (ffmpeg/ffmpeg.exe) uses PATH. Explicit relative
/// paths are resolved relative to the config, not the shell's working dir.
fn resolve_executable(value: &str, config_dir: &Path) -> String {
    let p = Path::new(value);
    if p.is_absolute() { return value.to_owned(); }
    if p.components().count() == 1 && !value.starts_with('.') { return value.to_owned(); }
    config_dir.join(p).to_string_lossy().into_owned()
}

fn resolve_model(value: &str, config_dir: &Path) -> String {
    let p = Path::new(value);
    if p.is_absolute() { value.to_owned() }
    else { config_dir.join(p).to_string_lossy().into_owned() }
}

fn default_frame_queue() -> usize { 2 }
fn default_decoder_threads() -> usize { 1 }
fn default_filter_threads() -> usize { 1 }
fn default_ffmpeg() -> String { "ffmpeg".into() }
fn default_cpu() -> String { "cpu".into() }
fn default_one() -> usize { 1 }
fn default_two() -> usize { 1 }
fn default_ten() -> usize { 10 }
fn default_seven() -> u32 { 7 }
fn default_five() -> u64 { 5 }
fn default_tcp_queue() -> usize { 1024 }
fn default_width() -> u32 { 1280 }
fn default_height() -> u32 { 720 }
fn default_fps() -> u32 { 5 }
fn default_plate_threshold() -> f32 { 0.5 }
fn default_ocr_threshold() -> f32 { 0.8 }

impl AppConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).with_context(|| format!("config unreadable: {}", path.display()))?;
        let mut config: Self = serde_json::from_str(&content).context("invalid JSON configuration")?;
        let full_config_path = std::fs::canonicalize(path)
            .with_context(|| format!("cannot resolve config path: {}", path.display()))?;
        let config_dir = full_config_path.parent().context("config path has no directory")?;
        // Environment overrides machine-specific paths only. Session settings
        // remain in JSON, per camera.
        if let Some(ffmpeg) = env::var_os("RUST_LPR_FFMPEG_PATH") {
            config.ffmpeg_path = ffmpeg.to_string_lossy().into_owned();
        }
        config.ffmpeg_path = resolve_executable(&config.ffmpeg_path, config_dir);
        config.inference.plate_detector = resolve_model(&config.inference.plate_detector, config_dir);
        config.inference.plate_ocr = resolve_model(&config.inference.plate_ocr, config_dir);
        // The child inherits an absolute ORT library path when its .env uses
        // a relative path. Absolute system variables are unchanged.
        if let Some(ort) = env::var_os("ORT_DYLIB_PATH") {
            let p = PathBuf::from(ort);
            if !p.is_absolute() && p.components().count() > 1 {
                // Prefer the configured directory, fallback to working dir.
                let candidate = config_dir.join(&p);
                if candidate.exists() {
                    env::set_var("ORT_DYLIB_PATH", candidate);
                }
            }
        }
        if config.inference.worker_count == 0 || config.inference.worker_count > 32 {
            bail!("inference.worker_count must be 1..=32");
        }
        if config.inference.intra_threads_per_worker == 0 { bail!("intra_threads_per_worker must be positive"); }
        if config.tcp_queue_capacity == 0 { bail!("tcp_queue_capacity must be positive"); }
        if !(1..=16).contains(&config.ffmpeg_decoder_threads) { bail!("ffmpeg_decoder_threads must be 1..=16"); }
        if !(1..=16).contains(&config.ffmpeg_filter_threads) { bail!("ffmpeg_filter_threads must be 1..=16"); }
        if !(1..=32).contains(&config.frame_queue_capacity) { bail!("frame_queue_capacity must be 1..=32"); }
        // The manager does not allocate sessions. Each camera process gets one
        // dedicated detector+OCR pair regardless of the total camera count.
        if !["cpu", "cuda"].contains(&config.inference.device.as_str()) { bail!("device must be cpu or cuda"); }
        for (key, v) in [("plate_threshold", config.inference.plate_threshold), ("ocr_threshold", config.inference.ocr_threshold)] {
            if !(0.0..=1.0).contains(&v) { bail!("{key} must be between 0 and 1"); }
        }
        let mut ids = HashSet::new();
        for camera in &config.cameras {
            camera.validate()?;
            if !ids.insert(&camera.id) { bail!("duplicate camera id: {}", camera.id); }
        }
        Ok(config)
    }
}

impl CameraConfig {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() || self.id.len() > 80 { bail!("camera ID must be 1..=80 chars"); }
        if !(self.rtsp_url.starts_with("rtsp://") || self.rtsp_url.starts_with("rtsps://")) {
            bail!("camera {} must use an RTSP URL", self.id);
        }
        if self.width == 0 || self.height == 0 || self.width > 3840 || self.height > 2160 || self.fps == 0 || self.fps > 60 {
            bail!("camera {} invalid resolution or fps", self.id);
        }
        if self.max_track_age_frames == 0 || self.max_track_idle_seconds == 0 { bail!("camera {} invalid tracker limits", self.id); }
        if let Some([x, y, w, h]) = self.roi {
            if x < 0.0 || y < 0.0 || w <= 0.0 || h <= 0.0 || x + w > 1.0001 || y + h > 1.0001 {
                bail!("camera {} ROI must be normalized x,y,w,h", self.id);
            }
        }
        if !self.polygon.is_empty() && self.polygon.len() < 3 { bail!("camera {} polygon must contain >=3 points", self.id); }
        if self.polygon.iter().any(|p| p.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))) {
            bail!("camera {} polygon points must be in 0..=1", self.id);
        }
        self.onnx_sessions.validate(&self.id)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_camera_settings_use_previous_defaults() {
        let camera: CameraConfig = serde_json::from_str(r#"{
          "id":"cam1", "rtsp_url":"rtsp://127.0.0.1/test"
        }"#).unwrap();
        let (detector, ocr) = camera.onnx_sessions.resolved();
        assert_eq!(detector.intra_threads, 4);
        assert_eq!(ocr.intra_threads, 1);
        assert!(!detector.intra_op_spinning);
        assert_eq!(detector.optimization_level, "all");
        assert!(ocr.memory_pattern && detector.cpu_mem_arena);
    }

    #[test]
    fn partial_per_camera_override_inherits_defaults() {
        let camera: CameraConfig = serde_json::from_str(r#"{
          "id":"cam2", "rtsp_url":"rtsp://127.0.0.1/test",
          "onnx_sessions":{"detector":{"intra_threads":0,"intra_op_spinning":true},"ocr":{"intra_threads":2}}
        }"#).unwrap();
        let (detector, ocr) = camera.onnx_sessions.resolved();
        assert_eq!(detector.intra_threads, 0);
        assert!(detector.intra_op_spinning);
        assert_eq!(ocr.intra_threads, 2);
        assert!(!ocr.intra_op_spinning);
    }
}
