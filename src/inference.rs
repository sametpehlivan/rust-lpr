//! Rust implementation of NetLpr's current PlateBasedPipeline:
//! plate detector (RGB NCHW f32 384x384) -> OCR (RGB NHWC u8 64x128).
//! Model output layout is taken from NetLpr.InferenceOR and ld modelinfo.
use anyhow::{anyhow, bail, Context, Result};
use image::{imageops::{self, FilterType}, RgbImage};
use ort::{session::{Session, builder::GraphOptimizationLevel}, value::TensorRef};
use std::{env, path::{Path, PathBuf}};
use crate::{config::{InferenceConfig, CameraSessionOverrides, SessionOptions}, domain::{BBox, PlateObservation}};

/// Exactly one detector + one OCR session in this camera process.
/// No session pool, no session-per-frame allocation, no shared-session mutex.
pub struct InferenceEngine {
    worker: InferenceWorker,
    settings: InferenceConfig,
}

struct InferenceWorker {
    detector: Session,
    ocr: Session,
    // Reuse the same NCHW float buffer for every detection frame.
    detector_input: Vec<f32>,
}

fn ort_library_name() -> &'static str {
    if cfg!(windows) { "onnxruntime.dll" }
    else if cfg!(target_os = "macos") { "libonnxruntime.dylib" }
    else { "libonnxruntime.so" }
}

fn find_ort_library() -> Result<PathBuf> {
    let lib_name = ort_library_name();
    let cwd = env::current_dir().context("get working directory")?;
    if let Some(value) = env::var_os("ORT_DYLIB_PATH") {
        let mut path = PathBuf::from(value);
        if path.is_dir() { path = path.join(lib_name); }
        if !path.is_absolute() { path = cwd.join(path); }
        if !path.is_file() {
            bail!("ORT_DYLIB_PATH does not point to an ONNX Runtime library: {}", path.display());
        }
        return Ok(path);
    }
    let mut candidates = Vec::new();
    if let Some(dir) = env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        candidates.push(dir.join(lib_name));
        candidates.push(dir.join("onnxruntime").join("lib").join(lib_name));
    }
    candidates.push(cwd.join(lib_name));
    candidates.push(cwd.join("onnxruntime").join("lib").join(lib_name));
    if let Some(paths) = env::var_os("PATH") {
        for dir in env::split_paths(&paths) { candidates.push(dir.join(lib_name)); }
    }
    if let Some(file) = candidates.into_iter().find(|p| p.is_file()) { return Ok(file); }
    bail!("ONNX Runtime library ({lib_name}) not found. Set ORT_DYLIB_PATH in .env, or place the native library beside the executable / in PATH")
}

impl InferenceEngine {
    pub fn load(settings: InferenceConfig, overrides: &CameraSessionOverrides) -> Result<Self> {
        // Dynamic loading across Windows, Linux and macOS. ORT_DYLIB_PATH
        // overrides the search; otherwise look beside the program, project,
        // and in PATH. On Unix, the loader may also need LD_LIBRARY_PATH.
        let runtime_dll = find_ort_library()?;

        #[cfg(feature = "cuda")]
        let env = if settings.device == "cuda" {
            ort::init_from(&runtime_dll)
                .map_err(|e| anyhow!("load ONNX DLL: {e}"))?
                .with_execution_providers([ort::ep::CUDA::default().build()])
                .build().context("initialize ONNX CUDA environment")?
        } else {
            ort::init_from(&runtime_dll)
                .map_err(|e| anyhow!("load ONNX DLL: {e}"))?
                .build().context("initialize ONNX CPU environment")?
        };
        #[cfg(not(feature = "cuda"))]
        let env = {
            if settings.device == "cuda" {
                bail!("CUDA requested, but binary lacks the 'cuda' feature");
            }
            ort::init_from(&runtime_dll)
                .map_err(|e| anyhow!("load ONNX DLL: {e}"))?
                .build().context("initialize ONNX CPU environment")?
        };

        let (detector_options, ocr_options) = overrides.resolved();
        let create_session = |model_path: &str, options: &SessionOptions| -> Result<Session> {
            let level = match options.optimization_level.to_ascii_lowercase().as_str() {
                "disabled" => GraphOptimizationLevel::Disable,
                "level1" => GraphOptimizationLevel::Level1,
                "level2" => GraphOptimizationLevel::Level2,
                "level3" => GraphOptimizationLevel::Level3,
                _ => GraphOptimizationLevel::All, // validated in AppConfig::load
            };
            let mut builder = Session::builder(&env)
                .map_err(|e| anyhow!("session builder: {e}"))?
                .with_parallel_execution(options.parallel_execution)
                .map_err(|e| anyhow!("parallel execution: {e}"))?
                .with_optimization_level(level)
                .map_err(|e| anyhow!("graph optimization: {e}"))?
                .with_intra_op_spinning(options.intra_op_spinning)
                .map_err(|e| anyhow!("intra spinning: {e}"))?
                .with_inter_op_spinning(options.inter_op_spinning)
                .map_err(|e| anyhow!("inter spinning: {e}"))?
                .with_memory_pattern(options.memory_pattern)
                .map_err(|e| anyhow!("memory pattern: {e}"))?
                .with_inter_threads(options.inter_threads)
                .map_err(|e| anyhow!("inter threads: {e}"))?;
            // Arena is on by default in ORT. Only set an explicit session key
            // when the user elects to disable it for this camera/session.
            if !options.cpu_mem_arena {
                builder = builder.with_config_entry("session.enable_cpu_mem_arena", "0")
                    .map_err(|e| anyhow!("CPU memory arena: {e}"))?;
            }
            if options.intra_threads != 0 {
                builder = builder.with_intra_threads(options.intra_threads)
                    .map_err(|e| anyhow!("intra threads: {e}"))?;
            }
            builder.commit_from_file(model_path)
                .map_err(|e| anyhow!("load model {model_path}: {e}"))
        };
        let detector = create_session(&settings.plate_detector, &detector_options)?;
        let ocr = create_session(&settings.plate_ocr, &ocr_options)?;
        tracing::info!(
            detector_intra_threads=detector_options.intra_threads,
            ocr_intra_threads=ocr_options.intra_threads,
            detector_parallel=detector_options.parallel_execution,
            ocr_parallel=ocr_options.parallel_execution,
            max_plates_per_frame=settings.max_plates_per_frame,
            "camera ONNX sessions initialized with per-camera settings"
        );
        Ok(Self {
            worker: InferenceWorker {
                detector, ocr,
                detector_input: vec![0.0; 3 * 384 * 384],
            },
            settings,
        })
    }

    pub fn infer(&mut self, image: &RgbImage, roi: Option<[f32; 4]>) -> Result<Vec<PlateObservation>> {
        self.worker.infer(image, roi, &self.settings)
    }
}

impl InferenceWorker {
    fn infer(&mut self, original: &RgbImage, roi: Option<[f32;4]>, settings: &InferenceConfig) -> Result<Vec<PlateObservation>> {
        let [rx, ry, rw, rh] = roi.unwrap_or([0.0, 0.0, 1.0, 1.0]);
        let w = original.width();
        let h = original.height();
        let x = ((rx*w as f32).floor() as u32).min(w.saturating_sub(1));
        let y = ((ry*h as f32).floor() as u32).min(h.saturating_sub(1));
        let crop_w = (((rw*w as f32).ceil() as u32).max(1)).min(w-x);
        let crop_h = (((rh*h as f32).ceil() as u32).max(1)).min(h-y);
        // image 0.25 resize requires ImageBuffer; crop_imm returns SubImage,
        // so to_image() makes one ROI copy before the resize buffer.
        let cropped = imageops::crop_imm(original, x, y, crop_w, crop_h).to_image();
        let resized = imageops::resize(&cropped, 384, 384, FilterType::Triangle);
        // NCHW order matches GetPlateDetConfig: normalize RGB uint8 -> f32 [0,1].
        let tensor = &mut self.detector_input;
        for (i, rgb) in resized.pixels().enumerate() {
            tensor[i] = rgb[0] as f32 / 255.0;
            tensor[384*384 + i] = rgb[1] as f32 / 255.0;
            tensor[2*384*384 + i] = rgb[2] as f32 / 255.0;
        }
        let input = TensorRef::from_array_view(([1usize,3,384,384], tensor.as_slice()))?;
        let detections = self.detector.run(ort::inputs![input])?;
        // The original NetLpr detector returns output0: [100,7].
        // Each row: [*, x1, y1, x2, y2, class_id, confidence].
        let (_, raw) = detections[0].try_extract_tensor::<f32>()?;
        if raw.len() % 7 != 0 { bail!("plate detection output has {} values, expected N*7", raw.len()); }
        let mut candidates = Vec::new();
        for det in raw.chunks_exact(7) {
            if det[5] != 0.0 || det[6] < settings.plate_threshold { continue; }
            let bbox = BBox {
                x1: x as f32 + det[1]*crop_w as f32/384.0,
                y1: y as f32 + det[2]*crop_h as f32/384.0,
                x2: x as f32 + det[3]*crop_w as f32/384.0,
                y2: y as f32 + det[4]*crop_h as f32/384.0,
            }.clamp(w, h);
            if bbox.width() < 2.0 || bbox.height() < 2.0 { continue; }
            candidates.push((bbox,det[6]));
        }
        drop(detections);
        candidates.sort_by(|a,b| b.1.total_cmp(&a.1));
        candidates.truncate(settings.max_plates_per_frame);
        let mut observations = Vec::with_capacity(candidates.len());
        for (bbox, detection_score) in candidates {
            let (plate, score) = self.read_plate(original, bbox)?;
            if score >= settings.ocr_threshold && !plate.is_empty() {
                observations.push(PlateObservation { bbox, detection_score, plate, ocr_score: score });
            }
        }
        Ok(observations)
    }

    fn read_plate(&mut self, image: &RgbImage, bbox: BBox) -> Result<(String,f32)> {
        let x1 = bbox.x1.floor().max(0.0) as u32;
        let y1 = bbox.y1.floor().max(0.0) as u32;
        let x2 = bbox.x2.ceil().min(image.width() as f32) as u32;
        let y2 = bbox.y2.ceil().min(image.height() as f32) as u32;
        if x2 <= x1 || y2 <= y1 { return Ok((String::new(),0.0)); }
        let roi = imageops::crop_imm(image, x1, y1, x2 - x1, y2 - y1).to_image();
        let resized = imageops::resize(&roi, 128, 64, FilterType::Triangle);
        // Input `input`: [1, 64, 128, 3], NHWC uint8; no normalization.
        let raw = resized.as_raw();
        let input = TensorRef::from_array_view(([1usize,64,128,3], raw.as_slice()))?;
        let results = self.ocr.run(ort::inputs![input])?;
        let (_, data) = results[0].try_extract_tensor::<f32>()?;
        const ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
        let slots = data.len()/ALPHABET.len();
        if slots == 0 || data.len() % ALPHABET.len() != 0 {
            bail!("OCR output has {} values; expected slots*{}", data.len(), ALPHABET.len());
        }
        let mut plate = String::new();
        let (mut confidence,mut characters) = (0.0f32,0usize);
        for values in data.chunks_exact(ALPHABET.len()) {
            let (class, score) = values.iter().enumerate()
                .max_by(|(_,a),(_,b)| a.total_cmp(b))
                .context("empty OCR alphabet")?;
            let character = ALPHABET[class] as char;
            if character != '_' {
                plate.push(character);
                confidence += *score;
                characters += 1;
            }
        }
        Ok((plate,if characters == 0 {0.0} else {confidence/characters as f32}))
    }
}
