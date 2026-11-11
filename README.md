# Rust Lpr

**Multi-camera automatic license plate recognition (ALPR/ANPR) in Rust.**

Rust Lpr reads RTSP video streams, detects license plates with ONNX Runtime, recognizes plate text with an OCR model, tracks detections across frames, and sends completed-track events to another application over TCP.



> **Project status:** Development / experimental. The architecture and configuration described below reflect the current source code. Cross-platform builds and real-world performance have not been validated on every operating system or hardware configuration.

## Features

- **Multiple RTSP cameras:** independent capture and recognition pipelines.
- **One OS process per camera:** failures and ONNX sessions are isolated from other camera workers.
- **Dedicated inference thread:** each camera has one persistent detector/OCR worker; no new inference task is created for each frame.
- **ONNX Runtime:** separate plate-detection and OCR sessions, with **per-camera session options**.
- **FFmpeg decoding:** one FFmpeg subprocess per camera, with configurable decoder/filter thread counts.
- **Bounded frame queue:** drops older frames when inference falls behind to favor live-video freshness.
- **Plate tracking:** combines observations and emits events when tracks finish.
- **TCP event delivery:** newline-delimited JSON (NDJSON), with automatic reconnection attempts.
- **HTTP management API:** list, add, stop, and restart camera processes.
- **Portable configuration:** machine-specific binary paths in `.env`; camera and inference settings in JSON.

## Architecture

```text
                   Rust Lpr Manager
                   HTTP API :8080
                          |
             spawns / stops / restarts
                          |
           +--------------+--------------+
           |              |              |
      Camera Worker 1  Camera Worker 2  Camera Worker N
           |              |              |
      FFmpeg (RTSP)   FFmpeg (RTSP)  FFmpeg (RTSP)
           |              |              |
      Bounded Queue   Bounded Queue  Bounded Queue
           |              |              |
      Dedicated Thread (one per camera)
           |
      Detection ONNX -> OCR ONNX -> Tracking
           |
      TCP NDJSON -> your application / event receiver
```

The manager **does not load ONNX models**. Each camera process loads exactly **one detector session and one OCR session**. Decoder, inference, and manager processes still share the host's CPU and memory bandwidth.

## Requirements

Install these dependencies for your **own operating system and CPU architecture**:

1. A supported Rust toolchain and Cargo, plus any native build tools required by your Rust target.
2. **FFmpeg**, installed and accessible through `PATH` or provided as an executable path.
3. **ONNX Runtime native library**, compatible with the Rust `ort` binding used by this project (the current source uses the ONNX Runtime 1.23-era API).
4. The **plate detector and OCR ONNX model files** expected by this pipeline.
5. Reachable RTSP camera streams and, for event delivery, a TCP receiver.

Examples of the native ONNX Runtime library name:

| Platform | Library filename |
| --- | --- |
| Windows | `onnxruntime.dll` |
| Linux | `libonnxruntime.so` |
| macOS | `libonnxruntime.dylib` |

**Platform notes:** Windows GNU builds may require MSYS2/UCRT64 build tools and runtime DLLs; MSVC builds require the appropriate Microsoft C++ build tools. Linux and macOS require their own native build/runtime dependencies, **not MSYS2**. Use matching 64-bit architectures. Merely changing the library filename does not guarantee platform compatibility.

## Quick start

### 1. Clone and configure

```bash
git clone https://github.com/sametpehlivan/rust-lpr.git
cd rust-lpr
```

Copy `.env.example` to `.env`:

```bash
# Linux / macOS
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env` with the paths for your machine:

```dotenv
RUST_LPR_FFMPEG_PATH=ffmpeg
ORT_DYLIB_PATH=./onnxruntime/lib/onnxruntime.dll
```

On Linux use `libonnxruntime.so`; on macOS use `libonnxruntime.dylib`. The relative runtime path is resolved against the configuration directory when the file exists. An existing OS environment variable takes precedence over the `.env` entry. If your runtime is discoverable automatically, `ORT_DYLIB_PATH` may be omitted.

> `.env` is optional when all dependencies are discoverable through the system environment. Never commit real `.env` values or camera credentials.

### 2. Provide the models

The default configuration expects:

```text
models/
  detector.onnx
  ocr.onnx
```


### 3. Configure cameras

Edit `config.example.json` or copy it to your own local configuration. For example, the relevant sections can look like this:

```json
{
  "api_bind": "127.0.0.1:8080",
  "tcp_destination": "127.0.0.1:9000",
  "frame_queue_capacity": 2,
  "ffmpeg_decoder_threads": 1,
  "ffmpeg_filter_threads": 1,
  "inference": {
    "plate_detector": "models/detector.onnx",
    "plate_ocr": "models/ocr.onnx",
    "device": "cpu",
    "plate_threshold": 0.5,
    "ocr_threshold": 0.8,
    "max_plates_per_frame": 10
  },
  "cameras": [
    {
      "id": "entrance-1",
      "rtsp_url": "rtsp://127.0.0.1:8554/live/1",
      "width": 768,
      "height": 432,
      "fps": 25,
      "roi": [0.0, 0.0, 1.0, 1.0],
      "polygon": [],
      "max_track_age_frames": 7,
      "max_track_idle_seconds": 5,
      "onnx_sessions": {
        "detector": { "intra_threads": 4 },
        "ocr": { "intra_threads": 1 }
      }
    }
  ]
}
```

Replace the example RTSP URL with your own source. The `roi` array uses normalized **`[x, y, width, height]`** coordinates; `[0, 0, 1, 1]` selects the full frame. Camera IDs must be unique. `fps` is the frame rate requested from the FFmpeg output filter, not necessarily the frame rate decoded at the camera input.

### 4. Build and run

```bash
cargo build --release
cargo run --release -- --config config.example.json
```

The built executable is in `target/release/` (with `.exe` on Windows). Its filename follows the `name` in `Cargo.toml`; if you renamed the Cargo package to `rustlpr`, the executable will be `rustlpr` / `rustlpr.exe`.

## Per-camera ONNX Runtime settings

Every camera has its own `onnx_sessions.detector` and `onnx_sessions.ocr` configuration. Any omitted fields inherit the defaults:

| Option | Detector default | OCR default |
| --- | ---: | ---: |
| `intra_threads` | `4` | `1` |
| `inter_threads` | `1` | `1` |
| `parallel_execution` | `false` | `false` |
| `optimization_level` | `all` | `all` |
| `intra_op_spinning` | `false` | `false` |
| `inter_op_spinning` | `false` | `false` |
| `memory_pattern` | `true` | `true` |
| `cpu_mem_arena` | `true` | `true` |

For example, to change one camera's CPU parallelism:

```json
"onnx_sessions": {
  "detector": { "intra_threads": 6 },
  "ocr": { "intra_threads": 2 }
}
```

`intra_threads: 0` lets ONNX Runtime choose its default thread count. More threads are **not guaranteed to improve FPS**: multiple camera processes may compete for the same CPU cores and memory bandwidth. Measure throughput and latency with the intended number of cameras running.

**Legacy configuration:** `inference.worker_count` may still appear in older JSON files, but it does **not** create additional sessions in this process-per-camera architecture. Every camera worker has one detector/OCR session pair. `max_plates_per_frame: 10` allows up to ten sequential OCR attempts per frame; it does not spawn ten OCR sessions.

## Camera management API

The HTTP management interface listens on `api_bind` (default example: `127.0.0.1:8080`).

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Manager health check |
| `GET` | `/api/cameras` | List camera processes and PIDs |
| `POST` | `/api/cameras` | Start a camera from a JSON `CameraConfig` body |
| `DELETE` | `/api/cameras/{id}` | Stop a camera |
| `POST` | `/api/cameras/{id}/restart` | Restart a camera |

For example:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/api/cameras
curl -X POST http://127.0.0.1:8080/api/cameras/entrance-1/restart
curl -X DELETE http://127.0.0.1:8080/api/cameras/entrance-1
```

The manager first asks a camera worker to stop gracefully. If it has not exited within the shutdown grace period, the manager can terminate it forcibly. A process reported as launched is not proof that the RTSP stream and ONNX sessions have fully initialized; inspect worker logs.

## TCP event output

When a plate track completes, Rust Lpr sends one JSON object per line (**NDJSON**) to `tcp_destination`. Events include a generated event ID, camera/source ID, tracked ID, timestamp, recognized plate, confidence, observations, and tracking data. Your receiving service must already be listening on the specified host and port.

**Delivery is best-effort.** The sender attempts to reconnect and retry pending writes, but there is no durable disk queue, receiver acknowledgment, or guaranteed exactly-once delivery. If these events drive billing, access control, or other critical workflows, add persistent delivery, acknowledgments, and idempotency on the receiving side.

## Performance and diagnostics

Each camera worker logs diagnostics approximately every ten seconds, including:

- `output_fps` and `inference_fps`
- `inference_ms` and `inference_max_ms`
- `queue_depth`, `queue_drops_10s`, and `queue_age_ms`
- `ffmpeg_output_read_ms` and `tcp_queue_dropped`

`inference_ms` is an end-to-end frame processing measurement; **it is not solely the ONNX detector's `Session.run()` time**. The input pipeline also performs crop/resize and OCR when plates are detected.

To limit latency and memory growth, a camera worker clamps the frame queue to **at most two RGB frames**, even if an older config specifies a larger capacity. This deliberately drops frames when processing cannot keep up. Increasing FPS, camera count, ONNX threads, or resolution increases system load; measure under realistic conditions.

## Important notes before deployment

1. **Protect the management API.** It has no built-in authentication or TLS. Keep `api_bind` on loopback by default; do not expose it to the public Internet without appropriate access controls and a secure reverse proxy.
2. **Keep credentials private.** RTSP URLs may contain usernames/passwords. Avoid committing real camera URLs or secrets in JSON; protect local configuration files and logs.
3. **Provide native dependencies.** Rust source portability does not mean one binary runs on every OS. FFmpeg and ONNX Runtime must be installed or supplied for each target platform, with matching architecture and compatible native libraries.
4. **Check model compatibility and licenses.** The detector expects RGB float32 NCHW input `[1,3,384,384]`; OCR expects RGB uint8 NHWC input `[1,64,128,3]`. Replacing either model can require code changes. Verify redistribution rights for model weights, FFmpeg, ONNX Runtime, and bundled binaries.
5. **Do not assume fixed inference speed.** CPU utilization, memory use, latency, and FPS vary by hardware, model, preprocessing workload, and number of concurrent camera processes.
6. **Recognitions can be wrong.** OCR confidence is not proof of identity. Test in your actual environment before using output for access, payment, or enforcement decisions.
7. **Treat plate data responsibly.** License plates, camera credentials, and event history may be sensitive or regulated. Apply appropriate retention, access control, and applicable privacy-law requirements.
8. **Expect network interruptions.** RTSP reconnects and TCP reconnects are supported, but disconnects can still cause frame drops and missed events. Monitor service health.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ONNX Runtime library ... not found` | Set `ORT_DYLIB_PATH` to the actual `.dll`, `.so`, or `.dylib`; check native dependencies. |
| FFmpeg fails to start | Check `RUST_LPR_FFMPEG_PATH`, executable permissions, and `ffmpeg -version`. |
| Camera process exits immediately | Verify RTSP credentials, stream reachability, model paths, and worker log output. |
| Low inference FPS | Compare one camera vs. several; tune per-camera detector threads, input resolution, and CPU contention. |
| High queue drops | Inference is slower than frame arrival. This can be intentional for low-latency operation. |
| TCP connection refused | Start the receiving service on the `tcp_destination` address/port. |
| Native build failure | Check OS-specific toolchain/linker setup and library architecture. |

## Development

```bash
cargo fmt --check
cargo clippy --all-targets
cargo test
cargo build --release
```

These are suggested local validation commands, **not a claim that the current project has been tested successfully on all targets**.

