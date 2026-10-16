# Native SAM3 tracking on live ROS RGB

This adds a separate native tracking service on localhost:8116. The existing image
segmentation service (8114) and legacy OpenCV demo are unchanged.

## Start

From the project root:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  -m agentic_robots.services.sam3.tracking_server --device cuda:0
```

In a second terminal, using ROS-compatible system Python:

```bash
cd /home/user/Documents/Code/agentic_robots
source /home/user/d405_ros/host_env.sh
/usr/bin/python3 tests/service_tests/sam3_ros_tracking.py
```

The viewer now uses lossless packed RGB8 HTTP bodies by default (no PNG/JPEG
compression and no base64 image encoding). The RealSense publisher must already be running. The default topic is
`/camera/d405/color/image_raw`; override with `--topic`.
Click inside the object on the frozen initial frame, or launch with
`--prompt 'black handle'` for text-prompt segmentation. The highest-scoring
candidate above `--min-score` (default 0.3) initializes the native tracker using
`add_new_mask`. Inspect the mask and press SPACE to approve before tracking.
Text is used only for initialization, not repeated detection. Multiple matching
handles may be ambiguous; confidence does not prove it selected your intended one.
`r` resets when inference is idle; `q` exits. The viewer shows masks on the exact frames used for inference,
not stale masks painted on newer frames. The ROS callback retains only the latest
frame while one HTTP inference request runs in a worker.

## Implementation

- Builds the vendored `build_sam3_video_model` and uses its native tracker/backbone.
- Initializes via `add_new_points_or_box`; uses `propagate_in_video_preflight`
  then forward `propagate_in_video` on each newly appended frame.
- Keeps SAM3 temporal memory and object ID 1. No OpenCV MIL tracker, mask
  translation, or per-frame image re-segmentation is used.
- Normalization matches the native loader: square RGB resize, mean/std 0.5.
- `/track/start`: `image_base64` (PNG), exactly one of `point_xy` (pixel coordinates)
  or `text_prompt`, and optional `min_score` (0–1).
- `/track/frame`: `image_base64`, `session_id`. Response includes raw uint8
  base64 mask, shape, frame index, object ID and inference latency.
- `/track/start_raw`: binary `application/octet-stream` body of packed RGB8 bytes;
  query parameters `width`, `height`, and either `text_prompt` or `point_x`/`point_y`;
  optional `min_score`. Validates dimensions (1–4096) and exact byte count.
- `/track/frame_raw`: same binary body, query `width`, `height`, `session_id`.
- Both raw endpoints return the same JSON mask format as the original PNG endpoints.
- `/track/reset`: `session_id`; releases tracking state.

## Limits

Single object, single session, forward-only, fixed image dimensions, CUDA only.
A new start replaces the previous session. Text initialization lazily loads an
additional SAM3 image model on the same GPU. Inference is serialized. Stateful
requests are not automatically retried. Failed inference invalidates the session.
Images are discarded after use. Tracking state and normalized input tensors stay
on GPU by default; `--offload-state-to-cpu` restores CPU offloading. ROS input,
raw pixel packing, HTTP transfer, image resize and GUI display still use CPU work.
This is raw-byte transport, not shared memory or zero-copy. PNG endpoints remain
available for older clients.
Sessions stop after 300 processed frames by default (`--max-frames`) to bound temporal-state growth.
Select again with `r` at that limit. This is not an indefinite-stream memory-pruning
implementation. No automatic re-detection after occlusion is promised.

Camera FPS is not inference FPS. Frames are skipped when inference cannot keep up;
fast motion across skipped frames can reduce tracking accuracy. The first-frame
prompt uses a frozen image, so avoid moving the target before selecting it.

## Validation

API tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/user/miniforge3/envs/agentic_robotics/bin/python -m pytest \
  tests/service_tests/test_sam3_tracking_server.py -q
```

A real GPU smoke test on four consecutive frames of the bundled bedroom video
produced non-empty masks and sequential frame indices. Results and overlays:
`tests/service_tests/outputs/sam3_native_tracking/`. Initial prompt ~0.59 s;
three subsequent requests ~0.074–0.084 s each. These short-test timings are not
an end-to-end live tracking benchmark or an accuracy evaluation.

CPU vs GPU-state comparison on the same 90 captured frames (79 measured after
warmup), using `yellow cable` because the black handle was out of view:
- CPU state: 6.66 end-to-end FPS, 15.94 server FPS.
- GPU state: 6.79 end-to-end FPS, 16.74 server FPS.
- Standalone default PNG encoding averaged 78.5 ms/frame over 30 frames.
These are single-run measurements, not statistically established speedups.
Artifacts: `benchmark_cpu_state.json`, `benchmark_gpu_state.json`, and
`benchmark_frames.npy` in the same output directory. Re-run with
`tests/service_tests/sam3_tracking_benchmark.py`.

After removing PNG encoding, the same fixed clip and warmup protocol measured:
- Raw RGB + GPU state: **15.60 end-to-end FPS**, 16.62 server FPS.
- PNG + GPU state: **6.77 end-to-end FPS**, 16.64 server FPS.
- Approximately 2.3x throughput improvement, with unchanged input pixel values.
Artifacts: `benchmark_raw_gpu.json`, `benchmark_png_gpu.json`.
Use `--transport raw` (default) or `--transport png` on the benchmark script.
This measures request/response processing, not ROS capture or GUI cadence;
live frame rate may differ. Five API/client tests pass, including exact RGB pixel
preservation and malformed-body validation.
