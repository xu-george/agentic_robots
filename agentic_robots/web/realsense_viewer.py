from __future__ import annotations

import base64
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from agentic_robots.apis.perception import PerceptionApi
from agentic_robots.utils.grasp_overlay import blend_mask, draw_grasp_overlay


ObservationProvider = Callable[[], dict[str, Any]]


@dataclass
class ViewerHandle:
    host: str
    port: int
    server: Any
    thread: threading.Thread

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=2.0)


def start_realsense_viewer(
    *,
    observation_provider: ObservationProvider,
    object_prompt: str,
    host: str = "127.0.0.1",
    port: int = 8120,
    refresh_s: float = 1.5,
    top_k: int = 5,
) -> ViewerHandle:
    """Start a read-only RealSense perception viewer in a background thread."""
    import uvicorn

    app = create_app(
        observation_provider=observation_provider,
        object_prompt=object_prompt,
        refresh_s=refresh_s,
        top_k=top_k,
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="realsense-viewer", daemon=True)
    thread.start()
    _wait_for_server(server)
    return ViewerHandle(host=host, port=port, server=server, thread=thread)


def create_app(
    *,
    observation_provider: ObservationProvider,
    object_prompt: str,
    refresh_s: float = 1.5,
    top_k: int = 5,
):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Agentic Robots RealSense Viewer")
    perception = PerceptionApi(observation_provider=observation_provider)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _html(refresh_s=refresh_s, object_prompt=object_prompt)

    @app.get("/api/frame")
    def frame() -> JSONResponse:
        try:
            obs = observation_provider()
            rgb = np.asarray(obs["rgb"], dtype=np.uint8)
            depth = np.asarray(obs["depth"], dtype=np.float32)
            intrinsics = np.asarray(obs["intrinsics"], dtype=np.float64)
            masks = perception.segment_object(rgb, object_prompt)
            best_mask = masks[0].mask if masks else None
            overlay = blend_mask(rgb, np.asarray(best_mask, dtype=bool)) if best_mask is not None else rgb

            grasp_score = None
            grasp_count = 0
            grasp_error = None
            if best_mask is not None:
                try:
                    grasp = perception.get_finger_grasp(
                        depth,
                        intrinsics,
                        masks,
                        obs.get("camera_pose"),
                    )
                    overlay = draw_grasp_overlay(
                        rgb=rgb,
                        mask=best_mask,
                        grasps=grasp.plan.grasps,
                        scores=grasp.plan.scores,
                        intrinsics=intrinsics,
                        top_k=top_k,
                    )
                    grasp_score = float(grasp.score)
                    grasp_count = int(np.asarray(grasp.plan.grasps).reshape(-1, 4, 4).shape[0])
                except Exception as exc:
                    grasp_error = str(exc)

            payload = {
                "timestamp": time.time(),
                "object_prompt": object_prompt,
                "rgb": _image_data_url(rgb),
                "depth": _image_data_url(_depth_to_rgb(depth)),
                "overlay": _image_data_url(overlay),
                "mask_count": len(masks),
                "mask_score": float(masks[0].score) if masks else None,
                "grasp_count": grasp_count,
                "grasp_score": grasp_score,
                "grasp_error": grasp_error,
            }
            return JSONResponse(payload)
        except Exception as exc:
            return JSONResponse({"error": str(exc), "timestamp": time.time()}, status_code=500)

    return app


def _wait_for_server(server: Any, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server.started:
            return
        time.sleep(0.05)


def _image_data_url(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to encode image")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _depth_to_rgb(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < 5.0)
    if not np.any(valid):
        normalized = np.zeros(depth.shape, dtype=np.uint8)
    else:
        near, far = np.percentile(depth[valid], [2.0, 98.0])
        if far <= near:
            far = near + 1e-3
        normalized = np.clip((depth - near) / (far - near) * 255.0, 0, 255).astype(np.uint8)
        normalized[~valid] = 0
    return cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def _html(*, refresh_s: float, object_prompt: str) -> str:
    refresh_ms = max(250, int(refresh_s * 1000))
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RealSense Perception Viewer</title>
  <style>
    body {{ margin: 0; font-family: Inter, system-ui, sans-serif; background: #101318; color: #eef2f7; }}
    header {{ padding: 16px 20px; border-bottom: 1px solid #2a3340; display: flex; justify-content: space-between; gap: 16px; }}
    main {{ padding: 18px; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
    .card {{ background: #161b22; border: 1px solid #2a3340; border-radius: 14px; overflow: hidden; }}
    .card h2 {{ margin: 0; padding: 10px 12px; font-size: 14px; color: #aab8c5; border-bottom: 1px solid #2a3340; }}
    img {{ width: 100%; display: block; background: #05070a; }}
    #status {{ color: #8ee6a8; font-size: 14px; }}
    #meta {{ padding: 0 18px 18px; color: #aab8c5; font-size: 13px; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div><strong>RealSense Perception Viewer</strong> · prompt: <code>{object_prompt}</code></div>
    <div id="status">starting</div>
  </header>
  <main>
    <section class="card"><h2>RGB</h2><img id="rgb"></section>
    <section class="card"><h2>Depth</h2><img id="depth"></section>
    <section class="card"><h2>Mask + Grasp Hypotheses</h2><img id="overlay"></section>
  </main>
  <div id="meta"></div>
  <script>
    async function update() {{
      const status = document.getElementById('status');
      try {{
        const res = await fetch('/api/frame', {{cache: 'no-store'}});
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);
        document.getElementById('rgb').src = data.rgb;
        document.getElementById('depth').src = data.depth;
        document.getElementById('overlay').src = data.overlay;
        status.textContent = 'live ' + new Date(data.timestamp * 1000).toLocaleTimeString();
        status.style.color = '#8ee6a8';
        document.getElementById('meta').textContent =
          `masks=${{data.mask_count}} mask_score=${{data.mask_score ?? 'n/a'}} ` +
          `grasps=${{data.grasp_count}} grasp_score=${{data.grasp_score ?? 'n/a'}} ` +
          `${{data.grasp_error ? 'grasp_error=' + data.grasp_error : ''}}`;
      }} catch (err) {{
        status.textContent = String(err);
        status.style.color = '#ff8c8c';
      }}
    }}
    update();
    setInterval(update, {refresh_ms});
  </script>
</body>
</html>
"""
