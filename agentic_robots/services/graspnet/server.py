from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel

from agentic_robots.services.base import decode_numpy, encode_numpy


DEFAULT_GRASPNET_VENDOR_ROOT = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "contact_graspnet_pytorch"
).resolve()


class GraspNetEngine:
    """Thin wrapper around Contact-GraspNet.

    The heavy vendor package is loaded only when ``load`` is called.
    """

    def __init__(self) -> None:
        self.estimator: Any | None = None

    def load(
        self,
        checkpoint_dir: str | None = None,
        vendor_root: str | None = None,
        config_overrides: list[str] | None = None,
    ) -> None:
        """Load the real Contact-GraspNet model.

        Defaults point at the existing CAP-X checkout so the first
        reimplementation can reuse its vendored model without copying it.
        """
        vendor = Path(
            vendor_root
            or os.environ.get("GRASPNET_VENDOR_ROOT", "")
            or DEFAULT_GRASPNET_VENDOR_ROOT
        )
        checkpoint = Path(
            checkpoint_dir
            or os.environ.get("GRASPNET_CHECKPOINT_DIR", "")
            or vendor / "checkpoints/contact_graspnet/checkpoints"
        )

        pointnet_root = vendor / "Pointnet_Pointnet2_pytorch"
        for path in (vendor, pointnet_root):
            if str(path) not in sys.path:
                sys.path.append(str(path))

        from contact_graspnet_pytorch.checkpoints import CheckpointIO
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

        config = _load_contact_graspnet_config(checkpoint.parent, config_overrides or [])
        estimator = GraspEstimator(config)
        checkpoint_io = CheckpointIO(checkpoint_dir=str(checkpoint), model=estimator.model)
        checkpoint_io.load("model.pt")
        self.estimator = estimator

    def plan_from_depth(self, req: Any) -> dict[str, str]:
        if self.estimator is None:
            raise RuntimeError("GraspNet model is not loaded")

        depth = decode_numpy(req.depth_base64)
        cam_k = decode_numpy(req.cam_K_base64)
        segmap = decode_numpy(req.segmap_base64)
        z_range = req.z_range or [0.2, 2.0]

        pc_full, pc_segments, _ = self.estimator.extract_point_clouds(
            depth,
            cam_k,
            segmap=segmap,
            segmap_id=req.segmap_id,
            skip_border_objects=req.skip_border_objects,
            z_range=z_range,
        )
        return self._predict(
            pc_full=pc_full,
            pc_segments=pc_segments,
            segmap_id=req.segmap_id,
            local_regions=req.local_regions,
            filter_grasps=req.filter_grasps,
            forward_passes=req.forward_passes,
        )

    def plan_from_point_clouds(self, req: Any) -> dict[str, str]:
        if self.estimator is None:
            raise RuntimeError("GraspNet model is not loaded")

        pc_full = decode_numpy(req.pc_full_base64)
        pc_segment = decode_numpy(req.pc_segment_base64)
        pc_segments = {req.segmap_id: pc_segment}
        return self._predict(
            pc_full=pc_full,
            pc_segments=pc_segments,
            segmap_id=req.segmap_id,
            local_regions=req.local_regions,
            filter_grasps=req.filter_grasps,
            forward_passes=req.forward_passes,
        )

    def _predict(
        self,
        *,
        pc_full: np.ndarray,
        pc_segments: dict[int, np.ndarray],
        segmap_id: int,
        local_regions: bool,
        filter_grasps: bool,
        forward_passes: int,
    ) -> dict[str, str]:
        pred_grasps, scores, contact_points, _ = self.estimator.predict_scene_grasps(
            pc_full,
            pc_segments=pc_segments,
            local_regions=local_regions,
            filter_grasps=filter_grasps,
            forward_passes=forward_passes,
        )

        grasps_out = np.asarray(pred_grasps.get(segmap_id, np.asarray([])))
        scores_out = np.asarray(scores.get(segmap_id, np.asarray([])))
        contacts_out = np.asarray(contact_points.get(segmap_id, np.asarray([])))

        return {
            "grasps_base64": encode_numpy(grasps_out),
            "scores_base64": encode_numpy(scores_out),
            "contact_pts_base64": encode_numpy(contacts_out),
        }


class PlanRequest(BaseModel):
    depth_base64: str
    cam_K_base64: str
    segmap_base64: str
    segmap_id: int = 1
    local_regions: bool = True
    filter_grasps: bool = True
    skip_border_objects: bool = False
    z_range: list[float] | None = None
    forward_passes: int = 2
    max_retries: int = 10


class PointCloudPlanRequest(BaseModel):
    pc_full_base64: str
    pc_segment_base64: str
    segmap_id: int = 1
    local_regions: bool = True
    filter_grasps: bool = True
    forward_passes: int = 2
    max_retries: int = 10


def create_app(engine: GraspNetEngine | None = None):
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="Agentic Robots GraspNet Service")
    engine = engine or GraspNetEngine()
    gpu_lock = asyncio.Semaphore(1)

    async def run_locked(fn, *args):
        async with gpu_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(fn, *args))

    @app.get("/health")
    def health():
        return {"ok": engine.estimator is not None, "service": "graspnet"}

    @app.post("/plan")
    async def plan(req: PlanRequest = Body(...)):
        try:
            return await run_locked(engine.plan_from_depth, req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/plan_point_clouds")
    async def plan_point_clouds(req: PointCloudPlanRequest = Body(...)):
        try:
            return await run_locked(engine.plan_from_point_clouds, req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main(
    host: str = "127.0.0.1",
    port: int = 8115,
    checkpoint_dir: str | None = None,
    vendor_root: str | None = None,
) -> None:
    import uvicorn

    engine = GraspNetEngine()
    engine.load(checkpoint_dir=checkpoint_dir, vendor_root=vendor_root)
    uvicorn.run(create_app(engine), host=host, port=port)


def _load_contact_graspnet_config(checkpoint_parent: Path, overrides: list[str]) -> dict[str, Any]:
    config_path = checkpoint_parent / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    for override in overrides:
        key, raw_value = override.split(":", 1)
        with contextlib.suppress(Exception):
            raw_value = eval(raw_value)
        keys = [int(part) if part.isdigit() else part for part in key.split(".")]
        _recursive_assign(config, keys, raw_value)

    config["DATA"]["classes"] = None
    return config


def _recursive_assign(data: dict[str, Any], keys: list[Any], value: Any) -> None:
    if len(keys) == 1:
        data[keys[0]] = value
        return
    _recursive_assign(data[keys[0]], keys[1:], value)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Agentic Robots GraspNet service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8115)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--vendor-root")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        host=args.host,
        port=args.port,
        checkpoint_dir=args.checkpoint_dir,
        vendor_root=args.vendor_root,
    )
