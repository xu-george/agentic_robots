import asyncio
import functools
from typing import Any

import numpy as np

from agentic_robots.services.base import decode_image, encode_mask, encode_numpy


class Sam3Engine:
    """Thin wrapper around SAM3.

    Heavy SAM3 imports happen inside ``load`` so importing this module stays cheap.
    """

    def __init__(self) -> None:
        self.model: Any | None = None
        self.processor: Any | None = None
        self.device = "cuda"

    def load(self, device: str = "cuda") -> None:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = device
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.set_device(int(device.split(":")[-1]) if ":" in device else 0)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model = build_sam3_image_model(enable_inst_interactivity=True)
        self.model = model.to(device) if hasattr(model, "to") else model
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=0.0)

    def segment(self, image_base64: str, prompt: str) -> list[dict[str, Any]]:
        import torch

        if self.processor is None:
            raise RuntimeError("SAM3 model is not loaded")

        image = decode_image(image_base64)
        device_type = "cuda" if "cuda" in self.device else "cpu"
        with torch.autocast(device_type, dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(state=state, prompt=prompt)

        masks = _to_numpy(output.get("masks"))
        boxes = _to_numpy(output.get("boxes"))
        scores = _to_numpy(output.get("scores"))

        if masks is None or boxes is None or scores is None:
            return []
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        results: list[dict[str, Any]] = []
        for mask, box, score in zip(masks, boxes, scores):
            mask_bool = mask > 0
            results.append(
                {
                    "mask_base64": encode_mask(mask_bool),
                    "shape": list(mask_bool.shape),
                    "box": [float(v) for v in box],
                    "score": float(score),
                    "label": prompt,
                }
            )
        return sorted(results, key=lambda item: item["score"], reverse=True)

    def segment_point(self, image_base64: str, point_coords: list[float]) -> dict[str, Any]:
        import torch

        if self.processor is None or self.model is None:
            raise RuntimeError("SAM3 model is not loaded")

        image = decode_image(image_base64)
        point_xy = np.asarray([point_coords], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int64)
        device_type = "cuda" if "cuda" in self.device else "cpu"

        with torch.autocast(device_type, dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            masks, scores, _ = self.model.predict_inst(
                state,
                point_coords=point_xy,
                point_labels=point_labels,
                multimask_output=True,
            )

        masks_np = np.asarray(masks)
        scores_np = np.asarray(scores)
        order = np.argsort(scores_np)[::-1]
        masks_np = masks_np[order]
        scores_np = scores_np[order]

        return {
            "scores": scores_np.astype(float).tolist(),
            "masks_base64": encode_numpy(masks_np),
            "masks_shape": list(masks_np.shape),
            "masks_dtype": str(masks_np.dtype),
        }


def create_app(engine: Sam3Engine | None = None):
    from fastapi import Body, FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="Agentic Robots SAM3 Service")
    engine = engine or Sam3Engine()
    gpu_lock = asyncio.Semaphore(1)

    class SegmentRequest(BaseModel):
        image_base64: str
        text_prompt: str

    class PointRequest(BaseModel):
        image_base64: str
        point_coords: list[float]

    async def run_locked(fn, *args):
        async with gpu_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(fn, *args))

    @app.get("/health")
    def health():
        return {"ok": engine.processor is not None, "service": "sam3"}

    @app.post("/segment")
    async def segment(req: SegmentRequest = Body(...)):
        try:
            return {"results": await run_locked(engine.segment, req.image_base64, req.text_prompt)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/segment_point")
    async def segment_point(req: PointRequest = Body(...)):
        try:
            return await run_locked(engine.segment_point, req.image_base64, req.point_coords)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main(device: str = "cuda", host: str = "127.0.0.1", port: int = 8114) -> None:
    import uvicorn

    engine = Sam3Engine()
    engine.load(device=device)
    uvicorn.run(create_app(engine), host=host, port=port)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Agentic Robots SAM3 service.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8114)
    return parser.parse_args()


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    elif hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float") and str(getattr(value, "dtype", "")) == "torch.bfloat16":
        value = value.float()
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


if __name__ == "__main__":
    args = _parse_args()
    main(device=args.device, host=args.host, port=args.port)
