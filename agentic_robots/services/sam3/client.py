from __future__ import annotations

import os

import numpy as np
from PIL import Image

from agentic_robots.services.base import (
    ServiceConfig,
    decode_mask,
    decode_numpy,
    encode_image,
    post_json,
)
from agentic_robots.services.sam3.types import PointPromptResult, Sam3Mask


DEFAULT_SAM3_URL = os.environ.get("SAM3_SERVICE_URL", "http://127.0.0.1:8114")


class Sam3Client:
    """HTTP client for text-prompt and point-prompt SAM3 segmentation."""

    def __init__(self, url: str = DEFAULT_SAM3_URL, timeout_s: float = 60.0):
        self.config = ServiceConfig(url=url, timeout_s=timeout_s)

    def segment(self, image: np.ndarray | Image.Image, prompt: str) -> list[Sam3Mask]:
        payload = {"image_base64": encode_image(image), "text_prompt": prompt}
        data = post_json(self.config, "/segment", payload)

        masks: list[Sam3Mask] = []
        for item in data.get("results", []):
            shape = tuple(item["shape"])
            box = tuple(float(v) for v in item["box"])
            masks.append(
                Sam3Mask(
                    mask=decode_mask(item["mask_base64"], shape),
                    box=box,  # type: ignore[arg-type]
                    score=float(item["score"]),
                    label=str(item.get("label", prompt)),
                )
            )
        return masks

    def segment_point(
        self, image: np.ndarray | Image.Image, point_xy: tuple[float, float]
    ) -> list[PointPromptResult]:
        payload = {"image_base64": encode_image(image), "point_coords": list(point_xy)}
        data = post_json(self.config, "/segment_point", payload)

        scores = [float(score) for score in data.get("scores", [])]
        shape = tuple(data.get("masks_shape", (0, 0, 0)))
        dtype = np.dtype(data.get("masks_dtype", "float32"))
        masks = decode_numpy(data["masks_base64"]).astype(dtype, copy=False)
        masks = masks.reshape(shape).astype(bool)

        return [PointPromptResult(mask=mask, score=score) for mask, score in zip(masks, scores)]

