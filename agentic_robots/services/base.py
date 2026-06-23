from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from PIL import Image


class ServiceError(RuntimeError):
    """Raised when a service request fails."""


@dataclass(frozen=True)
class ServiceConfig:
    url: str
    timeout_s: float = 60.0
    retries: int = 2
    retry_delay_s: float = 0.3


def post_json(config: ServiceConfig, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON to a service with small retry handling."""
    url = f"{config.url.rstrip('/')}/{path.lstrip('/')}"
    last_error: Exception | None = None
    for attempt in range(config.retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=config.timeout_s)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < config.retries:
                time.sleep(config.retry_delay_s)
    raise ServiceError(f"POST {url} failed: {last_error}") from last_error


def encode_image(image: np.ndarray | Image.Image) -> str:
    """Encode an RGB image as base64 PNG."""
    if isinstance(image, np.ndarray):
        image_u8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_u8).convert("RGB")
    else:
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image(image_base64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")


def encode_mask(mask: np.ndarray) -> str:
    """Encode a boolean/uint8 mask as raw bytes."""
    return base64.b64encode(mask.astype(np.uint8, copy=False).tobytes()).decode("utf-8")


def decode_mask(mask_base64: str, shape: tuple[int, ...]) -> np.ndarray:
    data = base64.b64decode(mask_base64)
    return np.frombuffer(data, dtype=np.uint8).reshape(shape).astype(bool)


def encode_numpy(array: np.ndarray) -> str:
    """Encode an ndarray using np.save so dtype and shape are preserved."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_numpy(array_base64: str) -> np.ndarray:
    data = base64.b64decode(array_base64)
    return np.load(io.BytesIO(data), allow_pickle=False)
