from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RealSenseFrame:
    rgb: np.ndarray
    depth_m: np.ndarray


class RealSenseCamera:
    """Small aligned RGB-D wrapper for Intel RealSense cameras."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int = 30,
        serial: str | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.pipeline = None
        self.align = None
        self.depth_scale_m = 0.001

    def start(self) -> None:
        if self.pipeline is not None:
            return

        import pyrealsense2 as rs

        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.pipeline = rs.pipeline()
        try:
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale_m = float(depth_sensor.get_depth_scale())
        except Exception:
            self.pipeline = None
            self.align = None
            raise

    def capture(self, *, settle_frames: int = 3) -> RealSenseFrame:
        if self.pipeline is None or self.align is None:
            self.start()

        frames = None
        for _ in range(max(1, int(settle_frames))):
            frames = self.align.process(self.pipeline.wait_for_frames(2000))
        if frames is None:
            raise RuntimeError("No RealSense frames received")

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense did not provide aligned color/depth frames")

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        return RealSenseFrame(
            rgb=cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB),
            depth_m=depth_raw.astype(np.float32) * self.depth_scale_m,
        )

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError as exc:
                if "cannot be called before start" not in str(exc):
                    raise
        self.pipeline = None
        self.align = None
