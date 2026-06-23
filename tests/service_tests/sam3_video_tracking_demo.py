from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = PROJECT_ROOT / "third_party/sam3/assets/videos/bedroom.mp4"
DEFAULT_OUTPUT_DIR = Path("tests/service_tests/outputs/sam3_video_tracking")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original SAM3 video tracker API on a short video."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-frames", type=int, default=45)
    parser.add_argument("--point-x", type=float, default=210.0)
    parser.add_argument("--point-y", type=float, default=350.0)
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-temporal-disambiguation",
        action="store_true",
        help="Disable extra SAM3 temporal heuristics for a smaller/faster demo.",
    )
    return parser.parse_args()


def read_video_frames(video_path: Path, max_frames: int):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    frames = []
    while len(frames) < max_frames:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(bgr)

    capture.release()
    if not frames:
        raise SystemExit(f"No frames read from video: {video_path}")
    return frames


def overlay_mask(frame_bgr, mask):
    import cv2
    import numpy as np

    overlay = frame_bgr.copy()
    overlay[mask] = np.array([0, 40, 255], dtype=overlay.dtype)
    return cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0)


def save_demo_video(frames, output_path: Path, fps: float = 15.0) -> None:
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer: {output_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    args = parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_video_model

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = read_video_frames(args.video, args.max_frames)
    height, width = frames[0].shape[:2]

    started_at = time.perf_counter()
    sam3_model = build_sam3_video_model(
        device=args.device,
        apply_temporal_disambiguation=not args.no_temporal_disambiguation,
    )
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    build_latency_s = time.perf_counter() - started_at

    inference_state = predictor.init_state(video_path=str(args.video))

    rel_points = [[args.point_x / width, args.point_y / height]]
    points_tensor = torch.tensor(rel_points, dtype=torch.float32)
    labels_tensor = torch.tensor([1], dtype=torch.int32)

    prompt_started_at = time.perf_counter()
    _, out_obj_ids, _, video_res_masks = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=args.obj_id,
        points=points_tensor,
        labels=labels_tensor,
        clear_old_points=False,
    )
    prompt_latency_s = time.perf_counter() - prompt_started_at

    first_mask = (video_res_masks[0] > 0.0).detach().cpu().numpy().squeeze()
    cv2.imwrite(str(args.output_dir / "frame_0000_prompt_overlay.jpg"), overlay_mask(frames[0], first_mask))

    tracked_masks: dict[int, dict[int, np.ndarray]] = {}
    propagate_started_at = time.perf_counter()
    for frame_idx, obj_ids, _, masks, obj_scores in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=args.max_frames,
        reverse=False,
        propagate_preflight=True,
        tqdm_disable=True,
    ):
        tracked_masks[frame_idx] = {
            int(obj_id): (masks[i] > 0.0).detach().cpu().numpy().squeeze()
            for i, obj_id in enumerate(obj_ids)
        }
    propagate_latency_s = time.perf_counter() - propagate_started_at

    overlay_frames = []
    for frame_idx, frame in enumerate(frames[: args.max_frames]):
        obj_masks = tracked_masks.get(frame_idx, {})
        if args.obj_id in obj_masks:
            out = overlay_mask(frame, obj_masks[args.obj_id])
        else:
            out = frame
        overlay_frames.append(out)
        if frame_idx in {0, min(10, len(frames) - 1), min(30, len(frames) - 1)}:
            cv2.imwrite(str(args.output_dir / f"frame_{frame_idx:04d}_overlay.jpg"), out)

    output_video = args.output_dir / "sam3_tracking_demo.mp4"
    save_demo_video(overlay_frames, output_video)

    summary = {
        "video": str(args.video),
        "output_video": str(output_video),
        "frames_requested": args.max_frames,
        "frames_read": len(frames),
        "tracked_frames": len(tracked_masks),
        "obj_ids": [int(obj_id) for obj_id in out_obj_ids],
        "point": [args.point_x, args.point_y],
        "build_latency_s": build_latency_s,
        "prompt_latency_s": prompt_latency_s,
        "propagate_latency_s": propagate_latency_s,
        "tracking_fps": len(tracked_masks) / propagate_latency_s if propagate_latency_s else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
