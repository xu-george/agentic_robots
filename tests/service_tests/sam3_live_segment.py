from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("tests/service_tests/outputs/sam3_live_segment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live camera segmentation with the SAM3 service."
    )
    parser.add_argument("prompt", help='Text prompt, for example "banana" or "red cup".')
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="OpenCV camera index. Omit to use the first readable camera.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8114", help="SAM3 service URL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Number of frames to segment. Use 0 to run until Ctrl-C.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.0,
        help="Minimum seconds between captured frames.",
    )
    parser.add_argument("--max-results", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--segment-every",
        type=int,
        default=5,
        help="Submit one frame to SAM3 every N captured frames when the worker is idle.",
    )
    parser.add_argument(
        "--segment-width",
        type=int,
        default=640,
        help="Resize frames to this width before SAM3. Use 0 to keep camera resolution.",
    )
    parser.add_argument(
        "--segment-height",
        type=int,
        default=0,
        help="Optional segmentation height. Use 0 to preserve aspect ratio.",
    )
    parser.add_argument(
        "--save-per-frame",
        action="store_true",
        help="Save every segmented mask/overlay. By default only latest outputs and JSONL are saved.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help=(
            "Track the latest SAM3 mask between SAM3 refreshes for smoother display. "
            "SAM3 still refreshes every --segment-every frames."
        ),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live overlay window in addition to saving latest files.",
    )
    return parser.parse_args()


def check_service(url: str) -> None:
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=3.0)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"SAM3 service is not reachable at {url}. "
            "Start it with: python -m agentic_robots.services.sam3.server"
        ) from exc


def open_camera(camera_index: int | None, width: int, height: int):
    import cv2

    indexes = [camera_index] if camera_index is not None else list(range(10))
    for index in indexes:
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)

        if not capture.isOpened():
            capture.release()
            continue

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ok, _ = capture.read()
        if ok:
            return capture, index

        capture.release()

    if camera_index is None:
        raise SystemExit("Could not find a readable camera in indexes 0-9")
    raise SystemExit(f"Could not open camera index {camera_index}")


def save_mask(mask, path: Path) -> None:
    import numpy as np
    from PIL import Image

    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    image.save(path)


def make_overlay(image, mask):
    import numpy as np
    from PIL import Image

    base = image.convert("RGBA")
    alpha = Image.fromarray(mask.astype(np.uint8) * 120, mode="L")
    red = Image.new("RGBA", base.size, (255, 40, 40, 0))
    red.putalpha(alpha)
    return Image.alpha_composite(base, red)


def make_overlay_bgr(frame_bgr, mask):
    import numpy as np
    import cv2

    overlay = frame_bgr.copy()
    overlay[mask] = np.array([0, 40, 255], dtype=overlay.dtype)
    return cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0)


def resize_for_segmentation(image, width: int, height: int):
    if width <= 0 and height <= 0:
        return image

    src_width, src_height = image.size
    if width <= 0:
        width = round(src_width * (height / src_height))
    elif height <= 0:
        height = round(src_height * (width / src_width))

    if (width, height) == image.size:
        return image

    return image.resize((width, height))


def resize_mask(mask, size: tuple[int, int]):
    from PIL import Image
    import numpy as np

    if mask.shape[::-1] == size:
        return mask

    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask_image = mask_image.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(mask_image) > 0


def bbox_from_mask(mask) -> tuple[int, int, int, int] | None:
    import numpy as np

    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def clamp_box(box, frame_shape) -> tuple[int, int, int, int] | None:
    height, width = frame_shape[:2]
    x, y, w, h = (int(round(v)) for v in box)
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    if w <= 1 or h <= 1:
        return None
    return x, y, w, h


def crop_mask_template(mask, box):
    x, y, w, h = box
    return mask[y : y + h, x : x + w].copy()


def render_tracked_mask(template_mask, box, frame_shape):
    import cv2
    import numpy as np

    clamped = clamp_box(box, frame_shape)
    if clamped is None:
        return None
    x, y, w, h = clamped
    resized = cv2.resize(
        template_mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    mask = np.zeros(frame_shape[:2], dtype=bool)
    mask[y : y + h, x : x + w] = resized
    return mask


def create_tracker(cv2):
    if hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()
    raise SystemExit(
        "This OpenCV build does not provide TrackerMIL_create, so --track cannot run. "
        "Install opencv-contrib-python or a GUI OpenCV build with tracking support."
    )


def initialize_tracker(cv2, frame_bgr, box):
    tracker = create_tracker(cv2)
    tracker.init(frame_bgr, tuple(int(v) for v in box))
    return tracker


def result_record(
    frame_index: int,
    result,
    mask_path: Path,
    overlay_path: Path,
    latency_s: float,
    image_size: tuple[int, int],
    segment_size: tuple[int, int],
) -> dict:
    return {
        "frame_index": frame_index,
        "label": result.label,
        "score": result.score,
        "box": list(result.box),
        "image_size": list(image_size),
        "segment_size": list(segment_size),
        "latency_s": latency_s,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    }


def check_display_support(cv2) -> None:
    build_info = cv2.getBuildInformation()
    has_gui = not any(line.strip() == "GUI:                           NONE" for line in build_info.splitlines())
    if not has_gui:
        raise SystemExit(
            "OpenCV was installed without GUI support, so --display cannot use cv2.imshow. "
            "Install a GUI OpenCV build, for example: "
            "python -m pip uninstall opencv-python-headless && "
            "python -m pip install opencv-python==4.10.0.84"
        )

    if os.name == "posix" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise SystemExit(
            "--display needs a graphical desktop session. No DISPLAY or WAYLAND_DISPLAY "
            "environment variable is set."
        )


def segment_frame(
    client,
    rgb,
    frame_index: int,
    prompt: str,
    max_results: int,
    segment_width: int,
    segment_height: int,
):
    from PIL import Image

    started_at = time.perf_counter()
    image = Image.fromarray(rgb)
    segment_image = resize_for_segmentation(image, segment_width, segment_height)
    results = client.segment(segment_image, prompt)[:max_results]
    latency_s = time.perf_counter() - started_at
    return {
        "frame_index": frame_index,
        "image": image,
        "segment_image": segment_image,
        "results": results,
        "latency_s": latency_s,
    }


def handle_segment_result(
    result_data: dict,
    prompt: str,
    service_url: str,
    output_dir: Path,
    jsonl,
    latest_frame_path: Path,
    latest_mask_path: Path,
    latest_overlay_path: Path,
    save_per_frame: bool,
):
    frame_index = result_data["frame_index"]
    image = result_data["image"]
    segment_image = result_data["segment_image"]
    results = result_data["results"]
    latency_s = result_data["latency_s"]

    image.save(latest_frame_path)
    records = []
    best_mask = None
    best_box = None

    if results:
        best = results[0]
        full_mask = resize_mask(best.mask, image.size)
        best_mask = full_mask
        best_box = bbox_from_mask(full_mask)
        overlay = make_overlay(image, full_mask)
        save_mask(full_mask, latest_mask_path)
        overlay.save(latest_overlay_path)

        for result_index, result in enumerate(results):
            if save_per_frame:
                mask_path = output_dir / f"frame_{frame_index:04d}_mask_{result_index:02d}.png"
                overlay_path = output_dir / f"frame_{frame_index:04d}_overlay_{result_index:02d}.png"
                full_result_mask = resize_mask(result.mask, image.size)
                save_mask(full_result_mask, mask_path)
                make_overlay(image, full_result_mask).save(overlay_path)
            else:
                mask_path = latest_mask_path
                overlay_path = latest_overlay_path

            records.append(
                result_record(
                    frame_index,
                    result,
                    mask_path,
                    overlay_path,
                    latency_s,
                    image.size,
                    segment_image.size,
                )
            )

        print(
            f"frame={frame_index} score={best.score:.3f} latency={latency_s:.2f}s "
            f"segment_size={segment_image.size} overlay={latest_overlay_path}"
        )
    else:
        print(f"frame={frame_index} no masks latency={latency_s:.2f}s")

    jsonl.write(
        json.dumps(
            {
                "frame_index": frame_index,
                "prompt": prompt,
                "service_url": service_url,
                "time": time.time(),
                "count": len(records),
                "latency_s": latency_s,
                "image_size": list(image.size),
                "segment_size": list(segment_image.size),
                "results": records,
            }
        )
        + "\n"
    )
    jsonl.flush()
    return {
        "display_path": latest_overlay_path if results else latest_frame_path,
        "image": image,
        "mask": best_mask,
        "box": best_box,
        "has_results": bool(results),
    }


def main() -> None:
    args = parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import cv2
    import numpy as np

    from agentic_robots.services.sam3.client import Sam3Client

    if args.segment_every < 1:
        raise SystemExit("--segment-every must be >= 1")

    if args.display:
        check_display_support(cv2)

    check_service(args.url)
    client = Sam3Client(url=args.url)
    capture, camera_index = open_camera(args.camera_index, args.width, args.height)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "results.jsonl"
    latest_frame_path = args.output_dir / "latest_frame.jpg"
    latest_overlay_path = args.output_dir / "latest_overlay.png"
    latest_mask_path = args.output_dir / "latest_mask.png"

    captured_count = 0
    segmented_count = 0
    latest_overlay_bgr = None
    tracker = None
    track_template_mask = None
    pending: Future | None = None
    print(
        f"Using camera index {camera_index}. Press Ctrl-C to stop. "
        f"segment_every={args.segment_every} segment_width={args.segment_width} "
        f"track={args.track}"
    )

    try:
        with (
            jsonl_path.open("a", encoding="utf-8") as jsonl,
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            while args.frames <= 0 or segmented_count < args.frames:
                ok, bgr = capture.read()
                if not ok:
                    raise SystemExit("Camera read failed")

                display_bgr = latest_overlay_bgr if latest_overlay_bgr is not None else bgr
                if args.track and tracker is not None and track_template_mask is not None:
                    tracked, box = tracker.update(bgr)
                    if tracked:
                        tracked_mask = render_tracked_mask(track_template_mask, box, bgr.shape)
                        if tracked_mask is not None:
                            display_bgr = make_overlay_bgr(bgr, tracked_mask)
                    else:
                        tracker = None
                        track_template_mask = None

                if args.display:
                    cv2.imshow("SAM3 live segment", display_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if pending is not None and pending.done():
                    handled = handle_segment_result(
                        pending.result(),
                        args.prompt,
                        args.url,
                        args.output_dir,
                        jsonl,
                        latest_frame_path,
                        latest_mask_path,
                        latest_overlay_path,
                        args.save_per_frame,
                    )
                    output_path = handled["display_path"]
                    if output_path.exists():
                        latest_overlay_bgr = cv2.imread(str(output_path))
                    if args.track and handled["mask"] is not None and handled["box"] is not None:
                        source_bgr = cv2.cvtColor(
                            np.asarray(handled["image"]), cv2.COLOR_RGB2BGR
                        )
                        tracker = initialize_tracker(cv2, source_bgr, handled["box"])
                        track_template_mask = crop_mask_template(
                            handled["mask"], handled["box"]
                        )
                    segmented_count += 1
                    pending = None

                should_segment = captured_count % args.segment_every == 0
                if pending is None and should_segment:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    pending = executor.submit(
                        segment_frame,
                        client,
                        rgb.copy(),
                        captured_count,
                        args.prompt,
                        args.max_results,
                        args.segment_width,
                        args.segment_height,
                    )

                captured_count += 1
                time.sleep(max(args.interval_s, 0.0))

            if pending is not None and (args.frames > 0 and segmented_count < args.frames):
                handled = handle_segment_result(
                    pending.result(),
                    args.prompt,
                    args.url,
                    args.output_dir,
                    jsonl,
                    latest_frame_path,
                    latest_mask_path,
                    latest_overlay_path,
                    args.save_per_frame,
                )
                if args.track and handled["mask"] is not None and handled["box"] is not None:
                    source_bgr = cv2.cvtColor(
                        np.asarray(handled["image"]), cv2.COLOR_RGB2BGR
                    )
                    tracker = initialize_tracker(cv2, source_bgr, handled["box"])
                    track_template_mask = crop_mask_template(
                        handled["mask"], handled["box"]
                    )
                segmented_count += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        capture.release()
        if args.display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    print(f"Wrote live results to {args.output_dir}")


if __name__ == "__main__":
    main()
