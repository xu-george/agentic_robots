"""ROS RGB -> native SAM3 service. Click frozen first frame; r resets, q quits."""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agentic_robots.services.base import decode_mask
from agentic_robots.services.sam3.tracking_client import track_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/camera/d405/color/image_raw")
    parser.add_argument("--url", default="http://127.0.0.1:8116")
    parser.add_argument('--prompt', help='Text prompt for initial segmentation, e.g. black handle')
    parser.add_argument('--min-score', type=float, default=0.3)
    args = parser.parse_args()
    url = args.url.rstrip("/")
    requests.get(url + "/health", timeout=3).raise_for_status()
    rclpy.init()
    node = rclpy.create_node("sam3_native_live_viewer")
    latest = None
    sequence = 0
    selected = []
    frozen = None
    session = None
    pending = None
    shown = None
    submitted = -1
    status = "Waiting for camera"
    approved = False
    prompt_sent = False
    pool = ThreadPoolExecutor(max_workers=1)

    def receive(msg):
        nonlocal latest, sequence
        if msg.encoding not in ("rgb8", "bgr8"):
            return
        image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        image = image[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        latest = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else image.copy()
        sequence += 1

    sub = node.create_subscription(Image, args.topic, receive, qos_profile_sensor_data)
    window = "Native SAM3 tracking: click object | r reset | q quit"
    cv2.namedWindow(window)

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and frozen is not None and session is None and pending is None and not args.prompt:
            selected[:] = [x, y]

    cv2.setMouseCallback(window, click)

    def infer(frame, session_id, point):
        started = time.monotonic()
        result = track_rgb(
            url, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), session_id=session_id,
            text_prompt=args.prompt, point_xy=None if args.prompt else point,
            min_score=args.min_score)
        mask = decode_mask(result["mask_base64"], tuple(result["shape"]))
        overlay = frame.copy()
        overlay[mask] = (0, 80, 255)
        return result, cv2.addWeighted(frame, 0.55, overlay, 0.45, 0), time.monotonic() - started

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
            if latest is None:
                time.sleep(0.01)
                continue
            if frozen is None:
                frozen = latest.copy()
                shown = frozen.copy()
                status = f'Segmenting: {args.prompt}' if args.prompt else 'Click object on this frozen frame'
            if pending is not None and pending.done():
                try:
                    result, shown, latency = pending.result()
                    session = result["session_id"]
                    if result['frame_index'] == 0:
                        score = result.get('score')
                        print(f'Initial segmentation: prompt={args.prompt!r}, score={score}. SPACE accepts; R retries.', flush=True)
                        status = 'Check mask: SPACE accept | R retry | Q quit'
                    else:
                        status = f'Native SAM3 | frame {result["frame_index"]} | {1/latency:.1f} FPS'
                except Exception as exc:
                    print(exc, flush=True)
                    status = "Tracking stopped; press r to select again"
                    session = None
                    selected.clear()
                pending = None
            if pending is None:
                if session is None and (selected or (args.prompt and not prompt_sent)):
                    pending = pool.submit(infer, frozen.copy(), None, selected.copy())
                    prompt_sent = True
                    selected.clear()
                    submitted = sequence
                elif session is not None and approved and sequence != submitted:
                    pending = pool.submit(infer, latest.copy(), session, None)
                    submitted = sequence
            display = shown.copy()
            cv2.putText(display, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xff
            if key == ord("q"):
                break
            if key == ord(' ') and session is not None:
                approved = True
            if key == ord("r") and pending is None:
                if session:
                    requests.post(url + "/track/reset", json={"session_id": session}, timeout=5).raise_for_status()
                frozen = None
                session = None
                approved = False
                prompt_sent = False
                selected.clear()
    finally:
        pool.shutdown(wait=True)
        # Include a session created by an in-flight start request before quit.
        if pending is not None:
            try:
                session = pending.result()[0]["session_id"]
            except Exception:
                pass
        if session:
            try:
                requests.post(url + "/track/reset", json={"session_id": session}, timeout=5)
            except requests.RequestException:
                pass
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
