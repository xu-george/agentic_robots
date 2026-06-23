from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPX_GRASPNET_ROOT = (
    PROJECT_ROOT / "third_party" / "contact_graspnet_pytorch"
)
DEFAULT_SAMPLE = CAPX_GRASPNET_ROOT / "test_data" / "0.npy"
DEFAULT_OUTPUT_DIR = Path("tests/service_tests/outputs/graspnet_plan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Contact-GraspNet service planning.")
    parser.add_argument(
        "sample",
        nargs="?",
        type=Path,
        default=DEFAULT_SAMPLE,
        help="Path to a .npy/.npz sample with depth, K, and seg/segmap.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8115", help="GraspNet service URL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--segmap-id", type=int, default=None)
    parser.add_argument("--forward-passes", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def check_service(url: str) -> None:
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=3.0)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"GraspNet service is not reachable at {url}. "
            "Start it with: python -m agentic_robots.services.graspnet.server"
        ) from exc


def load_sample(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        item = data
    elif data.shape == ():
        item = data.item()
    else:
        raise SystemExit(f"{path} must contain a dict-like sample, got shape {data.shape}")

    depth = np.asarray(item["depth"], dtype=np.float32)
    cam_k = np.asarray(item["K"], dtype=np.float64).reshape(3, 3)
    segmap = np.asarray(item.get("segmap", item.get("seg")))
    rgb = np.asarray(item["rgb"], dtype=np.uint8) if "rgb" in item else None

    if segmap is None:
        raise SystemExit(f"{path} does not contain 'seg' or 'segmap'")
    return depth, cam_k, segmap, rgb


def choose_segmap_id(segmap: np.ndarray) -> int:
    ids, counts = np.unique(segmap.astype(np.int64), return_counts=True)
    candidates = [(int(seg_id), int(count)) for seg_id, count in zip(ids, counts) if seg_id > 0]
    if not candidates:
        raise SystemExit("Segmap has no positive object ids")
    return max(candidates, key=lambda item: item[1])[0]


def save_rgb(rgb: np.ndarray | None, path: Path) -> str | None:
    if rgb is None:
        return None

    Image.fromarray(rgb).save(path)
    return str(path)


def depth_to_point_cloud(
    depth: np.ndarray,
    cam_k: np.ndarray,
    rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    y, x = np.where(depth > 0)
    z = depth[y, x]
    world_x = (x - cam_k[0, 2]) * z / cam_k[0, 0]
    world_y = (y - cam_k[1, 2]) * z / cam_k[1, 1]
    pc = np.stack([world_x, world_y, z], axis=-1).astype(np.float32)
    colors = rgb[y, x].astype(np.uint8) if rgb is not None else None
    return pc, colors


def project(points: np.ndarray, cam_k: np.ndarray, image_size: tuple[int, int]):
    width, height = image_size
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    uvw = (cam_k @ points.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
    valid = (
        (points[:, 2] > 1e-6)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv, valid


def gripper_wire_points(grasp: np.ndarray, opening: float = 0.08) -> np.ndarray:
    if str(CAPX_GRASPNET_ROOT) not in sys.path:
        sys.path.append(str(CAPX_GRASPNET_ROOT))
    import contact_graspnet_pytorch.mesh_utils as mesh_utils

    gripper = mesh_utils.create_gripper("panda")
    control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5 * (control_points[1, :] + control_points[2, :])
    line_points = np.array(
        [
            np.zeros((3,)),
            mid_point,
            control_points[1],
            control_points[3],
            control_points[1],
            control_points[2],
            control_points[4],
        ],
        dtype=np.float32,
    )
    line_points[2:, 0] = np.sign(line_points[2:, 0]) * opening / 2.0
    return line_points @ grasp[:3, :3].T + grasp[:3, 3]


def save_official_style_outputs(
    *,
    output_dir: Path,
    sample_path: Path,
    depth: np.ndarray,
    cam_k: np.ndarray,
    segmap: np.ndarray,
    rgb: np.ndarray | None,
    segmap_id: int,
    grasps: np.ndarray,
    scores: np.ndarray,
    contact_points: np.ndarray,
) -> dict[str, str | None]:
    pc_full, pc_colors = depth_to_point_cloud(depth, cam_k, rgb)

    pred_grasps_cam = {segmap_id: grasps}
    scores_dict = {segmap_id: scores}
    contact_pts_dict = {segmap_id: contact_points}

    prediction_path = output_dir / f"predictions_{sample_path.stem}.npz"
    np.savez(
        prediction_path,
        pc_full=pc_full,
        pred_grasps_cam=pred_grasps_cam,
        scores=scores_dict,
        contact_pts=contact_pts_dict,
        pc_colors=pc_colors,
    )

    seg_overlay_path = output_dir / "official_debug_rgb_seg.png"
    render_path = output_dir / "official_grasps_top20.png"
    if rgb is None:
        return {
            "official_prediction_path": str(prediction_path),
            "official_debug_rgb_seg_path": None,
            "official_grasp_render_path": None,
        }

    image = Image.fromarray(rgb).convert("RGBA")
    mask = segmap.astype(np.int64) == segmap_id
    if mask.shape[:2] == rgb.shape[:2] and mask.any():
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        alpha = Image.fromarray((mask.astype(np.uint8) * 85), mode="L")
        color = Image.new("RGBA", image.size, (40, 170, 255, 0))
        color.putalpha(alpha)
        image = Image.alpha_composite(image, color)
    image.convert("RGB").save(seg_overlay_path)

    render = image.copy()
    draw = ImageDraw.Draw(render)
    order = np.argsort(scores)[::-1]
    top = order[:20]
    min_score = float(scores[top].min()) if top.size else 0.0
    max_score = float(scores[top].max()) if top.size else 1.0

    for idx in top[::-1]:
        points = gripper_wire_points(grasps[idx])
        uv, valid = project(points, cam_k, render.size)
        if not valid.any():
            continue
        ratio = (float(scores[idx]) - min_score) / max(max_score - min_score, 1e-8)
        color = (
            int(80 + 175 * ratio),
            int(220 * (1.0 - ratio)),
            int(255 * (1.0 - ratio)),
            230,
        )
        for start, end in zip(range(0, 6), range(1, 7)):
            if valid[start] and valid[end]:
                draw.line([tuple(uv[start]), tuple(uv[end])], fill=color, width=3)

    if order.size:
        best = int(order[0])
        points = gripper_wire_points(grasps[best])
        uv, valid = project(points, cam_k, render.size)
        for start, end in zip(range(0, 6), range(1, 7)):
            if valid[start] and valid[end]:
                draw.line([tuple(uv[start]), tuple(uv[end])], fill=(255, 40, 40, 255), width=5)
    render.convert("RGB").save(render_path)

    return {
        "official_prediction_path": str(prediction_path),
        "official_debug_rgb_seg_path": str(seg_overlay_path),
        "official_grasp_render_path": str(render_path),
    }


def score_to_heat_color(values: np.ndarray) -> np.ndarray:
    """Small jet-like colormap without adding a matplotlib dependency."""
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def save_grasp_value_heatmap(
    *,
    output_dir: Path,
    rgb: np.ndarray | None,
    cam_k: np.ndarray,
    segmap: np.ndarray,
    segmap_id: int,
    scores: np.ndarray,
    contact_points: np.ndarray,
) -> str | None:
    if rgb is None or scores.size == 0 or contact_points.size == 0:
        return None

    height, width = rgb.shape[:2]
    points = np.asarray(contact_points, dtype=np.float64).reshape(-1, 3)
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    count = min(len(points), len(score_values))
    points = points[:count]
    score_values = score_values[:count]

    uv, valid = project(points, cam_k, (width, height))
    uv = uv[valid]
    score_values = score_values[valid]
    if uv.size == 0:
        return None

    score_min = float(score_values.min())
    score_max = float(score_values.max())
    normalized_scores = (score_values - score_min) / max(score_max - score_min, 1e-8)

    yy, xx = np.mgrid[0:height, 0:width]
    heat = np.zeros((height, width), dtype=np.float32)
    sigma = 12.0
    radius = int(3 * sigma)
    for (u, v), value in zip(uv, normalized_scores):
        cx, cy = int(round(u)), int(round(v))
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        dist2 = (xx[y0:y1, x0:x1] - u) ** 2 + (yy[y0:y1, x0:x1] - v) ** 2
        blob = np.exp(-dist2 / (2.0 * sigma * sigma)) * float(value)
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], blob)

    if heat.max() <= 0:
        return None
    heat = heat / heat.max()

    mask = segmap.astype(np.int64) == segmap_id
    if mask.shape[:2] == heat.shape:
        heat = np.where(mask, heat, 0.0)

    base = rgb.astype(np.float32) / 255.0
    color = score_to_heat_color(heat)
    alpha = np.clip(heat * 0.75, 0.0, 0.75)[..., None]
    blended = (base * (1.0 - alpha) + color * alpha) * 255.0

    heatmap_path = output_dir / "grasp_value_heatmap.png"
    Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).save(heatmap_path)
    return str(heatmap_path)


def main() -> None:
    args = parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.services.graspnet.client import GraspNetClient

    check_service(args.url)

    depth, cam_k, segmap, rgb = load_sample(args.sample)
    segmap_id = args.segmap_id if args.segmap_id is not None else choose_segmap_id(segmap)

    client = GraspNetClient(url=args.url, timeout_s=240.0)
    plan = client.plan_from_depth(
        depth,
        cam_k,
        segmap,
        segmap_id=segmap_id,
        forward_passes=args.forward_passes,
        max_retries=args.max_retries,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "grasps.npy", plan.grasps)
    np.save(args.output_dir / "scores.npy", plan.scores)
    np.save(args.output_dir / "contact_points.npy", plan.contact_points)
    rgb_path = save_rgb(rgb, args.output_dir / "sample_rgb.png")
    official_paths = save_official_style_outputs(
        output_dir=args.output_dir,
        sample_path=args.sample,
        depth=depth,
        cam_k=cam_k,
        segmap=segmap,
        rgb=rgb,
        segmap_id=segmap_id,
        grasps=plan.grasps,
        scores=plan.scores,
        contact_points=plan.contact_points,
    )
    heatmap_path = save_grasp_value_heatmap(
        output_dir=args.output_dir,
        rgb=rgb,
        cam_k=cam_k,
        segmap=segmap,
        segmap_id=segmap_id,
        scores=plan.scores,
        contact_points=plan.contact_points,
    )

    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "sample": str(args.sample),
                "service_url": args.url,
                "segmap_id": segmap_id,
                "grasp_count": int(len(plan.grasps)),
                "grasps_shape": list(plan.grasps.shape),
                "scores_shape": list(plan.scores.shape),
                "contact_points_shape": list(plan.contact_points.shape),
                "best_score": float(plan.scores.max()) if plan.scores.size else None,
                "rgb_path": rgb_path,
                "grasps_path": str(args.output_dir / "grasps.npy"),
                "scores_path": str(args.output_dir / "scores.npy"),
                "contact_points_path": str(args.output_dir / "contact_points.npy"),
                "grasp_value_heatmap_path": heatmap_path,
                **official_paths,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Found {len(plan.grasps)} grasp(s)")
    print(f"Wrote {result_path}")
    print(f"Wrote official-style predictions: {official_paths['official_prediction_path']}")
    print(f"Wrote official-style render: {official_paths['official_grasp_render_path']}")
    print(f"Wrote grasp value heatmap: {heatmap_path}")
    if plan.scores.size:
        print(f"Best score: {float(plan.scores.max()):.3f}")


if __name__ == "__main__":
    main()
