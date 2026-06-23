from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("tests/service_tests/outputs/sam3_segment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 text-prompt segmentation.")
    parser.add_argument("image", type=Path, help="Path to an RGB image.")
    parser.add_argument("prompt", help='Text prompt, for example "banana" or "red cup".')
    parser.add_argument("--url", default="http://127.0.0.1:8114", help="SAM3 service URL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-results", type=int, default=5)
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


def save_mask(mask, path: Path) -> None:
    import numpy as np
    from PIL import Image

    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    image.save(path)


def save_overlay(image, mask, path: Path) -> None:
    import numpy as np
    from PIL import Image

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha = Image.fromarray(mask.astype(np.uint8) * 120, mode="L")
    red = Image.new("RGBA", base.size, (255, 40, 40, 0))
    red.putalpha(alpha)
    composed = Image.alpha_composite(base, Image.alpha_composite(overlay, red))
    composed.save(path)


def main() -> None:
    args = parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from PIL import Image

    from agentic_robots.services.sam3.client import Sam3Client

    check_service(args.url)

    image = Image.open(args.image).convert("RGB")
    client = Sam3Client(url=args.url)
    masks = client.segment(image, args.prompt)[: args.max_results]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for index, result in enumerate(masks):
        mask_path = args.output_dir / f"mask_{index:02d}.png"
        overlay_path = args.output_dir / f"overlay_{index:02d}.png"

        save_mask(result.mask, mask_path)
        save_overlay(image, result.mask, overlay_path)

        records.append(
            {
                "index": index,
                "label": result.label,
                "score": result.score,
                "box": list(result.box),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )

    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "image": str(args.image),
                "prompt": args.prompt,
                "service_url": args.url,
                "count": len(records),
                "results": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Found {len(records)} mask(s)")
    print(f"Wrote {result_path}")
    for record in records:
        print(
            f"#{record['index']} score={record['score']:.3f} "
            f"box={record['box']} overlay={record['overlay_path']}"
        )


if __name__ == "__main__":
    main()
