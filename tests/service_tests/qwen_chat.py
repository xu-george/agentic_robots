from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("tests/service_tests/outputs/qwen_chat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple local Qwen chat test.")
    parser.add_argument("--url", default="http://127.0.0.1:8112", help="Qwen service URL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: qwen service ok",
        help="Prompt to send to the local Qwen service.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    return parser.parse_args()


def check_service(url: str) -> None:
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=5.0)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Qwen service is not reachable at {url}. "
            "Start it with: python -m agentic_robots.services.foundation_model.qwen.server"
        ) from exc


def main() -> None:
    args = parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.services.foundation_model.qwen.client import QwenClient

    check_service(args.url)

    client = QwenClient(url=args.url, timeout_s=240.0)
    result = client.complete(
        args.prompt,
        system="You are a concise robot assistant. Follow the user's requested output exactly.",
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "service_url": args.url,
                "model": result.model,
                "prompt": args.prompt,
                "content": result.content,
                "done": result.done,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Model: {result.model}")
    print(f"Response: {result.content}")
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
