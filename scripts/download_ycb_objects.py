from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "lift_objects.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ycb"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected YCB object meshes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objects", nargs="+", default=["banana", "bowl", "fork", "screwdriver"])
    args = parser.parse_args()

    config = _load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.objects:
        object_config = config["objects"][name]
        object_dir = args.output_dir / object_config["ycb_id"]
        mesh_path = object_dir / object_config["mesh"]
        if mesh_path.exists():
            print(f"[{name}] already present: {mesh_path}")
            continue

        archive_path = args.output_dir / f"{object_config['ycb_id']}_google_16k.tgz"
        if not archive_path.exists():
            print(f"[{name}] downloading {object_config['url']}")
            urllib.request.urlretrieve(object_config["url"], archive_path)

        print(f"[{name}] extracting {archive_path}")
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(args.output_dir, filter="data")

        if not mesh_path.exists():
            raise FileNotFoundError(f"Expected mesh was not extracted: {mesh_path}")
        print(f"[{name}] ready: {mesh_path}")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
