from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "services.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Agentic Robots services from config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--services", nargs="+", default=["sam3", "llm"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--detach", action="store_true", help="Leave services running and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching.")
    parser.add_argument("--wait-s", type=float, default=60.0)
    args = parser.parse_args()

    config = _load_config(args.config)
    processes: list[subprocess.Popen] = []

    for name in args.services:
        service = config["services"][name]
        health_url = service.get("health_url")
        if not args.dry_run and _is_healthy(health_url):
            print(f"[{name}] already healthy: {health_url}")
            continue
        command = _build_command(args.python, service)
        print(f"[{name}] {' '.join(command)}")
        if args.dry_run:
            continue
        processes.append(_start_service(name, service, command, detach=args.detach))
        _wait_for_health(name, health_url, timeout_s=args.wait_s)

    if args.dry_run or args.detach:
        return

    print("Services are running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
            failed = [proc for proc in processes if proc.poll() is not None]
            if failed:
                failed_codes = [proc.returncode for proc in failed]
                raise RuntimeError(
                    f"{len(failed)} service process(es) exited unexpectedly: {failed_codes}"
                )
    except KeyboardInterrupt:
        print("Stopping services...")
    finally:
        for proc in processes:
            _terminate_process(proc)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_command(python: str, service: dict[str, Any]) -> list[str]:
    command = [
        python,
        "-m",
        str(service["module"]),
        "--host",
        str(service.get("host", "127.0.0.1")),
        "--port",
        str(service["port"]),
    ]
    for key, value in (service.get("args") or {}).items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    return command


def _start_service(
    name: str,
    service: dict[str, Any],
    command: list[str],
    *,
    detach: bool,
) -> subprocess.Popen:
    log_path = PROJECT_ROOT / service.get("log_file", f"tests/service_tests/outputs/services/{name}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=detach,
    )


def _wait_for_health(name: str, health_url: str | None, *, timeout_s: float) -> None:
    if not health_url:
        return
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(health_url, timeout=2.0)
            if response.ok and response.json().get("ok", True):
                print(f"[{name}] healthy: {health_url}")
                return
            last_error = RuntimeError(f"{response.status_code}: {response.text[:200]}")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"{name} did not become healthy at {health_url}: {last_error}")


def _is_healthy(health_url: str | None) -> bool:
    if not health_url:
        return False
    try:
        response = requests.get(health_url, timeout=2.0)
    except requests.RequestException:
        return False
    return bool(response.ok and response.json().get("ok", True))


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        os.kill(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
