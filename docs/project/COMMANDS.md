# Commands And Environment Notes

## Project Path

```bash
cd /home/user/Documents/Code/agentic_robots
```

## Environment

```bash
conda activate agentic_robotics
```

Direct Python path:

```text
/home/user/miniforge3/envs/agentic_robotics/bin/python
```

## Launch Services

Launch local Qwen, SAM3, and GraspNet services:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  scripts/launch_services.py --services llm sam3 graspnet --detach --wait-s 180
```

Dry-run service launch command:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  scripts/launch_services.py --dry-run --services llm sam3 graspnet
```

## Real Lift Agent

```bash
DISPLAY=:0 XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.KKN2Q3 \
/home/user/miniforge3/envs/agentic_robotics/bin/python tests/agent_test/real_lift_agent.py \
  --object-prompt "banana" \
  --execute-real \
  --confirm-clear-workspace \
  --use-llm \
  --max-attempts 5 \
  --move-home-first
```

## Service Ports

- SAM3 service: `8114`
- GraspNet service: `8115`
- Qwen service: `8112`

## Useful Checks

Check package imports, CUDA, and RealSense:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python - <<'PY'
import importlib.util
import torch
import pyrealsense2 as rs

print("agentic_robots:", importlib.util.find_spec("agentic_robots").origin)
print("sam3:", importlib.util.find_spec("sam3").origin)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("realsense devices:", len(rs.context().query_devices()))
PY
```

Run a SAM3 segmentation smoke test:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/service_tests/sam3_segment.py /path/to/image.png "banana"
```

## Notes

- Use `--no-grasp-window` to disable the live grasp visualization window.
- Use `--grasp-window-top-k` to control how many grasp hypotheses are drawn.
