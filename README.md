# Agentic Robots

A concise reimplementation of the CAP-X style robot-agent stack.

Initial focus:

- real robot first
- Robosuite environments
- future Genesis environments
- clean service clients/servers for perception, grasping, motion, and VLMs

## Environment

The development environment is available as a named conda environment:

```bash
conda activate agentic_robot
```

It lives at:

```text
/home/binzhao/miniconda3/envs/agentic_robot
```

This environment was cloned from CAP-X's working environment so it already has
the GPU stack and editable SAM3 package available.

The original project-local clone is still available at
`/home/binzhao/Documents/Code/agentic_robots/.conda`, but use the named
environment for normal work.

Useful checks:

```bash
python - <<'PY'
import importlib.util
import torch

print("agentic_robots:", importlib.util.find_spec("agentic_robots").origin)
print("sam3:", importlib.util.find_spec("sam3").origin)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
```

Start the SAM3 service:

```bash
python -m agentic_robots.services.sam3.server --device cuda --port 8114
```

Run the manual SAM3 segmentation test:

```bash
python tests/service_tests/sam3_segment.py /path/to/image.png "banana"
```

## Services And Agent Runner

Service launch config lives in:

```text
configs/services.yaml
```

Launch SAM3 and the local Qwen LLM service:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  scripts/launch_services.py --services sam3 llm
```

For a quick command check without starting the models:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  scripts/launch_services.py --dry-run --services sam3 llm
```

Run the PyBullet stacking code agent:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/agent_test/stacking_agent.py --no-fallback
```

The agent asks the Qwen service for high-level Python code, then executes only
the allowlisted task APIs against the PyBullet stacking environment.
