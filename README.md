# Agentic Robots

A concise reimplementation of the CAP-X style robot-agent stack.

Initial focus:

- real robot first
- Robosuite environments
- future Genesis environments
- clean service clients/servers for perception, grasping, motion, and VLMs

## Environment Setup

Create a local conda environment from a fresh checkout:

```bash
cd /home/user/Documents/Code/agentic_robots

conda create -n agentic_robots python=3.10 -y
conda activate agentic_robots

python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch first so the perception and grasping services can use CUDA. If
your machine uses a different CUDA version, choose the matching command from
the PyTorch install page.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install the third-party packages in editable mode:

```bash
python -m pip install -e third_party/sam3
python -m pip install -e third_party/contact_graspnet_pytorch
```

Then install this project and the development tools:

```bash
python -m pip install -e ".[dev]"
```

Useful environment checks:

```bash
python - <<'PY'
import importlib.util
import torch

print("agentic_robots:", importlib.util.find_spec("agentic_robots").origin)
print("sam3:", importlib.util.find_spec("sam3").origin)
print("contact_graspnet_pytorch:", importlib.util.find_spec("contact_graspnet_pytorch").origin)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
```

## Third-Party Packages

This repo expects these vendor packages to live under `third_party/`:

```text
third_party/
  sam3/
  contact_graspnet_pytorch/
```

Keep them installed with editable installs during development:

```bash
python -m pip install -e third_party/sam3
python -m pip install -e third_party/contact_graspnet_pytorch
```

Do not commit generated package output such as `build/`, `*.egg-info/`,
`__pycache__/`, checkpoints, or test outputs. Large model weights should be
downloaded separately or stored outside Git.

Contact-GraspNet expects its checkpoint at the default vendor location:

```text
third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints/model.pt
```

You can override that path when launching the service:

```bash
python -m agentic_robots.services.graspnet.server \
  --checkpoint-dir /path/to/contact_graspnet/checkpoints
```

## Services

Start the SAM3 service:

```bash
python -m agentic_robots.services.sam3.server --device cuda --port 8114
```

Start the Contact-GraspNet service:

```bash
python -m agentic_robots.services.graspnet.server --port 8115
```

Run the manual SAM3 segmentation test:

```bash
python tests/service_tests/sam3_segment.py /path/to/image.png "banana"
```

Service launch config lives in:

```text
configs/services.yaml
```

For a quick command check without starting the services:

```bash
python scripts/launch_services.py --dry-run --services sam3
```

Run the PyBullet stacking code agent:

```bash
python tests/agent_test/stacking_agent.py --no-fallback
```

The agent asks the configured foundation model for high-level Python code, then
executes only the allowlisted task APIs against the PyBullet stacking
environment.
