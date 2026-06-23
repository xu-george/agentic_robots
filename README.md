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
conda activate agentic_robotics
```

It lives at:

```text
/home/user/miniforge3/envs/agentic_robotics
```

This environment was cloned from CAP-X's working environment so it already has
the GPU stack and editable SAM3 package available.

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

Start the SAM3 service directly:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  -m agentic_robots.services.sam3.server --device cuda --port 8114
```

Run the manual SAM3 segmentation test:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/service_tests/sam3_segment.py /path/to/image.png "banana"
```

## Services

Service launch config lives in:

```text
configs/services.yaml
```

Launch the local Qwen LLM, SAM3, and GraspNet services:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  scripts/launch_services.py --services llm sam3 graspnet --detach --wait-s 180
```

For a quick command check without starting the models:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  scripts/launch_services.py --dry-run --services llm sam3 graspnet
```

## Unified Code Agent

The task-specific agent classes and task factory branches were removed. The
agent core is now only a generic LLM/code-execution loop in:

```text
agentic_robots/agent/code_agent.py
```

Use it by passing an environment, a natural-language task description, the
allowlisted APIs available to generated code, and task-local success/diagnostic
callbacks:

```python
from agentic_robots.agent import CodeAgent, CodeAgentConfig

config = CodeAgentConfig(
    task_description="Stack the red cube on top of the blue cube.",
    api_functions=api_functions,
    success_fn=env.is_success,
    scene_context_fn=lambda: {"objects": env.get_ground_truth()},
    diagnostics_fn=lambda: {"robot": env.get_obs()["robot"]},
    fallback_code=None,
)
agent = CodeAgent(env, config)
result = agent.run()
```

`CodeAgent` has no hardcoded `lift` or `stacking` definitions. The demo scripts
show how to configure the same agent for different environments and tasks.

## Test Demo Commands

Run these from the repo root:

```bash
cd /home/user/Documents/Code/agentic_robots
```

Stacking with the real Qwen service:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/agent_test/stacking_agent.py \
  --no-fallback \
  --max-attempts 3
```

Stacking with GUI and video:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/agent_test/stacking_agent.py \
  --gui \
  --record-video \
  --max-attempts 3 \
  --hold-s 10
```

Lift smoke test without GPU perception services:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/agent_test/lift_agent.py \
  --object banana \
  --fake-perception \
  --max-attempts 1
```

Lift with real Qwen, SAM3, GraspNet, and refreshed top-5 grasp markers:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  scripts/launch_services.py --services llm sam3 graspnet --detach --wait-s 180
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/agent_test/lift_agent.py \
  --object screwdriver \
  --no-fallback \
  --show-grasp-pose \
  --record-video \
  --max-attempts 1 \
  --hold-s 10
```

Change `--object` to `banana`, `bowl`, `fork`, or `screwdriver`.

## Stacking Demo

Run the PyBullet stacking task through the general code agent:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/agent_test/stacking_agent.py --no-fallback
```

The agent asks the Qwen service for high-level Python code, then executes only
the allowlisted task APIs against the PyBullet stacking environment.

Optional GUI/video run:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/agent_test/stacking_agent.py \
  --gui \
  --record-video \
  --max-attempts 3 \
  --hold-s 10
```

## YCB Lift Demo

Download the supported YCB meshes:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  scripts/download_ycb_objects.py --objects banana bowl fork screwdriver
```

Run the YCB lift pipeline with SAM3, GraspNet, Qwen, and PyBullet:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  scripts/launch_services.py --services llm sam3 graspnet --detach --wait-s 180
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/agent_test/lift_agent.py \
  --object banana \
  --no-fallback \
  --show-grasp-pose \
  --record-video \
  --max-attempts 1
```

Supported lift objects are configured in:

```text
configs/lift_objects.yaml
```

Current object names:

```text
banana
bowl
fork
screwdriver
```

Run another object by changing `--object`:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/agent_test/lift_agent.py \
  --object screwdriver \
  --no-fallback \
  --show-grasp-pose \
  --record-video \
  --max-attempts 1
```

`--show-grasp-pose` refreshes the PyBullet debug marker every time the LLM code
calls `get_best_grasp_pose()`. It draws the latest top 5 GraspNet candidates,
highlights the best raw GraspNet marker, and returns the converted executable
Panda pose to the generated code.

For a robot-only smoke test without GPU perception services, add
`--fake-perception`.
