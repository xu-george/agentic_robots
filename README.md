# Agentic Robots

A concise reimplementation of the CAP-X style robot-agent stack.

Initial focus:

- real robot first
- Robosuite environments
- future Genesis environments
- clean service clients/servers for perception, grasping, motion, and VLMs

## Code Structure Overview

The repo is organized around a small set of reusable layers:

```text
agentic_robots/
  agent/
    code_agent.py        Generic LLM -> Python code -> runtime loop
    runtime.py           Executes generated code against allowlisted APIs
  apis/
    action/api.py        Robot-only action API: move, open, close, pose
    perception/api.py    SAM3 + GraspNet perception API
  envs/
    pybullet/            Simulation environments and FrankaSim backend
    real/franka/         Real Franka backend and libfranka bridge wrapper
  services/
    foundation_model/    Local Qwen service client/server
    sam3/                SAM3 segmentation service client/server
    graspnet/            Contact-GraspNet service client/server
    motion/pyroki/       Experimental PyRoKi motion service
configs/
  services.yaml          Local service launch config
  lift_objects.yaml      YCB lift object config
  franka_real.yaml       Real Franka host, workspace, motion, and TCP config
scripts/
  launch_services.py     Starts Qwen, SAM3, and GraspNet
tests/
  agent_test/            End-to-end stacking and lift demos
  api_test/              API and real-hardware checks
  service_tests/         Individual service smoke tests
```

The main runtime flow is:

```text
PyBullet or real robot backend
        |
ActionApi(robot) + PerceptionApi(...)
        |
CodeAgentConfig(api_functions=..., success_fn=...)
        |
CodeAgent asks Qwen for short Python code
        |
AgentRuntime executes only allowlisted API functions
        |
Task success callback checks the environment or robot state
```

`ActionApi` is intentionally robot-only, so the same generated action code can
target `FrankaSim` or `FrankaReal` as long as the robot object implements:

```text
get_end_effector_pose()
move_to_pose(position, quaternion_wxyz, ...)
open_gripper(...)
close_gripper(...)
```

Frame recording, simulation stepping, object state, and video capture stay in
the environment/backend instead of the action API.

## Demo Snapshots

Example SAM3 segmentation output:

![SAM3 segmentation overlay](tests/service_tests/outputs/sam3_segment/overlay_00.png)

Example Contact-GraspNet top grasp candidates:

![Contact-GraspNet top grasps](tests/service_tests/outputs/graspnet_plan/official_grasps_top20.png)

Generated demo videos, when recorded, are written to:

```text
tests/agent_test/outputs/stacking_agent/stacking_agent.mp4
tests/agent_test/outputs/lift_agent/lift_agent.mp4
```

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

During multi-attempt runs, `CodeAgent` resets environments that expose
`reset()` before each revision attempt. This prevents a failed first attempt
from disturbing the scene and making the corrected attempt run in a damaged
state.

## Real Franka Connection

The real Franka backend lives in:

```text
agentic_robots/envs/real/franka/franka_real.py
agentic_robots/envs/real/franka/franka_action_bridge/
configs/franka_real.yaml
```

`FrankaReal` talks to the robot through a small C++ libfranka bridge. Build it
from the repo root:

```bash
cd /home/user/Documents/Code/agentic_robots
cmake -S agentic_robots/envs/real/franka/franka_action_bridge \
  -B agentic_robots/envs/real/franka/franka_action_bridge/build \
  -DFranka_DIR=third_party/libfranka/build
cmake --build agentic_robots/envs/real/franka/franka_action_bridge/build
```

Real robot connection and safety limits are configured in:

```text
configs/franka_real.yaml
```

Current defaults:

```yaml
host: 172.16.0.2

motion:
  default_duration_s: 5.0
  max_translation_m: 0.05
  orientation_tolerance: 0.005

workspace:
  x: [0.20, 0.80]
  y: [-0.45, 0.45]
  z: [0.20, 0.80]

tcp:
  offset_from_ee: [0.0, 0.0, 0.105]
```

Run a read-only connection check first. This pings the robot, verifies the
bridge, reads the current TCP pose, and reads the gripper state:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/api_test/franka_real_hardware_sequence.py \
  --host 172.16.0.2
```

If the workspace is clear, the user stop is reachable, and the robot is ready,
run a small guarded real-motion test:

```bash
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/api_test/franka_real_hardware_sequence.py \
  --host 172.16.0.2 \
  --execute-real \
  --confirm-clear-workspace \
  --dz 0.01
```

Real motion is disabled by default in `FrankaReal`. Code must explicitly
construct it with `motion_enabled=True`, and bridge-side workspace limits still
guard every Cartesian move.

Minimal real-backend usage:

```python
from agentic_robots.apis.action import ActionApi
from agentic_robots.envs.real import FrankaReal

robot = FrankaReal(host="172.16.0.2", motion_enabled=False)
robot.load()

action_api = ActionApi(robot)
pose = action_api.get_end_effector_pose()
print(pose.position, pose.quaternion_wxyz)
```

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
  scripts/launch_services.py --services llm sam3 graspnet --detach
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/agent_test/lift_agent.py \
  --object screwdriver \
  --no-fallback \
  --show-grasp-pose \
  --record-video \
  --max-attempts 3 \
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
  --max-attempts 3
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
  --max-attempts 3
```

`--show-grasp-pose` refreshes the PyBullet debug marker every time the LLM code
calls `get_best_grasp_pose()`. It draws the latest top 5 GraspNet candidates,
highlights the best raw GraspNet marker, and returns the converted executable
Panda pose to the generated code.

The Contact-GraspNet raw grasp transform is a gripper-frame origin, not the
Franka fingertip/TCP point. `PerceptionApi` converts it to the executable Panda
target with the local GraspNet-to-Panda offset from the vendored gripper
geometry:

```text
third_party/contact_graspnet_pytorch/gripper_control_points/panda_gripper_coords.yml
gripper_left_tip_flat / gripper_right_tip_flat z ~= 0.112204 m
```

For a robot-only smoke test without GPU perception services, add
`--fake-perception`.
