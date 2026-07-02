# Agentic Robots APIs

This package contains high-level robot APIs built on top of lower-level
services. The design follows the CAP-X pattern: API classes expose Python
functions through `functions()`, and each function docstring is suitable for
LLM-generated robot code or a future REST wrapper.

## Perception

`agentic_robots.apis.perception.PerceptionApi` wraps the current SAM3 and
GraspNet services.

Exposed functions:

```text
get_observation
segment_object
segment_point
get_object_pose
mask_to_3d_points
get_oriented_bounding_box_from_3d_points
plan_grasp
get_finger_grasp
```

Current responsibilities:

```text
RGB image + text prompt -> segmentation mask
RGB-D + text prompt -> object 3D pose estimate
depth + intrinsics + segmentation -> grasp candidates
depth + intrinsics + segmentation -> finger-tip grasp target pose
```

Grasp frame convention:

```text
plan_grasp returns raw Contact-GraspNet Panda gripper/control-point
transforms whose origin is the Panda hand/wrist base (NOT the finger tips).
PerceptionApi.get_finger_grasp() applies a +0.105 m local-Z offset by
default so the returned pose is the project grasp point between the two
Panda finger tips, i.e. where the Franka hand link should move to. This
matches PyBullet link 11, panda_grasptarget. Real robot experiments can
pass apply_grasp_target_offset=False when FrankaReal.get_end_effector_pose()
should command the raw GraspNet hand-base frame.
```

The perception API does not execute robot motion. The future `action` API should
consume perception outputs and handle IK, trajectory planning, gripper commands,
and lower-level robot actions used by task code.

## Action

`agentic_robots.apis.action.ActionApi` wraps a robot object directly. The same
thin API can call the PyBullet `FrankaSim` backend or the real Franka backend in
`agentic_robots.envs.real.FrankaReal`, as long as the robot exposes the action
methods below.

Exposed functions:

```text
get_end_effector_pose
move_to_pose
open_gripper
close_gripper
```

The action API intentionally does not expose task-level shortcuts such as
`pick_object` or `place_object_on`. The agent must compose those behaviors from
lower-level robot actions.

Build the real bridge from the repository root:

```bash
cmake -S agentic_robots/envs/real/franka/franka_action_bridge \
  -B agentic_robots/envs/real/franka/franka_action_bridge/build \
  -DFranka_DIR=third_party/libfranka/build
cmake --build agentic_robots/envs/real/franka/franka_action_bridge/build
```

Real Franka runtime safety values live in:

```text
configs/franka_real.yaml
```

The Python `FrankaReal` wrapper reads that file and passes the configured
workspace bounds to the C++ bridge for every `move-relative` call. The same
file also defines `tcp.offset_from_ee`, which is applied in the current
end-effector frame before `get_end_effector_pose()` reports the command point.
The current real config uses `[0, 0, 0]`, so the real-lift test requests the raw
Contact-GraspNet gripper frame instead of the grasp-point offset.

Read-only real backend test:

```bash
python tests/api_test/franka_real_hardware_sequence.py
```
