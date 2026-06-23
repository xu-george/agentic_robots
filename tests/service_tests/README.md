# Service Tests

This folder contains manual service tests. The tests live under the root
`tests/` folder, but outside the `agentic_robots/` Python package. They call the
service clients the same way the future API layer will.

Use the project conda environment:

```bash
cd /home/binzhao/Documents/Code/agentic_robots
conda activate agentic_robot
```

## API Tests

High-level API tests live in:

```text
tests/api_test/
```

The perception API is the first high-level API layer above the low-level
services. It follows the CAP-X `ApiBase.functions()` pattern and exposes Python
functions that can later be wrapped as REST endpoints or injected into generated
robot code.

Current perception functions:

```text
get_observation
segment_object
segment_point
get_object_pose
mask_to_3d_points
get_oriented_bounding_box_from_3d_points
plan_grasp
```

Run the lightweight smoke test without starting GPU services:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/api_test/perception_api.py
```

This test uses fake SAM3 and GraspNet clients to verify the API wiring, RGB-D
projection, object pose output, grasp-plan output, and generated API docs.

## PyBullet Sim Tests

Lightweight PyBullet simulation code lives under:

```text
agentic_robots/envs/pybullet/
```

Current files:

```text
franka_sim.py    -> loads the Panda robot and exposes action-style controls
stacking_env.py  -> creates table/cubes/robot and returns task observations
```

Camera information is combined on the task environment, not inside the
observation:

```text
env.camera_info.intrinsics
env.camera_info.extrinsics
env.camera_info.view_matrix
env.camera_info.projection_matrix
```

The observation contains images, robot state, and ground-truth values that can
stand in for fake perception API output:

```text
obs["images"]["rgb"]
obs["images"]["depth"]
obs["images"]["segmentation"]
obs["robot"]["joint_positions"]
obs["robot"]["eef_position"]
obs["robot"]["eef_quaternion_wxyz"]
obs["ground_truth"]["objects"]
```

Run the smoke test:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/sim_test/pybullet_stacking_env.py
```

## SAM3 Segmentation

The SAM3 package is vendored inside this project:

```text
sam3 -> /home/binzhao/Documents/Code/agentic_robots/third_party/sam3/sam3
```

Start the SAM3 service in one terminal:

```bash
cd /home/binzhao/Documents/Code/agentic_robots
conda activate agentic_robots
python -m agentic_robots.services.sam3.server --device cuda --port 8114
```

Run segmentation from another terminal:

```bash
cd /home/binzhao/Documents/Code/agentic_robots
conda activate agentic_robot
python tests/service_tests/sam3_segment.py /path/to/image.png "banana"
```

Outputs are written to:

```text
tests/service_tests/outputs/sam3_segment/
```

Output files:

```text
results.json
mask_00.png
overlay_00.png
...
```

Clean old static SAM3 outputs when you want a fresh run:

```bash
gio trash tests/service_tests/outputs/sam3_segment/results.json \
  tests/service_tests/outputs/sam3_segment/mask_*.png \
  tests/service_tests/outputs/sam3_segment/overlay_*.png
```

## RealSense Example

A RealSense frame was captured through OpenCV because `pyrealsense2` is not
installed in this environment. The current saved frame is:

```text
tests/service_tests/outputs/realsense_capture/current_realsense.jpg
```

Run SAM3 on that captured frame:

```bash
cd /home/binzhao/Documents/Code/agentic_robots
conda activate agentic_robot
python tests/service_tests/sam3_segment.py \
  tests/service_tests/outputs/realsense_capture/current_realsense.jpg \
  "banana"
```

The previous static SAM3 result files have been cleared. A successful historical
banana run from the RealSense frame produced:

```text
prompt: banana
best score: 0.910
best box: [900.85, 386.37, 985.94, 591.19]
```

## Live RealSense Segmentation

The live camera test uses OpenCV `VideoCapture`, which works with the RealSense
color stream exposed as a Linux video device. Start the SAM3 service first, then
run a short live segmentation test from another terminal:

```bash
cd /home/binzhao/Documents/Code/agentic_robots
conda activate agentic_robot
python tests/service_tests/sam3_live_segment.py "banana" \
  --frames 10 \
  --segment-every 5 \
  --segment-width 640
```

Run continuously until `Ctrl-C`:

```bash
python tests/service_tests/sam3_live_segment.py "banana"
```

Show a live OpenCV overlay window:

```bash
python tests/service_tests/sam3_live_segment.py "banana" \
  --display \
  --track \
  --segment-every 5 \
  --segment-width 640
```

The display loop is asynchronous: the camera window keeps refreshing while a
single background SAM3 worker segments the latest submitted frame. Lower
`--segment-width` for faster segmentation, increase `--segment-every` to run
SAM3 less often, and add `--save-per-frame` only when you need every overlay
written to disk.

Add `--track` for smoother interactive viewing. In this mode SAM3 refreshes the
mask every `--segment-every` frames, then OpenCV tracks that object between SAM3
refreshes and renders an approximate tracked mask on every camera frame.

`--display` requires a GUI-enabled OpenCV build. The `agentic_robot`
environment must use `opencv-python`, not `opencv-python-headless`, because the
headless package cannot open `cv2.imshow` windows.

If you want to force a specific RealSense video node, add `--camera-index 2`
or `--camera-index 4`. Both were readable on this machine during the setup
check.

Outputs are written to:

```text
tests/service_tests/outputs/sam3_live_segment/
```

Live output files:

```text
latest_frame.jpg
latest_mask.png
latest_overlay.png
results.jsonl
frame_0000_mask_00.png
frame_0000_overlay_00.png
...
```

## Contact-GraspNet Planning

The GraspNet service uses the vendored Contact-GraspNet checkout and checkpoint:

```text
/home/user/Documents/Code/agentic_robots/third_party/contact_graspnet_pytorch
```

Start the GraspNet service in one terminal:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  -m agentic_robots.services.graspnet.server --port 8115
```

Run the manual planning test from another terminal:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/service_tests/graspnet_plan.py --forward-passes 1 --max-retries 3
```

By default the test uses the bundled Contact-GraspNet sample:

```text
/home/user/Documents/Code/agentic_robots/third_party/contact_graspnet_pytorch/test_data/0.npy
```

Outputs are written to:

```text
tests/service_tests/outputs/graspnet_plan/
```

Output files:

```text
results.json
sample_rgb.png
grasps.npy
scores.npy
contact_points.npy
predictions_0.npz
official_debug_rgb_seg.png
official_grasps_top20.png
```

## PyRoKi Motion Planning

PyRoKi is the robot kinematics service. It does not use camera images, depth, or
segmentation masks. It takes a robot end-effector target pose and returns robot
joint positions.

The service currently loads the Franka Panda model by default:

```text
robot: panda_description
target link: panda_hand
collision asset: agentic_robots/services/motion/pyroki/assets/panda_spheres.json
```

Start the PyRoKi service in one terminal:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  -m agentic_robots.services.motion.pyroki.server --port 8116
```

Run the simple IK/planning test from another terminal:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/service_tests/pyroki_motion.py --timesteps 5
```

The simple test uses two hard-coded end-effector poses:

```text
start_pose_wxyz_xyz = [1.0, 0.0, 0.0, 0.0, 0.45, 0.0, 0.45]
end_pose_wxyz_xyz   = [1.0, 0.0, 0.0, 0.0, 0.50, 0.0, 0.45]
```

Pose format:

```text
[qw, qx, qy, qz, x, y, z]
```

where quaternion `wxyz` is the gripper orientation and `xyz` is the target
gripper position in the robot/world frame.

Service endpoints:

```text
GET  /health
POST /ik
POST /plan
```

`POST /ik` input:

```json
{
  "target_pose_wxyz_xyz": [1.0, 0.0, 0.0, 0.0, 0.45, 0.0, 0.45],
  "prev_cfg": null
}
```

`POST /ik` output:

```json
{
  "joint_positions": [0.0, 0.0, "..."]
}
```

`POST /plan` input:

```json
{
  "start_pose_wxyz_xyz": [1.0, 0.0, 0.0, 0.0, 0.45, 0.0, 0.45],
  "end_pose_wxyz_xyz": [1.0, 0.0, 0.0, 0.0, 0.50, 0.0, 0.45],
  "timesteps": 5,
  "dt": 0.02
}
```

`POST /plan` output:

```json
{
  "waypoints": [[0.0, 0.0, "..."]],
  "dt": 0.02
}
```

Outputs are written to:

```text
tests/service_tests/outputs/pyroki_motion/
```

Output files:

```text
results.json
start_joint_positions.npy
end_joint_positions.npy
waypoints.npy
```

The output joint vectors have 8 values for the current Panda model. The first
7 values are the arm joints. The final value comes from the loaded gripper/finger
joint in the robot description.

This PyRoKi test does need a robot model, because IK cannot be solved without
robot kinematics. The service loads that model automatically from
`robot_descriptions`.

This simple test does not need a full planning scene. It does not consume table,
object, camera, or obstacle geometry. It is only:

```text
target 6D gripper pose -> robot joint positions
start/end gripper poses -> interpolated IK waypoints
```

For collision-aware planning with a real scene, use a richer planner input with
obstacles, or port/use CAP-X's cuRobo service. PyRoKi has collision utilities,
but this simple service test is intentionally only a kinematics smoke test.

## Local Qwen Foundation Model

The Qwen service is a local foundation-model service backed by Ollama:

```text
default model: qwen3.6:latest
Ollama URL: http://127.0.0.1:11434
service URL: http://127.0.0.1:8112
module: agentic_robots.services.foundation_model.qwen
```

This local `qwen3.6:latest` setup is a text LLM service inside the
`foundation_model` module. It is not a camera or robot-motion service, and it
does not need a robot model, depth image, segmentation mask, or planning scene.
It takes text messages and returns generated text.

Start the Qwen service in one terminal:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  -m agentic_robots.services.foundation_model.qwen.server --port 8112
```

Run the simple chat test from another terminal:

```bash
cd /home/user/Documents/Code/agentic_robots
/home/user/miniforge3/envs/agentic_robotics/bin/python \
  tests/service_tests/qwen_chat.py
```

Service endpoints:

```text
GET  /health
POST /chat
POST /chat/completions
```

`POST /chat` input:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a concise robot assistant."
    },
    {
      "role": "user",
      "content": "What should the robot do next?"
    }
  ],
  "temperature": 0.2,
  "max_tokens": 512
}
```

`POST /chat` output:

```json
{
  "model": "qwen3.6:latest",
  "content": "The robot should ...",
  "done": true
}
```

`POST /chat/completions` accepts the same message format and returns an
OpenAI-style response:

```json
{
  "id": "qwen-...",
  "object": "chat.completion",
  "created": 0,
  "model": "qwen3.6:latest",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

Outputs are written to:

```text
tests/service_tests/outputs/qwen_chat/
```

Output files:

```text
results.json
```

This wrapper can be used by higher-level robot-agent code for task reasoning,
planning text, or command interpretation. For image understanding, add a true
VLM/multimodal model under `foundation_model` and extend the request model with
image inputs. The local `qwen3.6:latest` setup is treated as text-only here.

## Original SAM3 Video Tracking Demo

The clean SAM3 examples track video by initializing a video session, adding an
object prompt on a frame, then propagating that prompt through the video. The
lower-level tracker API uses:

```python
from sam3.model_builder import build_sam3_video_model

sam3_model = build_sam3_video_model()
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone
inference_state = predictor.init_state(video_path=video_path)
predictor.add_new_points_or_box(...)
predictor.propagate_in_video(...)
```

Run the local demo on SAM3's bundled `bedroom.mp4`:

```bash
python tests/service_tests/sam3_video_tracking_demo.py \
  --max-frames 20 \
  --no-temporal-disambiguation
```

Outputs are written to:

```text
tests/service_tests/outputs/sam3_video_tracking/
```

## Verification

Quick checks:

```bash
cd /home/binzhao/Documents/Code/agentic_robots
conda activate agentic_robot
python -m compileall -q agentic_robots tests
python -m agentic_robots.services.sam3.server --help
python tests/service_tests/sam3_segment.py --help
python tests/service_tests/sam3_live_segment.py --help
```
