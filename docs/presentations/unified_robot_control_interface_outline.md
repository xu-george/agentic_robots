# Unified Robot Control Interface

**Subtitle:** A common GUI and integration framework for agentic planning, Isaac ROS execution, and Picker2 learned skills

## 1. Title
A unified robot control interface grounded in the current `agentic_robots`, Isaac ROS, Picker2 RL, and CaP-X repositories.

## 2. North star — one operator experience; multiple execution engines
- One control plane for tasks, sessions, approvals, stop/reset, and audit.
- Pluggable capabilities for agentic APIs, ROS 2 actions/services, simulation, and learned policies.
- One evidence model for state, images, poses, logs, metrics, and failure reasons.
- **Design rule:** separate operator intent, orchestration, and safety from backend execution.

## 3. Current landscape
- **agentic_robots:** code agent, allowlisted runtime, action/perception APIs, model services, PyBullet, and real Franka/RealSense.
- **Isaac ROS:** RGB-D, calibration, drivers, nvblox, detection/pose, cuMotion, and pick-and-place actions.
- **picker2_rl_grasping:** Genesis simulation, M1013 + 2FG14 + D455, a 7D action interface, privileged state, and luggage curriculum.
- Gap: no common task/session contract or operator GUI.

## 4. agentic_robots module view
Current flow: task intent → CodeAgent → allowlisted runtime → ActionApi/PerceptionApi → services → simulation or real robot.

The existing web module is a read-only RealSense perception viewer. It shows RGB, depth, masks, grasp hypotheses, and scores, but is not a session controller.

## 5. Preserve small planner-facing interfaces
- `ActionApi`: pose access, Cartesian move, open, close.
- `PerceptionApi`: observation, segmentation, pose, point cloud, and grasp planning.
- `AgentRuntime`: generated code can call registered functions only.
- Extend the same pattern with ROS-action and learned-policy adapters.

## 6. Isaac ROS as the production ROS graph
The manipulation workflow combines camera/simulation input, robot drivers and TF, nvblox, object detection/pose, cuMotion, and pick-and-place orchestration.

The unified gateway should be a ROS 2 client. It should not replace ROS lifecycle, TF, action cancellation, drivers, or controllers.

## 7. Lessons from the CaP-X GUI
Keep:
- configuration and model selection;
- start/stop/reset controls;
- session reconnect and event replay;
- streaming reasoning, code, execution results, and human feedback;
- embedded 3D visualization.

Add for physical robots:
- hardware mode and readiness;
- approvals and alarms;
- camera/TF overlays;
- capability health;
- immutable audit timeline.

## 8. Proposed unified operator workspace
A single workspace contains:
- task and phase timeline;
- agent trace and operator feedback;
- 3D/RGB-D visualization;
- evidence and telemetry;
- robot, ROS graph, perception, planner, policy, and safety health;
- clearly separated simulation/real and armed/disarmed modes.

## 9. Reference architecture
1. **Web client:** task builder, timeline, visualization, safety controls, audit viewer.
2. **Control gateway:** sessions, capability registry, events, approval policy, artifacts.
3. **Adapters:** agentic Python API, ROS 2 actions/services, RL policy, simulator, camera/viewer.
4. **Execution:** Isaac ROS, Franka bridge, Genesis, PyBullet, and model services.

## 10. Unified capability contract
Each capability declares:
- ID and execution mode;
- typed inputs/outputs;
- frame IDs and units;
- preconditions and safety class;
- timeout and mandatory cancellation behavior;
- progress events and evidence outputs;
- model/configuration version.

## 11. Observable and interruptible task lifecycle
`DRAFT → VALIDATE → PLAN → APPROVE → EXECUTE → VERIFY → DONE`

Failure/cancellation can transition execution to a safe stop. Success means task criteria were verified—not merely that code or a process exited.

## 12. Picker2 luggage task
Current implementation includes:
- fixed Picker2 base, Doosan M1013, OnRobot 2FG14, and D455 geometry;
- physical hook-fingertip grasping without an attachment constraint;
- 7D Cartesian/gripper actions and 46D privileged observations;
- parallel Genesis reset and stepping.

Task sequence:
`REACH → GRASP → LIFT → TRANSPORT → ROTATE FLAT → PLACE → RELEASE → RETREAT → READY`

Training completion and real-robot deployment remain work in progress.

## 13. Picker2 integration
Expose Picker2 as a versioned skill adapter:
1. perceive the handle and target;
2. select policy and object configuration;
3. validate workspace, TF, readiness, and approval;
4. run a deterministic inference process that publishes bounded 7D commands;
5. monitor phase, contacts, pose error, timeout, and anomalies;
6. verify flat, in-bounds, stable placement, release, and return-ready.

Do not import the training environment into the GUI server.

## 14. Safety model
Safety layers remain independent:
- operator controls;
- gateway authorization, approvals, and audit;
- capability checks for schemas, TF, limits, and timeouts;
- ROS controllers, hardware interlocks, and physical user stop.

The browser emergency control is a request, not the only emergency channel.

## 15. Roadmap
1. **Observe:** unified read-only sessions, camera/3D, health, and logs.
2. **Command:** capability registry, task lifecycle, ROS actions, stop/reset.
3. **Agent:** CodeAgent trace, approvals, retries, evidence, replay.
4. **Picker2:** policy runner, simulation evaluation, shadow mode, supervised hardware.

## 16. Recommendation
Build a thin, event-driven control plane over typed capabilities.

First deliverable: a read-only unified session viewer plus the capability/event schema. This creates the common foundation before adding command authority.

## 17. Repository evidence
The deck cites implementation files from:
- `/home/user/Documents/Code/agentic_robots`
- `/home/user/workspaces/isaac_ros-dev/src/isaac_ros_manipulation`
- `/home/user/Documents/Code/picker2_rl_grasping`
- `/home/user/Documents/Code/cap-x`
