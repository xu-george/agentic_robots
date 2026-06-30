# Status

## Current State

`agentic_robots` currently provides the main robot-agent control stack. It has a generic code-agent runtime, stable robot action APIs, perception service clients, local model service orchestration, simulation demos, and real robot test scripts.

## Finished Tasks

- Created a concise project documentation folder under `docs/project/`.
- Defined the project overview, goals, and system module structure.
- Built a generic `CodeAgent` runtime for LLM-generated Python code execution through allowlisted APIs.
- Kept `ActionApi` focused on robot-only actions such as moving, opening, closing, and pose access.
- Added service clients/servers for local Qwen, SAM3, and GraspNet workflows.
- Added configuration for local service launch, real Franka setup, RealSense calibration, and lift objects.
- Added real lift test flow with live grasp visualization inside `tests/agent_test/real_lift_agent.py`.
- Kept test-only visualization changes out of `PerceptionApi`.

## Known Working Components

- `CodeAgent` and runtime loop for API-limited task execution.
- `ActionApi` robot-control interface.
- `PerceptionApi` using SAM3 segmentation and GraspNet grasp planning.
- Local service launcher for Qwen, SAM3, and GraspNet.
- PyBullet / simulated task demos.
- Real Franka and RealSense integration path.
- Real lift agent test script with optional grasp overlay window.

## Current Limitations

- Reusable skill storage is not yet formalized.
- Planning still needs a clearer high-level task decomposition layer.
- Execution logs and success/failure feedback are not yet organized as a learning dataset.
- Real2Sim and Sim2Real interfaces are not yet fully connected inside this repo.
- Contact-rich task learning, such as luggage handling or polishing, is still future work.
