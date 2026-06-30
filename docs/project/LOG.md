# Project Log

## 2026-06-26

- Added test-local live grasp visualization support in `tests/agent_test/real_lift_agent.py`.
- Kept `PerceptionApi` unchanged.
- Kept the code change scoped to `tests/`.
- Removed the standalone `tests/agent_test/real_lift_gui.py`.
- Freed GPU afterward by stopping SAM3, GraspNet, and Qwen services and unloading the Ollama Qwen model.

## 2026-06-29

- Created project documentation folder at `docs/project/`.
- Added the initial concise project overview to `README.md`.
- Added the finalized system module structure to `README.md`.
- Updated project docs to focus on `agentic_robots` as the main control and orchestration repository.
- Filled `STATUS.md`, `NEXT.md`, `COMMANDS.md`, and `DECISIONS.md` with repo-specific current state and next steps.
- Created the Notion project docs page and seven module child pages.
- Generated and attached the Agentic Robots control-flow chart image.
- Added project motivation and a more detailed description to the local README and Notion project page.
- Added `STRUCTURE.md` with current services, APIs, environments, agent code, interfaces, and result artifacts.
- Created the Notion `Current Project Structure` child page.
- Created the detailed colorful structure picture with service/API/env/agent interfaces, data flow, data types, and module result images.

## 2026-06-30

- Rewrote the README Motivation and Detailed Description sections to make the project logic more fluent and concrete, connecting the problem, agent/API/service/env architecture, data flow, module results, and long-term skill-learning goal.
- Condensed the Motivation and changed the Detailed Description into one concise paragraph while keeping project-specific modules, data flow, results, and goals.
