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
