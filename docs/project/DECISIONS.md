# Decisions

## 2026-06-29 - Treat agentic_robots As The Main Control Repository

Reason: keep this repository focused on real robot execution, high-level APIs, perception wrappers, service orchestration, and LLM/code-agent planning.

Scope: simulation-specific training and model fine-tuning can connect later, but should not blur the stable control layer in this repo.

## 2026-06-29 - Planner Should Use Stable APIs And Skills

Reason: the planner should reason over high-level robot actions and reusable skills instead of raw model internals or low-level control details.

Scope: expose clean APIs for actions, perception, skill execution, and verification.

## 2026-06-29 - Keep Environment Logic Outside ActionApi

Reason: generated action code should transfer between simulation and real robot backends.

Scope: `ActionApi` should stay robot-only; frame recording, simulation stepping, object state, and video capture belong in the environment/backend.

## 2026-06-26 - Keep PerceptionApi Unchanged

Reason: keep the real robot perception API stable.

Scope: visualization changes should stay inside `tests/` unless George explicitly asks otherwise.

## Future Decisions

Add new decisions here with date, reason, and scope.
