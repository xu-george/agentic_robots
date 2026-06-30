# Next Plan

## Immediate Next Steps

- Stabilize real robot task execution for the current lift workflow.
- Cleanly define a reusable skill API layer on top of existing robot, perception, and motion APIs.
- Add a first skill record format for skills such as detect, grasp, pick, place, and verify.
- Add structured execution logging for plans, API calls, observations, success checks, and failures.
- Connect planner output to reusable skills instead of one-off task scripts.

## Later Work

- Build a real-to-sim scene handoff interface for simulation repos.
- Add evaluation feedback that can be reused for continuous learning.
- Prepare contact-rich task workflows such as luggage picking and polishing.
- Connect stored execution traces to future fine-tuning pipelines.
- Add safer real robot retry, rollback, and failure recovery behavior.

## Open Questions

- What should be the first formal skill schema?
- Which skills should be implemented first: pick-place only, or also contact-rich skills?
- What execution information is required for future fine-tuning?
- How much task planning should live in `agentic_robots` versus external training/simulation repos?

## Risks / Things To Watch

- Avoid letting planner code call low-level model internals directly.
- Keep real robot safety checks explicit before execution.
- Keep test-only visualization separate from stable perception APIs.
- Avoid mixing research experiments into the stable robot-control layer too early.
