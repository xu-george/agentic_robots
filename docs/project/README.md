# Agentic Robots Project

## Project Overview

This project builds a reusable robotic control framework for agentic robot learning and execution.

`agentic_robots` is the main control and orchestration repository. It focuses on real robot execution, stable high-level APIs, perception service wrappers, local LLM planning, and task-level demos.

## Motivation

Robot manipulation demos often work only as one-off scripts: they are tied to one object, one scene, or one task, and are hard to reuse when the environment changes. At the same time, vision-language models and code agents can reason about goals and task steps, but they still need stable perception, grasp planning, motion, robot safety, and execution feedback before they can control real hardware reliably.

`agentic_robots` is motivated by the need to connect high-level AI planning with reliable robot execution. It turns perception models, grasp planners, robot controllers, simulation tasks, and real hardware environments into stable services and APIs, so an agent can describe, plan, execute, inspect, and improve manipulation tasks without depending on fragile low-level scripts.

The long-term goal is a continuous-learning robot system that can accumulate reusable skills, record results and failures, and reuse successful behaviors across simulated and real tasks such as lifting, stacking, pick-and-place, luggage handling, and polishing.

## Detailed Description

`agentic_robots` is the control and orchestration layer for an agentic robot manipulation system: a user goal flows through `CodeAgent` and `AgentRuntime`, into high-level APIs such as `PerceptionApi` and `ActionApi`, then into model and robot services including Qwen for local planning, SAM3 for object segmentation, GraspNet for grasp hypotheses, and PyRoKi for motion waypoints, before execution in PyBullet Franka tasks or on the real Franka robot with RealSense input and calibration. The project keeps the planner separated from low-level robot code, exposes stable interfaces for observation, segmentation, grasp planning, motion, and task execution, and saves inspectable results such as SAM banana segmentation overlays, GraspNet grasp renders, RealSense grasp overlays, Qwen outputs, PyRoKi waypoint results, task logs, and final success or failure feedback. This structure supports both Real2Sim / Sim2Real research and reusable high-level robot APIs, making it easier to test tasks, compare module outputs, transfer behavior between simulation and real hardware, and later use saved execution experience to improve skills or fine-tune local models.

## Goals

- Build a stable, reusable robot control framework.
- Support real-to-sim testing and sim-to-real execution.
- Store reusable robot skills for future tasks.
- Enable continuous learning from task performance and execution history.

## Repository Role

This repository owns:

- High-level robot action APIs.
- Real robot execution through Franka and RealSense integration.
- Perception wrappers for services such as SAM3 and GraspNet.
- Local LLM planning and code-agent runtime.
- Service orchestration for Qwen, SAM3, and GraspNet.
- Real and simulated task demos such as lift and stacking.

## System Modules

1. **Real2Sim / Scene Reconstruction**
   Converts real-world robot observations into simulation-ready scene states, including objects, robot state, task targets, and environment context.

2. **Planning / Task Decomposition**
   Breaks user goals into high-level task sequences and subtasks that call stable robot APIs and reusable skills.

3. **Skill System**
   Wraps low-level perception and control tools, such as SAM, GraspNet, and robot motion APIs, into reusable high-level skills.

4. **Skill Learning and Storage**
   Stores reusable skills, execution traces, success conditions, failure cases, and examples for future reuse and improvement.

5. **Contact-Rich Task Fine-Tuning**
   Uses task performance from difficult manipulation tasks, such as luggage handling and polishing, to improve skills and fine-tune local small models.

6. **Sim2Real / Real Robot Transfer**
   Transfers validated plans and skills from simulation to the real robot through safe, stable robot control APIs.

7. **Evaluation and Feedback Loop**
   Records task outcomes and feeds success or failure information back into planning, skill selection, and learning.

## Repository

- Path: `/home/user/Documents/Code/agentic_robots`

## Notion

- Project docs page: https://app.notion.com/p/Agentic-Robots-Project-Docs-38e2486e606f8148bab5f0ee0ec834c1
- Current project structure page: https://app.notion.com/p/Current-Project-Structure-38e2486e606f8100befcdf21de2ced58

## Visuals

- Control flow chart: [assets/agentic_robots_flow.png](assets/agentic_robots_flow.png)
- Detailed structure picture: [assets/agentic_robots_detailed_structure.png](assets/agentic_robots_detailed_structure.png)

## Start Here

- Current status: [STATUS.md](STATUS.md)
- Current project structure: [STRUCTURE.md](STRUCTURE.md)
- Next plan: [NEXT.md](NEXT.md)
- Key decisions: [DECISIONS.md](DECISIONS.md)
- Commands and environment notes: [COMMANDS.md](COMMANDS.md)
