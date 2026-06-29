# Agentic Robots Project

## Project Overview

This project builds a reusable robotic control framework for agentic robot learning and execution.

The framework has two main directions:

- **Real2Sim / Sim2Real:** connect real robot tasks with parallel simulation environments such as Genesis-style simulation, inspired by projects like RoboGen.
- **Stable high-level control APIs:** provide reusable robot APIs, similar in spirit to CAP-X, so an AI planner can reason over tasks without depending on fragile low-level control code.

The long-term goal is a continuous-learning robot system that can store, reuse, and improve skills such as pick-and-place, luggage handling, polishing, and other contact-rich manipulation tasks.

The system is designed to work with local small language models. Task performance, execution feedback, and saved skills can later be used to improve planning behavior and fine-tune the local model.

## Goals

- Build a stable, reusable robot control framework.
- Support real-to-sim testing and sim-to-real execution.
- Store reusable robot skills for future tasks.
- Enable continuous learning from task performance and execution history.

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

## Start Here

- Current status: [STATUS.md](STATUS.md)
- Next plan: [NEXT.md](NEXT.md)
- Key decisions: [DECISIONS.md](DECISIONS.md)
- Commands and environment notes: [COMMANDS.md](COMMANDS.md)
