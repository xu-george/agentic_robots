# Commands And Environment Notes

## Project Path

```bash
cd /home/user/Documents/Code/agentic_robots
```

## Real Lift Agent

```bash
DISPLAY=:0 XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.KKN2Q3 \
/home/user/miniforge3/envs/agentic_robotics/bin/python tests/agent_test/real_lift_agent.py \
  --object-prompt "banana" \
  --execute-real \
  --confirm-clear-workspace \
  --use-llm \
  --max-attempts 5 \
  --move-home-first
```

## Service Ports

- SAM3 service: `8114`
- GraspNet service: `8115`
- Qwen service: `8112`

## Notes

- Use `--no-grasp-window` to disable the live grasp visualization window.
- Use `--grasp-window-top-k` to control how many grasp hypotheses are drawn.

