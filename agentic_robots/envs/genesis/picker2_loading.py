"""Thin adapter for the Picker2 two-luggage frozen-skill experiment.

Use the genesis_simulation environment and expose picker2_rl_grasping/src on
PYTHONPATH. No Franka hardware, perception service, or PyRoKi server is invoked.
"""


class Picker2LoadingBackend:
    def __init__(self, world):
        from picker2_rl_grasping.loading.controller import LoadingController
        self.world = world
        from picker2_rl_grasping.loading.manual import ManualLoadingController
        self.controller = (ManualLoadingController if world.loading_cfg.get('manual_sequence') else LoadingController)(world)

    def reset(self):
        from picker2_rl_grasping.loading.controller import LoadingController
        result = self.world.reset()
        from picker2_rl_grasping.loading.manual import ManualLoadingController
        self.controller = (ManualLoadingController if self.world.loading_cfg.get('manual_sequence') else LoadingController)(self.world)
        return result

    def get_obs(self):
        return {'objects': self.world.get_ground_truth(), 'stage': self.controller.stage,
                'loaded': self.world.loaded.tolist(), 'failed': self.world.failed.tolist(),
                'failure_reason': self.world.failure_reason}

    def get_ground_truth(self):
        return self.world.get_ground_truth()

    def is_success(self):
        return self.world.is_success()

    def load_luggage(self, index: int):
        try:
            return self.controller.load_one(int(index))
        except Exception as error:
            for lane in range(self.world.num_envs):
                self.world.fail(lane, str(error))
            raise

    def api_functions(self):
        return {'load_luggage': self.load_luggage,
                'get_loading_scene': self.get_ground_truth,
                'loading_success': self.is_success}

    def code_agent_config(self):
        from agentic_robots.agent.code_agent import CodeAgentConfig
        return CodeAgentConfig(
            task_description='Load luggage0 into the front (+X) platform slot, then luggage1 into the rear slot. '
                             'Each load uses frozen RL pickup, planned transport, conditional experimental '
                             'canonical-frame RL tipping for top-up luggage (side handle -Y, tip +Y), and explicit release/withdrawal. '
                             'Side-up luggage skips tipping. Stop on failure; do not report success unless verified.',
            api_functions=self.api_functions(), success_fn=self.is_success,
            scene_context_fn=self.get_ground_truth, diagnostics_fn=self.get_obs,
            fallback_code='load_luggage(0)\nload_luggage(1)',
        )
