__all__ = ["GraspNetClient", "GraspPlan"]


def __getattr__(name: str):
    if name == "GraspNetClient":
        from .client import GraspNetClient

        return GraspNetClient
    if name == "GraspPlan":
        from .types import GraspPlan

        return GraspPlan
    raise AttributeError(name)
