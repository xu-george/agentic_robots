__all__ = ["PointPromptResult", "Sam3Client", "Sam3Mask"]


def __getattr__(name: str):
    if name == "Sam3Client":
        from .client import Sam3Client

        return Sam3Client
    if name in {"PointPromptResult", "Sam3Mask"}:
        from .types import PointPromptResult, Sam3Mask

        return {"PointPromptResult": PointPromptResult, "Sam3Mask": Sam3Mask}[name]
    raise AttributeError(name)
