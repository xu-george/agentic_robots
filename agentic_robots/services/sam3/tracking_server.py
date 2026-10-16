"""Single-session, forward-only native SAM3 tracking for incoming RGB frames."""
import argparse
import threading
import time
import uuid
from types import SimpleNamespace

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from PIL import Image
from pydantic import BaseModel

from agentic_robots.services.base import decode_image, encode_mask


class FrameRequest(BaseModel):
    image_base64: str
    session_id: str


class StartRequest(BaseModel):
    image_base64: str
    point_xy: tuple[float, float] | None = None
    text_prompt: str | None = None
    min_score: float = 0.3


class SessionRequest(BaseModel):
    session_id: str


class NativeTracker:
    def __init__(self, device="cuda:0", max_frames=300, offload_state_to_cpu=False):
        import torch
        from sam3.model_builder import build_sam3_video_model

        if max_frames < 2:
            raise ValueError("max_frames must be >= 2")
        self.torch = torch
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.model = build_sam3_video_model(device=device)
        self.predictor = self.model.tracker
        self.predictor.backbone = self.model.detector.backbone
        self.max_frames = max_frames
        self.offload_state_to_cpu = offload_state_to_cpu
        self.state = None
        self.session_id = None
        self.index = 0
        self.text_processor = None

    def _image(self, image):
        # Match vendored SAM3 video loader: RGB, square resize, mean/std 0.5.
        size = self.predictor.image_size
        tensor = self.torch.from_numpy(np.asarray(image.resize((size, size))).copy())
        if not self.offload_state_to_cpu:
            tensor = tensor.to(self.device)
        return (tensor.permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5

    def _result(self, masks, started):
        masks = (masks > 0).detach().cpu().numpy()
        mask = masks[0].squeeze()
        out = {"session_id": self.session_id, "frame_index": self.index,
               "object_id": 1, "mask_base64": encode_mask(mask),
               "shape": list(mask.shape), "latency_s": time.perf_counter() - started}
        if len(masks) > 1:          # objects 2..n of a multi-mask session, in start order
            out["extra_masks_base64"] = [encode_mask(m.squeeze()) for m in masks[1:]]
        return out

    def start(self, request):
        self.torch.cuda.set_device(self.device)
        image = request.image if hasattr(request, 'image') else decode_image(request.image_base64)
        init_mask = getattr(request, 'mask', None)
        if init_mask is not None and init_mask.ndim == 2:
            init_mask = init_mask[None]                 # (n_objects, H, W)
        if (request.point_xy is not None) + (request.text_prompt is not None) + (init_mask is not None) != 1:
            raise ValueError("Provide exactly one of point_xy, text_prompt or mask")
        if init_mask is not None and not all(m.any() for m in init_mask):
            raise ValueError("Initial mask is empty")
        if not 0 <= request.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if request.text_prompt is not None and not request.text_prompt.strip():
            raise ValueError("Text prompt cannot be empty")
        if request.point_xy is not None:
            x, y = request.point_xy
            if not (0 <= x < image.width and 0 <= y < image.height):
                raise ValueError("Point must be inside the image")
        started = time.perf_counter()
        self.state = None
        self.session_id = None
        self.index = 0
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            state = self.predictor.init_state(video_height=image.height, video_width=image.width,
                                              num_frames=1, offload_state_to_cpu=self.offload_state_to_cpu)
            state["images"] = {0: self._image(image)}
            score = None
            if init_mask is not None:
                # start from caller-selected masks (e.g. one of several same-class objects); object ids 1..n
                for k, m in enumerate(init_mask):
                    _, _, _, masks = self.predictor.add_new_mask(
                        inference_state=state, frame_idx=0, obj_id=k + 1,
                        mask=self.torch.from_numpy(np.ascontiguousarray(m)).to(self.device))
            elif request.text_prompt is not None:
                if self.text_processor is None:
                    from sam3.model_builder import build_sam3_image_model
                    from sam3.model.sam3_image_processor import Sam3Processor
                    model = build_sam3_image_model(device='cpu').to(self.device)
                    self.text_processor = Sam3Processor(model, device=str(self.device), confidence_threshold=0.0)
                output = self.text_processor.set_text_prompt(
                    prompt=request.text_prompt,
                    state=self.text_processor.set_image(image))
                scores = output['scores'].flatten()
                if scores.numel() == 0:
                    raise ValueError('No matching object; try a different prompt')
                best = int(scores.argmax().item())
                score = float(scores[best].item())
                if score < request.min_score:
                    raise ValueError(f'Best match score {score:.3f} below threshold {request.min_score}')
                mask = output['masks'][best].squeeze() > 0
                if not bool(mask.any()):
                    raise ValueError('Text prompt produced an empty mask')
                _, _, _, masks = self.predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1, mask=mask)
            else:
                _, _, _, masks = self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=self.torch.tensor([[x / image.width, y / image.height]], dtype=self.torch.float32),
                    labels=self.torch.tensor([1], dtype=self.torch.int32))
            self.predictor.propagate_in_video_preflight(state)
            self.state = state
            self.session_id = uuid.uuid4().hex
            result = self._result(masks, started)
            result.update(text_prompt=request.text_prompt, score=score)
            return result

    def check_session(self, session_id):
        if self.state is None or session_id != self.session_id:
            raise ValueError("Unknown or expired session; start tracking again")

    def frame(self, request):
        self.torch.cuda.set_device(self.device)
        self.check_session(request.session_id)
        if self.index + 1 >= self.max_frames:
            raise ValueError("Session frame limit reached; reset and select the object again")
        image = request.image if hasattr(request, 'image') else decode_image(request.image_base64)
        state = self.state
        if image.size != (state["video_width"], state["video_height"]):
            raise ValueError("Frame dimensions changed; start a new session")
        started = time.perf_counter()
        self.index += 1
        state["num_frames"] = self.index + 1
        # Only the current image is needed; SAM3 retains its own temporal memory.
        state["images"] = {self.index: self._image(image)}
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            outputs = list(self.predictor.propagate_in_video(
                state, start_frame_idx=self.index, max_frame_num_to_track=0,
                reverse=False, tqdm_disable=True, propagate_preflight=False))
            if len(outputs) != 1 or outputs[0][0] != self.index:
                raise RuntimeError("Unexpected native tracking frame index")
            return self._result(outputs[0][3], started)


def create_app(engine):
    app = FastAPI(title="Native SAM3 live tracking")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"ok": True, "backend": "sam3_native", "max_frames": engine.max_frames}

    def run(fn, req):
        with lock:
            try:
                return fn(req)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except Exception as exc:
                # Failed inference may have partially mutated temporal memory.
                engine.state = None
                engine.session_id = None
                raise HTTPException(500, str(exc)) from exc

    @app.post("/track/start")
    def start(req: StartRequest):
        return run(engine.start, req)

    @app.post("/track/frame")
    def frame(req: FrameRequest):
        return run(engine.frame, req)

    def raw_image(body, width, height):
        if len(body) != width * height * 3:
            raise HTTPException(400, 'Expected exactly width * height * 3 bytes of packed RGB8')
        return Image.frombytes('RGB', (width, height), body)

    @app.post('/track/start_raw')
    def start_raw(
        body: bytes = Body(..., media_type='application/octet-stream'),
        width: int = Query(..., ge=1, le=4096),
        height: int = Query(..., ge=1, le=4096),
        text_prompt: str | None = None,
        point_x: float | None = None,
        point_y: float | None = None,
        min_score: float = Query(0.3, ge=0, le=1),
        with_mask: bool = False,
        n_masks: int = Query(1, ge=1, le=8),
    ):
        """with_mask: the body is the RGB8 image followed by n_masks width*height uint8 masks (non-zero = object);
        object k+1 is tracked from mask k and returned as mask_base64 (k=0) or extra_masks_base64[k-1]."""
        if (point_x is None) != (point_y is None):
            raise HTTPException(400, 'Both point_x and point_y are required')
        point = None if point_x is None else (point_x, point_y)
        mask = None
        if with_mask:
            n = width * height * 3
            if len(body) != n + n_masks * width * height:
                raise HTTPException(400, 'Expected packed RGB8 image, then n_masks width * height uint8 masks')
            mask = np.frombuffer(body[n:], np.uint8).reshape(n_masks, height, width) > 0
            body = body[:n]
        if (point is not None) + (text_prompt is not None) + (mask is not None) != 1:
            raise HTTPException(400, 'Provide exactly one point, text prompt or mask')
        return run(engine.start, SimpleNamespace(
            image=raw_image(body, width, height), point_xy=point,
            text_prompt=text_prompt, min_score=min_score, mask=mask))

    @app.post('/track/frame_raw')
    def frame_raw(
        body: bytes = Body(..., media_type='application/octet-stream'),
        width: int = Query(..., ge=1, le=4096),
        height: int = Query(..., ge=1, le=4096),
        session_id: str = Query(...),
    ):
        return run(engine.frame, SimpleNamespace(
            image=raw_image(body, width, height), session_id=session_id))

    @app.post("/track/reset")
    def reset(req: SessionRequest):
        def clear(request):
            engine.check_session(request.session_id)
            engine.state = None
            engine.session_id = None
            return {"ok": True}
        return run(clear, req)

    return app


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8116)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument('--offload-state-to-cpu', action='store_true', help='Use CPU tracking state instead of GPU (slower, lower VRAM)')
    args = parser.parse_args()
    uvicorn.run(create_app(NativeTracker(args.device, args.max_frames, args.offload_state_to_cpu)), host=args.host, port=args.port)
