"""Lossless packed RGB8 transport for the localhost native tracking service."""
import numpy as np
import requests


def track_rgb(url, rgb, *, session_id=None, text_prompt=None, point_xy=None,
              min_score=0.3, timeout=120):
    """Send raw pixels; no PNG/JPEG or base64 image conversion, and no retries."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected HxWx3 uint8 RGB image')
    params = {'width': rgb.shape[1], 'height': rgb.shape[0]}
    if session_id is None:
        if (text_prompt is None) == (point_xy is None):
            raise ValueError('Provide exactly one point or text prompt')
        endpoint = '/track/start_raw'
        params['min_score'] = min_score
        if text_prompt is not None:
            params['text_prompt'] = text_prompt
        else:
            params['point_x'], params['point_y'] = point_xy
    else:
        endpoint = '/track/frame_raw'
        params['session_id'] = session_id
    response = requests.post(
        url.rstrip('/') + endpoint, params=params,
        data=np.ascontiguousarray(rgb).tobytes(),
        headers={'Content-Type': 'application/octet-stream'}, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f'Tracking request failed ({response.status_code}): {response.text}')
    return response.json()
