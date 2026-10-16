"""Benchmark native SAM3 HTTP tracking on a fixed RGB N,H,W,3 uint8 NPY clip."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agentic_robots.services.base import encode_image, decode_mask
from agentic_robots.services.sam3.tracking_client import track_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('frames', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8116')
    p.add_argument('--prompt', default='black handle')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--transport', choices=['png', 'raw'], default='raw')
    a = p.parse_args()
    frames = np.load(a.frames)
    assert len(frames) > a.warmup + 1
    session = None
    rows = []
    try:
        for i, frame in enumerate(frames):
            start = time.perf_counter()
            if a.transport == 'raw':
                result = track_rgb(a.url, frame, session_id=session, text_prompt=a.prompt)
            else:
                payload = {'image_base64': encode_image(frame)}
                if i == 0:
                    payload['text_prompt'] = a.prompt
                else:
                    payload['session_id'] = session
                response = requests.post(a.url + ('/track/start' if i == 0 else '/track/frame'), json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
            session = result['session_id']
            mask = decode_mask(result['mask_base64'], tuple(result['shape']))
            assert result['frame_index'] == i and mask.shape == frame.shape[:2]
            rows.append({'frame': i, 'end_to_end_s': time.perf_counter()-start, 'server_s': result['latency_s']})
        measured = rows[a.warmup+1:]
        result = {'transport': a.transport, 'frames': str(a.frames), 'total_frames': len(frames), 'measured_frames': len(measured),
                  'end_to_end_fps': len(measured)/sum(r['end_to_end_s'] for r in measured),
                  'server_fps': len(measured)/sum(r['server_s'] for r in measured), 'samples': rows}
        a.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k:v for k,v in result.items() if k != 'samples'}, indent=2))
    finally:
        if session:
            requests.post(a.url+'/track/reset', json={'session_id': session}, timeout=10)


if __name__ == '__main__':
    main()
