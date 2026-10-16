"""API contract tests without loading model weights or requiring a GPU."""
from fastapi.testclient import TestClient
from agentic_robots.services.sam3.tracking_server import create_app


class FakeEngine:
    max_frames = 3
    state = None
    session_id = None

    def start(self, req):
        self.state = {}
        self.session_id = 'test'
        return {'session_id': 'test', 'frame_index': 0}

    def check_session(self, session_id):
        if self.state is None or session_id != self.session_id:
            raise ValueError('Unknown session')

    def frame(self, req):
        self.check_session(req.session_id)
        if getattr(req, 'image_base64', None) == 'fail':
            raise RuntimeError('inference failed')
        return {'session_id': self.session_id, 'frame_index': 1}


def test_session_lifecycle():
    client = TestClient(create_app(FakeEngine()))
    assert client.get('/health').json()['backend'] == 'sam3_native'
    assert client.post('/track/start', json={'image_base64': '', 'point_xy': [1, 2]}).status_code == 200
    assert client.post('/track/frame', json={'image_base64': '', 'session_id': 'wrong'}).status_code == 400
    assert client.post('/track/frame', json={'image_base64': '', 'session_id': 'test'}).status_code == 200
    assert client.post('/track/reset', json={'session_id': 'test'}).status_code == 200
    assert client.post('/track/frame', json={'image_base64': '', 'session_id': 'test'}).status_code == 400


def test_failure_invalidates_state():
    engine = FakeEngine()
    client = TestClient(create_app(engine))
    client.post('/track/start', json={'image_base64': '', 'point_xy': [1, 2]})
    assert client.post('/track/frame', json={'image_base64': 'fail', 'session_id': 'test'}).status_code == 500
    assert engine.state is None and engine.session_id is None


def test_raw_rgb_exact_pixels_and_validation():
    import numpy as np
    class PixelEngine(FakeEngine):
        def start(self, req):
            self.pixels = np.asarray(req.image)
            return super().start(req)

    engine = PixelEngine()
    client = TestClient(create_app(engine))
    pixels = np.array([[[255, 0, 17], [0, 99, 255]]], dtype=np.uint8)
    headers = {'Content-Type': 'application/octet-stream'}
    params = {'width': 2, 'height': 1, 'text_prompt': 'black handle'}
    response = client.post('/track/start_raw', params=params, content=pixels.tobytes(), headers=headers)
    assert response.status_code == 200
    np.testing.assert_array_equal(engine.pixels, pixels)
    assert client.post('/track/frame_raw', params={'width': 2, 'height': 1, 'session_id': 'test'}, content=pixels.tobytes(), headers=headers).status_code == 200
    assert client.post('/track/start_raw', params=params, content=b'bad', headers=headers).status_code == 400
    assert client.post('/track/start_raw', params={**params, 'width': 0}, content=pixels.tobytes(), headers=headers).status_code == 422
    assert client.post('/track/start_raw', params={**params, 'point_x': 1}, content=pixels.tobytes(), headers=headers).status_code == 400
    assert client.post('/track/start_raw', params={'width': 2, 'height': 1}, content=pixels.tobytes(), headers=headers).status_code == 400


def test_raw_client_validates_dtype():
    import numpy as np
    import pytest
    from agentic_robots.services.sam3.tracking_client import track_rgb
    with pytest.raises(ValueError, match='uint8'):
        track_rgb('http://unused', np.zeros((2, 2, 3), dtype=np.float32), text_prompt='handle')


def test_point_schema():
    client = TestClient(create_app(FakeEngine()))
    assert client.post('/track/start', json={'image_base64': '', 'point_xy': [1]}).status_code == 422
