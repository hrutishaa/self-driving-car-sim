"""Quick check without the simulator: build the model, train 2 epochs on fake data,
then start drive.py and send it telemetry over Socket.IO.

    python test_smoke.py
"""
import asyncio
import base64
import os
import subprocess
import sys
import tempfile
import time
from io import BytesIO

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    import keras
    from model import build_model
    from drive import Driver
    from utils import preprocess

    m = build_model()
    assert m.count_params() == 252219, m.count_params()
    print('model params', m.count_params())

    with tempfile.TemporaryDirectory() as tmp:
        data, out = os.path.join(tmp, 'data'), os.path.join(tmp, 'model.keras')
        subprocess.run([sys.executable, os.path.join(HERE, 'model.py'), '--synthetic', '-d', data,
                        '-n', '2', '-b', '32', '-o', out], check=True)
        model = keras.models.load_model(out)

        frame = (np.random.default_rng(0).random((160, 320, 3)) * 255).astype(np.uint8)
        assert preprocess(frame).shape == (66, 200, 3)
        buf = BytesIO(); Image.fromarray(frame).save(buf, format='JPEG')
        msg = {'steering_angle': '0', 'throttle': '0', 'speed': '5.0', 'image': base64.b64encode(buf.getvalue()).decode()}
        steer, thr = Driver(model)(msg)
        print(f'driver handler ok: steering {steer:+.3f} throttle {thr:+.3f}')

        # talk to drive.py the way the sim does
        port = '4599'
        proc = subprocess.Popen([sys.executable, os.path.join(HERE, 'drive.py'), out, '--port', port],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            import socketio
            got = {}
            client = socketio.Client()
            client.on('steer', lambda d: got.setdefault('steer', []).append(d))
            for _ in range(90):
                try:
                    client.connect(f'http://localhost:{port}'); break
                except Exception:
                    time.sleep(1)  # server still loading TensorFlow
            for _ in range(3):
                client.emit('telemetry', msg); time.sleep(0.5)
            client.disconnect()
            assert len(got.get('steer', [])) >= 2, got
            print('socket.io round trip ok:', got['steer'][-1])
        finally:
            proc.kill(); proc.wait(10)
    print('ok')


if __name__ == '__main__':
    main()
