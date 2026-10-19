"""Run a trained model against the Udacity sim (autonomous mode).

    python drive.py model.keras [--speed 20] [--record run1/]

The sim connects on port 4567 and sends telemetry with a base64 frame, we send back
steering. Throttle comes from a PI controller, not the network.
"""
from __future__ import annotations

import argparse
import base64
import os
import warnings
from datetime import datetime, timezone
warnings.filterwarnings('ignore', category=DeprecationWarning)
from io import BytesIO

import eventlet  # same as Udacity's drive.py, the old sim works with it
import eventlet.wsgi
import numpy as np
import socketio  # needs 4.x for the old sim, see requirements.txt
from PIL import Image

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import keras  # noqa: E402

from utils import preprocess  # noqa: E402


class PIController:
    def __init__(self, kp=0.1, ki=0.002):
        self.kp, self.ki, self.set_point, self.integral = kp, ki, 0.0, 0.0

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        error = self.set_point - measurement
        self.integral += error
        self.integral = float(np.clip(self.integral, -500, 500))
        return self.kp * error + self.ki * self.integral


class Driver:
    def __init__(self, model, max_speed=20.0, min_speed=10.0, record_dir=None):
        self.model, self.max_speed, self.min_speed = model, max_speed, min_speed
        self.controller = PIController()
        self.record_dir = record_dir

    def target_speed(self, steering):
        # slow down in bends
        return max(self.min_speed, self.max_speed - (self.max_speed - self.min_speed) * min(1.0, abs(steering) * 2.5))

    def __call__(self, data):
        speed = float(data['speed'])
        image = Image.open(BytesIO(base64.b64decode(data['image']))).convert('RGB')
        frame = np.asarray(image)
        x = preprocess(frame)[None].astype(np.float32)  # same crop/resize as training
        steering = float(self.model(x, training=False).numpy()[0, 0])
        steering = float(np.clip(steering, -1.0, 1.0))
        self.controller.set_desired(self.target_speed(steering))
        throttle = float(np.clip(self.controller.update(speed), -1.0, 1.0))
        if self.record_dir:
            stamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image.save(os.path.join(self.record_dir, f'{stamp}.jpg'))
        return steering, throttle


def create_app(driver: Driver):
    sio = socketio.Server(async_mode='eventlet', cors_allowed_origins='*')
    app = socketio.WSGIApp(sio)

    def send_control(sid, steering, throttle):
        sio.emit('steer', data={'steering_angle': str(steering), 'throttle': str(throttle)}, room=sid)

    @sio.on('connect')
    def connect(sid, environ):
        print('simulator connected', sid)
        send_control(sid, 0.0, 0.0)

    @sio.on('telemetry')
    def telemetry(sid, data):
        if not data:  # sim is in manual mode
            sio.emit('manual', data={}, room=sid)
            return
        steering, throttle = driver(data)
        print(f'steering {steering:+.3f}  throttle {throttle:+.3f}  speed {float(data["speed"]):5.1f}')
        send_control(sid, steering, throttle)

    return app, sio


def main():
    p = argparse.ArgumentParser(description='Remote driving for the Udacity simulator')
    p.add_argument('model', help='path to model.keras')
    p.add_argument('--speed', type=float, default=20.0, help='max speed (mph)')
    p.add_argument('--min-speed', type=float, default=10.0)
    p.add_argument('--port', type=int, default=4567)
    p.add_argument('--record', default='', help='folder to save the frames the car saw')
    args = p.parse_args()
    if args.record:
        os.makedirs(args.record, exist_ok=True)
    model = keras.models.load_model(args.model)
    app, _ = create_app(Driver(model, args.speed, args.min_speed, args.record or None))
    print(f'waiting for the simulator on port {args.port} (start it in Autonomous mode)')
    eventlet.wsgi.server(eventlet.listen(('', args.port)), app, log_output=False)


if __name__ == '__main__':
    main()
