"""Preprocessing and augmentation shared by model.py and drive.py.

preprocess() has to be the same at train and drive time, otherwise the car swerves.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
CROP_TOP, CROP_BOTTOM = 60, 25          # rows of sky / car hood in the 160x320 Udacity image
SIDE_CAMERA_CORRECTION = 0.2            # steering nudge for left/right camera images


def load_image(data_dir: str, image_file: str) -> np.ndarray:
    # the sim stores absolute paths from whatever machine recorded it
    image_file = image_file.strip()
    candidates = [image_file, os.path.join(data_dir, image_file),
                  os.path.join(data_dir, 'IMG', os.path.basename(image_file.replace('\\', '/')))]
    for path in candidates:
        if os.path.isfile(path):
            bgr = cv2.imread(path)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raise FileNotFoundError(f'image not found: {image_file} (looked in {data_dir})')


def crop(image: np.ndarray) -> np.ndarray:
    h = image.shape[0]
    top = round(CROP_TOP * h / 160)
    bottom = h - round(CROP_BOTTOM * h / 160)
    return image[top:bottom, :, :]


def resize(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)


def preprocess(image: np.ndarray) -> np.ndarray:
    # normalization is a layer in the model
    return resize(crop(image))


# augmentation
def choose_image(data_dir, center, left, right, steering, rng):
    choice = rng.integers(3)
    if choice == 1 and isinstance(left, str) and left.strip():
        return load_image(data_dir, left), steering + SIDE_CAMERA_CORRECTION
    if choice == 2 and isinstance(right, str) and right.strip():
        return load_image(data_dir, right), steering - SIDE_CAMERA_CORRECTION
    return load_image(data_dir, center), steering


def random_flip(image, steering, rng):
    if rng.random() < 0.5:
        return np.ascontiguousarray(image[:, ::-1, :]), -steering
    return image, steering


def random_translate(image, steering, rng, range_x=60, range_y=10):
    # not in the guide, but helps on the real sim. 0.002 steering per pixel
    tx = range_x * (rng.random() - 0.5)
    ty = range_y * (rng.random() - 0.5)
    steering += tx * 0.002
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = image.shape[:2]
    return cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REPLICATE), steering


def random_brightness(image, rng):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.6 + 0.8 * rng.random()), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment(data_dir, center, left, right, steering, rng, extra=False):
    image, steering = choose_image(data_dir, center, left, right, steering, rng)
    image, steering = random_flip(image, steering, rng)
    if extra:
        image, steering = random_translate(image, steering, rng)
        image = random_brightness(image, rng)
    return image, float(np.clip(steering, -1.0, 1.0))
