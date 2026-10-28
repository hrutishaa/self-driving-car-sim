"""Fast unit tests for the preprocessing and augmentation in utils.py.

These run in well under a second and don't need TensorFlow:

    pytest test_utils.py
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402
from utils import INPUT_SHAPE, augment, crop, load_image, preprocess, random_flip  # noqa: E402


def frame(h=160, w=320, seed=0):
    return (np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)


def test_preprocess_gives_network_input_shape():
    assert preprocess(frame()).shape == INPUT_SHAPE == (66, 200, 3)


def test_preprocess_scales_crop_for_other_resolutions():
    # same shape out no matter what size the camera renders at
    assert preprocess(frame(320, 640)).shape == INPUT_SHAPE


def test_crop_removes_sky_and_hood_rows():
    img = np.zeros((160, 320, 3), np.uint8)
    img[:60] = 255          # sky
    img[-25:] = 128         # hood
    out = crop(img)
    assert out.shape == (75, 320, 3)
    assert out.max() == 0   # only the road is left


def test_flip_mirrors_image_and_negates_steering():
    img = frame()

    class AlwaysFlip:
        def random(self):
            return 0.0

    out, steer = random_flip(img, 0.3, AlwaysFlip())
    assert steer == pytest.approx(-0.3)
    assert np.array_equal(out, img[:, ::-1, :])
    assert out.flags['C_CONTIGUOUS']  # cv2 needs contiguous arrays


def test_load_image_finds_files_recorded_on_another_machine(tmp_path):
    (tmp_path / 'IMG').mkdir()
    bgr = frame(20, 40)
    cv2.imwrite(str(tmp_path / 'IMG' / 'center_1.png'), bgr)
    # the sim writes absolute Windows paths from the recording machine
    img = load_image(str(tmp_path), r'  C:\Users\someone\sim\IMG\center_1.png ')
    assert np.array_equal(img, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path), 'nope.jpg')


def test_side_cameras_shift_steering(tmp_path):
    for name in ('c.png', 'l.png', 'r.png'):
        cv2.imwrite(str(tmp_path / name), frame(20, 40))

    class Pick:
        def __init__(self, choice):
            self.choice = choice

        def integers(self, n):
            return self.choice

    corr = utils.SIDE_CAMERA_CORRECTION
    assert utils.choose_image(str(tmp_path), 'c.png', 'l.png', 'r.png', 0.1, Pick(0))[1] == pytest.approx(0.1)
    assert utils.choose_image(str(tmp_path), 'c.png', 'l.png', 'r.png', 0.1, Pick(1))[1] == pytest.approx(0.1 + corr)
    assert utils.choose_image(str(tmp_path), 'c.png', 'l.png', 'r.png', 0.1, Pick(2))[1] == pytest.approx(0.1 - corr)
    # a row with no side image falls back to the center camera
    assert utils.choose_image(str(tmp_path), 'c.png', '', None, 0.1, Pick(1))[1] == pytest.approx(0.1)


def test_augment_keeps_steering_in_range(tmp_path):
    cv2.imwrite(str(tmp_path / 'c.png'), frame())
    rng = np.random.default_rng(0)
    for _ in range(200):
        img, steer = augment(str(tmp_path), 'c.png', 'c.png', 'c.png', 0.95, rng, extra=True)
        assert -1.0 <= steer <= 1.0
        assert img.shape == (160, 320, 3) and img.dtype == np.uint8
