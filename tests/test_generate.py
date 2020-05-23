from unittest.mock import call

import cv2
import numpy
import pytest

from hill_climbing_study.generate import (
    _center,
    _color,
    _radius,
    draw_circle_on_image,
    resize_original_image,
)


@pytest.mark.parametrize("max_circle_radius,expect", [(0.999, 1), (1.000, 2)])
def test__radius(mocker, max_circle_radius, expect):
    mock = mocker.patch("numpy.random.randint", return_value=int(max_circle_radius + 1))
    assert _radius(max_circle_radius) == expect
    mock.has_calls([call(1, 1), call(1, 2)], any_order=False)


def test__center(mocker):
    def _side_effect(low, high):
        return high

    mock = mocker.patch("numpy.random.randint", side_effect=_side_effect)
    assert _center(300, 400) == (400, 300)
    mock.has_calls([call(400), call(300)], any_order=False)
    mock.reset_mock()


def test__color(mocker):
    mock = mocker.patch("numpy.random.randint", return_value=128)
    assert _color() == (128, 128, 128)
    mock.has_calls([call(128), call(128), call(128)], any_order=False)


def test_draw_circle_on_image(mocker):
    _radius_mock = mocker.patch("hill_climbing_study.generate._radius", return_value=110)
    _center_mock = mocker.patch("hill_climbing_study.generate._center", return_value=(200, 150))
    _color_mock = mocker.patch("hill_climbing_study.generate._color", return_value=(63, 127, 255))
    image = numpy.full((300, 400), 127, dtype=numpy.uint8)
    expect_image = cv2.circle(image.copy(), (200, 150), 110, (63, 127, 255), -1)
    assert numpy.array_equal(draw_circle_on_image(image, 110), expect_image)
    _radius_mock.assert_called_once()
    _center_mock.assert_called_once()
    _color_mock.assert_called_once()


def test_resize_original_image():
    original_image = cv2.imread("tests/data/test_image.jpeg")
    resized = resize_original_image(original_image)
    assert cv2.absdiff(resized, cv2.imread("tests/data/test_resized.jpeg")).sum() < 1250000
