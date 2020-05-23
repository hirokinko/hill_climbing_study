import os
import tempfile
from functools import partial
from typing import Tuple

import cv2
import numpy
import requests

MAX_GENERATIONS = 5000
RAW_IMAGE_URL = "https://pbs.twimg.com/media/EYf0zecUEAAPPO2?format=jpg&name=large"
RESULT_UPLOAD_DIR = os.environ.get("RESULT_UPLOAD_DIR", f"{os.getcwd()}/var/image")
MAX_LONGSIDE_LENGTH = 400
LAST_MAX_CIRCLE_RADIUS = 0.5


def get_alphaed_image(circle_image: numpy.ndarray, working_image: numpy.ndarray) -> numpy.ndarray:
    alpha = 0.3
    beta = 1 - alpha
    return cv2.addWeighted(circle_image, alpha, working_image, beta, 0)


def _radius(max_circle_radius: float) -> int:
    if int(max_circle_radius + 1) == 1:
        return 1  # avoid ValueError when low and hight, both equals
    func = partial(numpy.random.randint, 1, int(max_circle_radius + 1))
    return func()


def _center(height: int, width: int) -> Tuple[int, int]:
    func = partial(numpy.random.randint, 0)
    return (func(width), func(height))  # x, y


def _color() -> Tuple[int, int, int]:
    rand_rgb = partial(numpy.random.randint, 0, 255)
    return (rand_rgb(), rand_rgb(), rand_rgb())  # R, G, B


def draw_circle_on_image(image: numpy.ndarray, max_circle_radius: float) -> numpy.ndarray:
    height, width = image.shape[:2]
    return cv2.circle(image, _center(height, width), _radius(max_circle_radius), _color(), -1)


# TODO write tests
def annealing_draw(
    original_image: numpy.ndarray,
    current_image: numpy.ndarray,
    max_circle_radius: float = 100.0,
    round: int = 1,
    cool_down_count: int = 100,
) -> Tuple[numpy.ndarray, float, int]:
    working_image = current_image.copy()
    current_difference = cv2.absdiff(original_image, current_image).sum()

    for _ in range(round):
        failed_count = 0

        while True:
            circle_image = draw_circle_on_image(
                image=working_image.copy(), max_circle_radius=max_circle_radius
            )
            alphaed_image = get_alphaed_image(circle_image, working_image)
            annealing_difference = cv2.absdiff(original_image, alphaed_image).sum()
            if annealing_difference < current_difference:
                current_difference = annealing_difference
                working_image = alphaed_image
                break
            else:
                failed_count += 1
                if failed_count > cool_down_count:
                    failed_count = 0
                    max_circle_radius *= 0.95

    return working_image, max_circle_radius, current_difference


# TODO Any image.
def download_original_image(filename: str) -> None:
    res = requests.get(RAW_IMAGE_URL)
    with open(filename, "wb") as f:
        f.write(res.content)


def resize_original_image(image: numpy.ndarray) -> numpy.ndarray:
    original_height, original_width = image.shape[:2]
    resize_rate = (
        MAX_LONGSIDE_LENGTH / original_width
        if original_width >= original_height
        else MAX_LONGSIDE_LENGTH / original_height
    )
    resized_size = (int(original_width * resize_rate), int(original_height * resize_rate))
    resized_image = cv2.resize(image, resized_size)
    return resized_image


def draw_current_image(generation, current_image) -> str:
    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
    current_image_filename = f"{RESULT_UPLOAD_DIR}/image_generation_{generation}.jpeg"
    cv2.imwrite(current_image_filename, current_image)
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
    return current_image_filename


def main() -> None:
    try:
        file_pointer, filename = tempfile.mkstemp(suffix=".jpeg")

        download_original_image(filename)
        original_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = resize_original_image(original_image)

        current_image = numpy.full(original_image.shape, 127, dtype=numpy.uint8)
        current_image = get_alphaed_image(current_image, current_image.copy())

        generation = 0
        max_circle_radius = 100.0
        while max_circle_radius > LAST_MAX_CIRCLE_RADIUS:
            current_image, max_circle_radius, current_difference = annealing_draw(
                original_image=original_image,
                current_image=current_image,
                max_circle_radius=max_circle_radius,
                round=10,
                cool_down_count=100,
            )
            if generation % 100 == 0:
                draw_current_image(generation=generation, current_image=current_image)
            generation += 1
    finally:
        os.close(file_pointer)
        os.remove(filename)


if __name__ == "__main__":
    main()
