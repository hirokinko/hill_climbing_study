import argparse
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy
import requests
from PIL import Image

RAW_IMAGE_URL = "https://pbs.twimg.com/media/EYf0zecUEAAPPO2?format=jpg&name=large"
DEFAULT_MAX_LONGSIDE_LENGTH = 400
DEFAULT_ROUND_PER_GENERATION = 1
DEFAULT_TERMINATION_CIRCLE_RADIUS = 0.5

ARTIFACTS_DESTINATION = Path(
    os.environ.get("ARTIFACTS_DESTINATION", f"{os.getcwd()}/var/artifacts")
)
FILE_OUTPUTS_DESTINATION = Path(
    os.environ.get("FILE_OUTPUTS_DESTINATION", f"{os.getcwd()}/var/file_outputs")
)
RESULT_UPLOAD_DIR = Path(f"{FILE_OUTPUTS_DESTINATION}") / "images"
if not RESULT_UPLOAD_DIR.exists():
    RESULT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_image_url", type=str, required=True)
    parser.add_argument("--max_longside_length", type=int, default=DEFAULT_MAX_LONGSIDE_LENGTH)
    parser.add_argument("--round_per_generation", type=int, default=DEFAULT_ROUND_PER_GENERATION)
    parser.add_argument(
        "--termination_circle_radius", type=float, default=DEFAULT_TERMINATION_CIRCLE_RADIUS
    )
    return parser.parse_args()


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
    round_per_generation: int = DEFAULT_ROUND_PER_GENERATION,
    cool_down_count: int = 100,
) -> Tuple[numpy.ndarray, float, int]:
    working_image = current_image.copy()
    current_difference = cv2.absdiff(original_image, current_image).sum()

    for _ in range(round_per_generation):
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


def download_original_image(raw_image_url: str, filename: str) -> None:
    res = requests.get(raw_image_url)
    with open(filename, "wb") as f:
        f.write(res.content)


def resize_original_image(image: numpy.ndarray, max_longside_length: float) -> numpy.ndarray:
    original_height, original_width = image.shape[:2]
    resize_rate = (
        max_longside_length / original_width
        if original_width >= original_height
        else max_longside_length / original_height
    )
    resized_size = (int(original_width * resize_rate), int(original_height * resize_rate))
    resized_image = cv2.resize(image, resized_size)
    return resized_image


# TODO write test
def draw_current_image(
    generation: int, current_image: numpy.ndarray, sub_generation: Optional[int] = None
) -> Path:
    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
    sub_geneation_str = f"_{sub_generation:03}" if sub_generation is not None else ""
    current_image_filename = (
        RESULT_UPLOAD_DIR / f"image_generation_{generation}{sub_geneation_str}.jpeg"
    )
    cv2.imwrite(str(current_image_filename), current_image)
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
    return current_image_filename


# TODO write test
def draw_pointillism(args: argparse.Namespace) -> None:
    try:
        file_pointer, filename = tempfile.mkstemp(suffix=".jpeg")

        download_original_image(args.raw_image_url, filename)
        original_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = resize_original_image(original_image, args.max_longside_length)

        current_image = numpy.full(original_image.shape, 127, dtype=numpy.uint8)
        current_image = get_alphaed_image(current_image, current_image.copy())

        generation = 0
        max_circle_radius = 100.0
        while max_circle_radius > args.termination_circle_radius:
            current_image, max_circle_radius, current_difference = annealing_draw(
                original_image=original_image,
                current_image=current_image,
                max_circle_radius=max_circle_radius,
                round_per_generation=args.round_per_generation,
                cool_down_count=100,
            )
            if generation % 50 == 0:
                draw_current_image(generation=generation, current_image=current_image)
            generation += 1

        # last generation
        draw_current_image(generation=generation, current_image=current_image)

        # original
        for sub_generation in range(50):
            draw_current_image(
                generation=generation + 1,
                current_image=original_image,
                sub_generation=sub_generation,
            )

    finally:
        os.close(file_pointer)
        os.remove(filename)


# TODO write test
def generate_gif_annimation() -> None:
    images = [
        Image.open(str(image_filepath))
        for image_filepath in sorted(
            RESULT_UPLOAD_DIR.glob("*.jpeg"), key=lambda p: p.stat().st_mtime_ns
        )
    ]
    images[0].save(
        str(RESULT_UPLOAD_DIR / "generated.gif"),
        save_all=True,
        append_images=images[1:],
        optimize=True,
        duration=100,
        loop=0,
    )


if __name__ == "__main__":
    args = parse_args()

    draw_pointillism(args)
    generate_gif_annimation()
