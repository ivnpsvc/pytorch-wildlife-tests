from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from constants import ANIMAL


@pytest.fixture
def coyote(images_dir):
    return Image.open(images_dir / "coyote_day.jpg").convert("RGB")


@pytest.mark.parametrize(
    "filename, mode",
    [
        ("coyote.png", "RGB"),
        ("coyote.webp", "RGB"),
        ("coyote_grayscale.jpg", "L"),
        ("coyote_transparent.png", "RGBA"),
        ("coyote_cmyk.jpg", "CMYK"),
    ],
)
def test_other_formats_and_color_modes_are_detected(
    detector, coyote, tmp_path, filename, mode
):
    photo = tmp_path / filename
    coyote.convert(mode).save(photo)
    result = detector.single_image_detection(str(photo))
    assert ANIMAL in result["detections"].class_id


def test_portrait_photo_is_detected_with_boxes_inside(detector, coyote, tmp_path):
    photo = tmp_path / "coyote_portrait.jpg"
    portrait = coyote.rotate(90, expand=True)
    portrait.save(photo)
    result = detector.single_image_detection(str(photo))
    assert ANIMAL in result["detections"].class_id
    width, height = portrait.size
    for x1, y1, x2, y2 in result["detections"].xyxy:
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height


def test_array_input_gives_the_same_result_as_a_path(detector, coyote, images_dir):
    from_path = detector.single_image_detection(str(images_dir / "coyote_day.jpg"))
    from_array = detector.single_image_detection(np.array(coyote))
    path_detections = from_path["detections"]
    array_detections = from_array["detections"]
    assert list(array_detections.class_id) == list(path_detections.class_id)
    assert array_detections.confidence == pytest.approx(
        path_detections.confidence, abs=0.01
    )


def test_grayscale_array_input_is_accepted(detector, coyote):
    grayscale = np.array(coyote.convert("L"))
    result = detector.single_image_detection(grayscale)
    assert ANIMAL in result["detections"].class_id


def test_array_input_uses_img_path_as_img_id(detector, coyote):
    result = detector.single_image_detection(
        np.array(coyote), img_path="camera_1/coyote.jpg"
    )
    assert result["img_id"] == "camera_1/coyote.jpg"


def test_array_input_without_img_path_gets_the_text_none_as_img_id(detector, coyote):
    # Characterization test (F15): documents current behavior, not necessarily
    # correct behavior. Without img_path, img_id becomes the text "None".
    result = detector.single_image_detection(np.array(coyote))
    assert result["img_id"] == "None"


@pytest.mark.xfail(
    raises=AttributeError,
    strict=True,
    reason="Known bug in 1.3.0 (F5): only str paths are accepted, not pathlib.Path",
)
def test_pathlib_path_input_is_accepted(detector, images_dir):
    result = detector.single_image_detection(Path(images_dir / "coyote_day.jpg"))
    assert ANIMAL in result["detections"].class_id
