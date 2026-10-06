from pathlib import Path

import pytest
from PIL import Image

PHOTOS = sorted(
    photo.name for photo in (Path(__file__).parent.parent / "images").glob("*.jpg")
)


@pytest.mark.parametrize("photo", PHOTOS)
def test_result_has_expected_fields(detect, photo):
    result = detect(photo)
    assert set(result.keys()) == {"img_id", "detections", "labels", "normalized_coords"}


@pytest.mark.parametrize("photo", PHOTOS)
def test_result_lists_have_matching_lengths(detect, photo):
    result = detect(photo)
    count = len(result["detections"])
    assert len(result["labels"]) == count
    assert len(result["normalized_coords"]) == count


@pytest.mark.parametrize("photo", PHOTOS)
def test_confidences_are_at_least_the_threshold(detect, detection_threshold, photo):
    result = detect(photo)
    for confidence in result["detections"].confidence:
        assert confidence >= detection_threshold


@pytest.mark.parametrize("photo", PHOTOS)
def test_boxes_are_inside_the_image(detect, images_dir, photo):
    result = detect(photo)
    width, height = Image.open(images_dir / photo).size
    for x1, y1, x2, y2 in result["detections"].xyxy:
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height


@pytest.mark.parametrize("photo", PHOTOS)
def test_normalized_coords_are_between_0_and_1(detect, photo):
    result = detect(photo)
    for box in result["normalized_coords"]:
        for value in box:
            assert 0 <= value <= 1
