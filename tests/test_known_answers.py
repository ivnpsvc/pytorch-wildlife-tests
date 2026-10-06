import pytest

ANIMAL = 0
PERSON = 1


@pytest.mark.parametrize(
    "photo, expected_class",
    [
        ("coyote_day.jpg", ANIMAL),
        ("deer_night.jpg", ANIMAL),
        ("raccoon_partial_night.jpg", ANIMAL),
        ("animals_three_cats_day.jpg", ANIMAL),
        ("animal_bird_day.jpg", ANIMAL),
        ("animal_squirrel_day.jpg", ANIMAL),
        ("animal_mountain_lion_night.jpg", ANIMAL),
        ("animal_deer_low_resolution.jpg", ANIMAL),
        ("animal_small_distant_day.jpg", ANIMAL),
        ("person_day.jpg", PERSON),
    ],
)
def test_photo_has_expected_detection(detect, photo, expected_class):
    result = detect(photo)
    assert expected_class in result["detections"].class_id


@pytest.mark.parametrize(
    "photo",
    [
        "empty_fence_day.jpg",
        "empty_hillside_day.jpg",
        "empty_night.jpg",
        "empty_vegetation_in_front.jpg",
    ],
)
def test_empty_photo_has_no_detections(detect, photo):
    result = detect(photo)
    assert len(result["detections"]) == 0