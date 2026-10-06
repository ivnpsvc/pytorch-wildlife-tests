import pytest

ANIMAL = 0
PERSON = 1
VEHICLE = 2
THRESHOLD = 0.2


def test_three_cats_photo_has_at_least_three_animals(detector, images_dir):
    result = detector.single_image_detection(
        str(images_dir / "animals_three_cats_day.jpg"), det_conf_thres=THRESHOLD
    )
    assert list(result["detections"].class_id).count(ANIMAL) >= 3


def test_car_photo_has_vehicle_detection(detector, images_dir):
    # The car is detected with only 0.22 confidence. A lower threshold
    # keeps a safe margin, so small numeric differences between machines
    # can't flip the result.
    result = detector.single_image_detection(
        str(images_dir / "vehicle_car_day.jpg"), det_conf_thres=0.1
    )
    assert VEHICLE in result["detections"].class_id


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="Model limitation: misses the bird when the camera is upside down",
)
def test_upside_down_camera_photo_has_animal_detection(detector, images_dir):
    result = detector.single_image_detection(
        str(images_dir / "animal_upside_down_camera.jpg"), det_conf_thres=THRESHOLD
    )
    assert ANIMAL in result["detections"].class_id


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="Model limitation: animal too close to the camera, confidence 0.14",
)
def test_too_close_photo_has_animal_detection(detector, images_dir):
    result = detector.single_image_detection(
        str(images_dir / "animal_too_close_blurry.jpg"), det_conf_thres=THRESHOLD
    )
    assert ANIMAL in result["detections"].class_id


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="Model limitation: person with only legs visible is not detected",
)
def test_person_and_dog_photo_has_person_detection(detector, images_dir):
    result = detector.single_image_detection(
        str(images_dir / "person_and_dog_day.jpg"), det_conf_thres=THRESHOLD
    )
    assert PERSON in result["detections"].class_id


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="Model limitation: branch and rock detected as an animal (0.65)",
)
def test_branch_across_lens_photo_has_no_detections(detector, images_dir):
    result = detector.single_image_detection(
        str(images_dir / "empty_branch_across_lens.jpg"), det_conf_thres=THRESHOLD
    )
    assert len(result["detections"]) == 0