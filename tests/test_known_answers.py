ANIMAL = 0
THRESHOLD = 0.2


def test_photo_with_animal_has_animal_detection(detector, images_dir):
    result = detector.single_image_detection(
        str(images_dir / "coyote_day.jpg"), det_conf_thres=THRESHOLD
    )
    detections = result["detections"]
    assert len(detections) >= 1
    assert ANIMAL in detections.class_id


def test_empty_photo_has_no_detections(detector, images_dir):
    result = detector.single_image_detection(
        str(images_dir / "empty_fence_day.jpg"), det_conf_thres=THRESHOLD
    )
    detections = result["detections"]
    assert len(detections) == 0