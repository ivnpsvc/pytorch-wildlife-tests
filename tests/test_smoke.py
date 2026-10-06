def test_detection_runs_on_a_photo(detector, images_dir):
    result = detector.single_image_detection(str(images_dir / "coyote_day.jpg"))
    assert "detections" in result