from pathlib import Path

from PytorchWildlife.models import detection as pw_detection

IMAGES = Path(__file__).parent.parent / "images"
ANIMAL = 0
THRESHOLD = 0.2


def test_photo_with_animal_has_animal_detection():
    model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
    result = model.single_image_detection(
        str(IMAGES / "coyote_day.jpg"), det_conf_thres=THRESHOLD
    )
    detections = result["detections"]
    assert len(detections) >= 1
    assert ANIMAL in detections.class_id


def test_empty_photo_has_no_detections():
    model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
    result = model.single_image_detection(
        str(IMAGES / "empty_fence_day.jpg"), det_conf_thres=THRESHOLD
    )
    detections = result["detections"]
    assert len(detections) == 0