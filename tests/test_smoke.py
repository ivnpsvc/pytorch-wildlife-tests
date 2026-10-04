from pathlib import Path

from PytorchWildlife.models import detection as pw_detection

IMAGES = Path(__file__).parent.parent / "images"


def test_detection_runs_on_a_photo():
    model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
    result = model.single_image_detection(str(IMAGES / "coyote_day.jpg"))
    assert "detections" in result