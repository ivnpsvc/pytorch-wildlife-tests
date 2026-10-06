from pathlib import Path

import pytest
from PytorchWildlife.models import detection as pw_detection


@pytest.fixture(scope="session")
def images_dir():
    return Path(__file__).parent.parent / "images"


@pytest.fixture(scope="session")
def detector():
    return pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")


@pytest.fixture(scope="session")
def detection_threshold():
    return 0.2


@pytest.fixture(scope="session")
def detect(detector, images_dir, detection_threshold):
    results = {}

    def run_detection(photo):
        if photo not in results:
            results[photo] = detector.single_image_detection(
                str(images_dir / photo), det_conf_thres=detection_threshold
            )
        return results[photo]

    return run_detection
