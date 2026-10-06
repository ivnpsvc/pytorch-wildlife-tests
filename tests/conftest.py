from pathlib import Path

import pytest
from PytorchWildlife.models import detection as pw_detection


@pytest.fixture(scope="session")
def images_dir():
    return Path(__file__).parent.parent / "images"


@pytest.fixture(scope="session")
def detector():
    return pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")