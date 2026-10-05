from pathlib import Path

import pytest
from PIL import UnidentifiedImageError
from PytorchWildlife.models import detection as pw_detection

IMAGES = Path(__file__).parent.parent / "images"


def test_missing_file_raises_file_not_found_error():
    model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
    with pytest.raises(FileNotFoundError):
        model.single_image_detection(str(IMAGES / "does_not_exist.jpg"))


def test_text_file_named_jpg_raises_unidentified_image_error(tmp_path):
    fake_photo = tmp_path / "not_a_photo.jpg"
    fake_photo.write_text("hello, I am not a picture")
    model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
    with pytest.raises(UnidentifiedImageError):
        model.single_image_detection(str(fake_photo))