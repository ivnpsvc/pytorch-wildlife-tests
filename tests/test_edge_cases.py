from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError
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

def test_tiny_image_runs_without_detections(tmp_path):
    tiny_photo = tmp_path / "tiny.jpg"
    Image.new("RGB", (1, 1)).save(tiny_photo)
    model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
    result = model.single_image_detection(str(tiny_photo))
    assert len(result["detections"]) == 0


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="Known bug in 1.3.0: default version 'yolov9c' is not a valid version",
)
def test_model_can_be_created_with_default_version():
    model = pw_detection.MegaDetectorV6()
    assert model is not None

def test_truncated_photo_is_processed_without_error(tmp_path):
    # Characterization test: documents current behavior, not necessarily
    # correct behavior. PytorchWildlife sets PIL's LOAD_TRUNCATED_IMAGES
    # to True, so a cut-off photo is filled with gray instead of raising
    # an error. The coyote is lost and the photo looks empty.
    # Open question for the maintainers: is this intended?
    full_photo = (IMAGES / "coyote_day.jpg").read_bytes()
    truncated_photo = tmp_path / "truncated.jpg"
    truncated_photo.write_bytes(full_photo[:20000])
    model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
    result = model.single_image_detection(str(truncated_photo))
    assert len(result["detections"]) == 0