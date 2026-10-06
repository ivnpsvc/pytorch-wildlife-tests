import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

BATCH_PHOTOS = ["coyote_day.jpg", "empty_night.jpg", "person_day.jpg"]


@pytest.fixture
def photo_folder(tmp_path, images_dir):
    for photo in BATCH_PHOTOS:
        shutil.copy(images_dir / photo, tmp_path / photo)
    return tmp_path


def by_photo(results):
    return {Path(result["img_id"]).name: result for result in results}


def test_batch_returns_one_result_per_photo(detector, photo_folder):
    results = detector.batch_image_detection(str(photo_folder))
    assert sorted(by_photo(results)) == sorted(BATCH_PHOTOS)
    assert len(results) == len(BATCH_PHOTOS)


def test_batch_size_does_not_change_results(detector, photo_folder):
    one_by_one = by_photo(
        detector.batch_image_detection(str(photo_folder), batch_size=1)
    )
    all_at_once = by_photo(
        detector.batch_image_detection(str(photo_folder), batch_size=16)
    )
    for photo in BATCH_PHOTOS:
        small = one_by_one[photo]["detections"]
        large = all_at_once[photo]["detections"]
        assert list(small.class_id) == list(large.class_id)
        assert small.confidence == pytest.approx(large.confidence, abs=0.01)


def test_batch_includes_photos_in_subfolders(detector, photo_folder, images_dir):
    subfolder = photo_folder / "camera_2"
    subfolder.mkdir()
    shutil.copy(images_dir / "deer_night.jpg", subfolder / "deer_night.jpg")
    results = detector.batch_image_detection(str(photo_folder))
    assert "deer_night.jpg" in by_photo(results)


def test_batch_ignores_non_image_files(detector, photo_folder):
    (photo_folder / "notes.txt").write_text("Camera 2, battery low")
    results = detector.batch_image_detection(str(photo_folder))
    assert len(results) == len(BATCH_PHOTOS)


def test_batch_on_empty_folder_returns_no_results(detector, tmp_path):
    assert detector.batch_image_detection(str(tmp_path)) == []


def test_batch_with_arrays_uses_position_as_img_id(detector, images_dir):
    arrays = [
        np.array(Image.open(images_dir / photo).convert("RGB"))
        for photo in BATCH_PHOTOS
    ]
    results = detector.batch_image_detection(arrays)
    assert [result["img_id"] for result in results] == ["0", "1", "2"]
