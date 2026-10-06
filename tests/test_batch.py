import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

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


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="Known bug in 1.3.0 (F9): single-image detection swaps red and blue",
)
def test_batch_and_single_detection_give_same_results(detector, photo_folder):
    batch_results = by_photo(detector.batch_image_detection(str(photo_folder)))
    for photo in BATCH_PHOTOS:
        single = detector.single_image_detection(str(photo_folder / photo))
        in_batch = batch_results[photo]["detections"]
        alone = single["detections"]
        assert list(alone.class_id) == list(in_batch.class_id)
        assert alone.confidence == pytest.approx(in_batch.confidence, abs=0.01)


def test_one_non_image_jpg_stops_the_whole_batch(detector, photo_folder):
    # Characterization test (F10): documents current behavior, not necessarily
    # correct behavior. One unreadable file makes the whole batch fail, and no
    # results are returned for the valid photos.
    # Open question for the maintainers: should the batch skip and report it?
    (photo_folder / "fake.jpg").write_text("hello, I am not a picture")
    with pytest.raises(UnidentifiedImageError):
        detector.batch_image_detection(str(photo_folder))
