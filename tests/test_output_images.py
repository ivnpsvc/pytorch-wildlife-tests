import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PytorchWildlife.utils import post_process

OUTPUT_PHOTOS = ["coyote_day.jpg", "empty_night.jpg", "person_day.jpg"]


@pytest.fixture
def results(detect):
    return [detect(photo) for photo in OUTPUT_PHOTOS]


def changed_pixel_share(first, second):
    # Share of pixels that differ clearly. Small differences are expected,
    # because saving a JPEG again changes some pixels slightly.
    first = np.array(Image.open(first).convert("RGB")).astype(int)
    second = np.array(Image.open(second).convert("RGB")).astype(int)
    return (np.abs(first - second) > 40).mean()


def file_names(folder):
    return sorted(path.name for path in folder.iterdir())


def test_annotated_images_one_per_photo(results, tmp_path):
    post_process.save_detection_images(results, str(tmp_path))
    assert file_names(tmp_path) == sorted(OUTPUT_PHOTOS)


def test_annotated_images_keep_the_original_size(results, tmp_path, images_dir):
    post_process.save_detection_images(results, str(tmp_path))
    for photo in OUTPUT_PHOTOS:
        annotated = Image.open(tmp_path / photo)
        original = Image.open(images_dir / photo)
        assert annotated.size == original.size


def test_annotated_image_has_boxes_drawn_on_it(results, tmp_path, images_dir):
    post_process.save_detection_images(results, str(tmp_path))
    photo = "coyote_day.jpg"
    assert changed_pixel_share(tmp_path / photo, images_dir / photo) > 0.005


def test_photo_without_detections_is_saved_unchanged(results, tmp_path, images_dir):
    post_process.save_detection_images(results, str(tmp_path))
    photo = "empty_night.jpg"
    assert changed_pixel_share(tmp_path / photo, images_dir / photo) < 0.001


def test_annotated_images_accept_a_single_result(results, tmp_path):
    post_process.save_detection_images(results[0], str(tmp_path))
    assert file_names(tmp_path) == ["coyote_day.jpg"]


def test_dot_images_one_per_photo(results, tmp_path):
    post_process.save_detection_images_dots(results, str(tmp_path))
    assert file_names(tmp_path) == sorted(OUTPUT_PHOTOS)


def test_crops_one_file_per_detection_named_class_index_photo(results, tmp_path):
    post_process.save_crop_images(results, str(tmp_path))
    expected = []
    for result in results:
        photo = Path(result["img_id"]).name
        for index, class_id in enumerate(result["detections"].class_id):
            expected.append(f"{class_id}_{index}_{photo}")
    assert file_names(tmp_path) == sorted(expected)


def test_crop_size_matches_the_box_size(results, tmp_path):
    post_process.save_crop_images(results, str(tmp_path))
    for result in results:
        photo = Path(result["img_id"]).name
        detections = result["detections"]
        for index, (box, class_id) in enumerate(
            zip(detections.xyxy, detections.class_id)
        ):
            x1, y1, x2, y2 = box
            crop = Image.open(tmp_path / f"{class_id}_{index}_{photo}")
            width, height = crop.size
            assert width == pytest.approx(x2 - x1, abs=2)
            assert height == pytest.approx(y2 - y1, abs=2)


def test_input_dir_keeps_the_subfolder_structure(detector, tmp_path, images_dir):
    camera_folder = tmp_path / "photos" / "camera_1"
    camera_folder.mkdir(parents=True)
    shutil.copy(images_dir / "coyote_day.jpg", camera_folder / "coyote_day.jpg")
    result = detector.single_image_detection(str(camera_folder / "coyote_day.jpg"))
    output = tmp_path / "annotated"
    post_process.save_detection_images(
        [result], str(output), input_dir=str(tmp_path / "photos")
    )
    assert (output / "camera_1" / "coyote_day.jpg").exists()


def test_overwrite_false_keeps_existing_files(results, tmp_path):
    (tmp_path / "notes.txt").write_text("Do not delete")
    post_process.save_detection_images(results, str(tmp_path), overwrite=False)
    assert (tmp_path / "notes.txt").exists()


def test_overwrite_true_deletes_everything_in_the_output_folder(results, tmp_path):
    # Characterization test (F14): documents current behavior, not necessarily
    # correct behavior. overwrite=True empties the whole output folder first,
    # including files the library did not create.
    (tmp_path / "notes.txt").write_text("Do not delete")
    post_process.save_detection_images(results[:1], str(tmp_path), overwrite=True)
    assert file_names(tmp_path) == ["coyote_day.jpg"]
