import json
import shutil

import pytest
from PytorchWildlife.models.detection import MegaDetectorV6
from PytorchWildlife.utils import post_process

from constants import ANIMAL, PERSON

JSON_PHOTOS = ["coyote_day.jpg", "empty_night.jpg", "person_day.jpg"]


@pytest.fixture
def results(detect):
    return [detect(photo) for photo in JSON_PHOTOS]


def save_and_load(results, path, **options):
    post_process.save_detection_json(
        results, str(path), categories=MegaDetectorV6.CLASS_NAMES, **options
    )
    return json.loads(path.read_text())


def test_saved_json_has_annotations_and_categories(results, tmp_path):
    saved = save_and_load(results, tmp_path / "detections.json")
    assert set(saved) == {"annotations", "categories"}
    assert len(saved["annotations"]) == len(JSON_PHOTOS)


def test_saved_json_matches_detection_results(results, tmp_path):
    saved = save_and_load(results, tmp_path / "detections.json")
    for result, annotation in zip(results, saved["annotations"]):
        detections = result["detections"]
        assert annotation["img_id"] == result["img_id"]
        assert annotation["category"] == list(detections.class_id)
        assert annotation["confidence"] == pytest.approx(list(detections.confidence))
        assert len(annotation["bbox"]) == len(detections)


def test_excluded_category_is_removed(results, tmp_path):
    saved = save_and_load(
        results, tmp_path / "detections.json", exclude_category_ids=[PERSON]
    )
    for annotation in saved["annotations"]:
        assert PERSON not in annotation["category"]


def test_exclude_file_path_makes_paths_relative(results, tmp_path, images_dir):
    saved = save_and_load(
        results, tmp_path / "detections.json", exclude_file_path=str(images_dir)
    )
    names = [annotation["img_id"] for annotation in saved["annotations"]]
    assert names == JSON_PHOTOS


@pytest.fixture
def source_folder(tmp_path, images_dir):
    folder = tmp_path / "photos"
    folder.mkdir()
    for photo in ["coyote_day.jpg", "empty_night.jpg", "person_day.jpg"]:
        shutil.copy(images_dir / photo, folder / photo)
    return folder


def write_detection_json(path, annotations):
    path.write_text(json.dumps({"annotations": annotations, "categories": None}))


def make_annotation(img_id, categories=(), confidences=()):
    return {
        "img_id": img_id,
        "bbox": [[0, 0, 10, 10] for _ in categories],
        "category": list(categories),
        "confidence": list(confidences),
    }


def sort_photos(tmp_path, source_folder, annotations, threshold=0.2):
    json_file = tmp_path / "detections.json"
    write_detection_json(json_file, annotations)
    destination = tmp_path / "sorted"
    post_process.detection_folder_separation(
        str(json_file), str(source_folder), str(destination), threshold
    )
    return destination


def test_folder_separation_sorts_animal_and_empty_photos(tmp_path, source_folder):
    destination = sort_photos(
        tmp_path,
        source_folder,
        [
            make_annotation("coyote_day.jpg", [ANIMAL], [0.9]),
            make_annotation("empty_night.jpg"),
        ],
    )
    assert (destination / "Animal" / "coyote_day.jpg").exists()
    assert (destination / "No_animal" / "empty_night.jpg").exists()


def test_folder_separation_puts_person_photos_in_no_animal(tmp_path, source_folder):
    destination = sort_photos(
        tmp_path, source_folder, [make_annotation("person_day.jpg", [PERSON], [0.9])]
    )
    assert (destination / "No_animal" / "person_day.jpg").exists()


def test_folder_separation_keeps_the_originals(tmp_path, source_folder):
    sort_photos(
        tmp_path, source_folder, [make_annotation("coyote_day.jpg", [ANIMAL], [0.9])]
    )
    assert (source_folder / "coyote_day.jpg").exists()


def test_animal_exactly_at_threshold_is_sorted_as_no_animal(tmp_path, source_folder):
    # Characterization test (F13): sorting uses ">" while detection uses ">=",
    # so an animal exactly at the threshold is detected but not sorted as one.
    destination = sort_photos(
        tmp_path, source_folder, [make_annotation("coyote_day.jpg", [ANIMAL], [0.2])]
    )
    assert (destination / "No_animal" / "coyote_day.jpg").exists()


@pytest.mark.xfail(
    raises=shutil.SameFileError,
    strict=True,
    reason="Known bug in 1.3.0 (F11): absolute paths copy photos onto themselves",
)
def test_folder_separation_works_with_absolute_paths(tmp_path, source_folder):
    photo = source_folder / "coyote_day.jpg"
    destination = sort_photos(
        tmp_path, source_folder, [make_annotation(str(photo), [ANIMAL], [0.9])]
    )
    assert (destination / "Animal" / "coyote_day.jpg").exists()
