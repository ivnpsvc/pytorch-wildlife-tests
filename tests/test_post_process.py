import json

import pytest
from PytorchWildlife.models.detection import MegaDetectorV6
from PytorchWildlife.utils import post_process

from constants import PERSON

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
