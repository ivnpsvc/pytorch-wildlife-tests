import csv
import json
import urllib.request
from pathlib import Path

import pytest

from constants import ANIMAL

# Downloads about 80 MB of photos and runs 400 detections.
pytestmark = pytest.mark.slow

SAMPLE_FILE = Path(__file__).parent / "data" / "accuracy_sample.csv"
BASE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/"
)
PHOTO_CACHE = Path.home() / ".cache" / "wildlife-tests" / "accuracy"
REPORT_FILE = Path(__file__).parent.parent / "reports" / "accuracy.json"
THRESHOLD = 0.2

# Baseline measured on 2026-10-07 with PytorchWildlife 1.3.0 and MDV6-yolov9-c:
#   batch:  recall 0.93, false positive rate 0.15
#   single: recall 0.92, false positive rate 0.13
# The limits below leave a margin of 5 percentage points, so the tests catch
# real regressions rather than small differences between machines.
# Some "empty" photos in the sample actually show an animal (label errors in the
# dataset), so the false positive rate is an upper bound. See tests/data/README.md.
MIN_RECALL = {"batch": 0.88, "single": 0.87}
MAX_FALSE_POSITIVE_RATE = {"batch": 0.20, "single": 0.18}


def load_sample():
    with open(SAMPLE_FILE, newline="") as file:
        return list(csv.DictReader(file))


def download(url, path):
    # Download to a temporary name first, so an interrupted download never
    # leaves a truncated photo behind (truncated photos are processed
    # silently, see F8).
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(url, partial)
    partial.rename(path)


@pytest.fixture(scope="module")
def sample():
    return load_sample()


@pytest.fixture(scope="module")
def sample_folder(sample):
    PHOTO_CACHE.mkdir(parents=True, exist_ok=True)
    for row in sample:
        path = PHOTO_CACHE / row["file_name"]
        if not path.exists():
            download(BASE_URL + row["file_name"], path)
    return PHOTO_CACHE


def found_animal(result):
    return ANIMAL in result["detections"].class_id


def measure(found, sample):
    animals = [row for row in sample if row["label"] == "animal"]
    empties = [row for row in sample if row["label"] == "empty"]
    missed = [row["file_name"] for row in animals if not found[row["file_name"]]]
    false_alarms = [row["file_name"] for row in empties if found[row["file_name"]]]
    return {
        "recall": 1 - len(missed) / len(animals),
        "false_positive_rate": len(false_alarms) / len(empties),
        "missed": missed,
        "false_alarms": false_alarms,
    }


@pytest.fixture(scope="module")
def accuracy(detector, sample, sample_folder):
    names = {row["file_name"] for row in sample}

    batch_results = detector.batch_image_detection(
        str(sample_folder), det_conf_thres=THRESHOLD
    )
    batch_found = {
        Path(result["img_id"]).name: found_animal(result)
        for result in batch_results
        if Path(result["img_id"]).name in names
    }

    single_found = {}
    for name in names:
        result = detector.single_image_detection(
            str(sample_folder / name), det_conf_thres=THRESHOLD
        )
        single_found[name] = found_animal(result)

    report = {
        "model": "MDV6-yolov9-c",
        "threshold": THRESHOLD,
        "animal_photos": sum(row["label"] == "animal" for row in sample),
        "empty_photos": sum(row["label"] == "empty" for row in sample),
        "batch": measure(batch_found, sample),
        "single": measure(single_found, sample),
    }
    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    return report


def test_sample_has_100_animal_and_100_empty_photos(sample):
    labels = [row["label"] for row in sample]
    assert labels.count("animal") == 100
    assert labels.count("empty") == 100


@pytest.mark.parametrize("mode", ["batch", "single"])
def test_recall_on_animal_photos(accuracy, mode):
    assert accuracy[mode]["recall"] >= MIN_RECALL[mode]


@pytest.mark.parametrize("mode", ["batch", "single"])
def test_false_positive_rate_on_empty_photos(accuracy, mode):
    assert accuracy[mode]["false_positive_rate"] <= MAX_FALSE_POSITIVE_RATE[mode]
