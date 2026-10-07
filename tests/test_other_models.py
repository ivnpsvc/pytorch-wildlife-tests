import pytest
from PIL import Image
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.utils import post_process

from constants import ANIMAL, PERSON, VEHICLE

# Every test in this file downloads large model weights (about 1.1 GB in total).
pytestmark = pytest.mark.slow

DETECTOR_VERSIONS = ["MDV6-yolov9-e", "MDV6-yolov10-c", "MDV6-yolov10-e", "MDV5a"]

# Photos with a clear answer for every detector version. Empty photos other than
# empty_night.jpg are left out: the larger V6 versions find objects in them.
KNOWN_ANSWERS = [
    ("coyote_day.jpg", ANIMAL),
    ("deer_night.jpg", ANIMAL),
    ("raccoon_partial_night.jpg", ANIMAL),
    ("animals_three_cats_day.jpg", ANIMAL),
    ("animal_bird_day.jpg", ANIMAL),
    ("animal_squirrel_day.jpg", ANIMAL),
    ("animal_mountain_lion_night.jpg", ANIMAL),
    ("animal_deer_low_resolution.jpg", ANIMAL),
    ("animal_small_distant_day.jpg", ANIMAL),
    ("person_day.jpg", PERSON),
    ("vehicle_car_day.jpg", VEHICLE),
]

CLASSIFIERS = {
    "Serengeti": pw_classification.AI4GSnapshotSerengeti,
    "Amazon": pw_classification.AI4GAmazonRainforest,
    "Opossum": pw_classification.AI4GOpossum,
}


def create_detector(version):
    if version == "MDV5a":
        return pw_detection.MegaDetectorV5(version="a")
    return pw_detection.MegaDetectorV6(version=version)


@pytest.fixture(scope="module", params=DETECTOR_VERSIONS)
def other_detector(request):
    return create_detector(request.param)


@pytest.mark.parametrize("photo, expected_class", KNOWN_ANSWERS)
def test_other_detectors_find_known_answers(
    other_detector, images_dir, photo, expected_class
):
    result = other_detector.single_image_detection(str(images_dir / photo))
    assert expected_class in result["detections"].class_id


def test_other_detectors_find_nothing_in_empty_night_photo(other_detector, images_dir):
    result = other_detector.single_image_detection(str(images_dir / "empty_night.jpg"))
    assert len(result["detections"]) == 0


def test_other_detectors_return_the_same_result_fields(other_detector, images_dir):
    result = other_detector.single_image_detection(str(images_dir / "coyote_day.jpg"))
    assert set(result) == {"img_id", "detections", "labels", "normalized_coords"}


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="Known bug in 1.3.0 (F7): RT-DETR runs at 1280 px with the YOLO predictor",
)
def test_rtdetr_detects_a_clear_animal(images_dir):
    detector = pw_detection.MegaDetectorV6(version="MDV6-rtdetr-c")
    result = detector.single_image_detection(str(images_dir / "coyote_day.jpg"))
    assert ANIMAL in result["detections"].class_id


@pytest.fixture(scope="module", params=list(CLASSIFIERS))
def classifier(request):
    return CLASSIFIERS[request.param]()


@pytest.fixture
def crop(detect, tmp_path):
    post_process.save_crop_images([detect("coyote_day.jpg")], str(tmp_path))
    return tmp_path / f"{ANIMAL}_0_coyote_day.jpg"


def test_classification_result_has_expected_fields(classifier, crop):
    result = classifier.single_image_classification(str(crop), img_id="coyote")
    assert {"img_id", "prediction", "class_id", "confidence"} <= set(result)
    assert result["img_id"] == "coyote"


def test_classification_prediction_matches_class_id(classifier, crop):
    result = classifier.single_image_classification(str(crop))
    assert classifier.CLASS_NAMES[int(result["class_id"])] == result["prediction"]


def test_classification_confidence_is_between_0_and_1(classifier, crop):
    result = classifier.single_image_classification(str(crop))
    assert 0 <= result["confidence"] <= 1


def test_classification_all_confidences_add_up_to_1(classifier, crop):
    result = classifier.single_image_classification(str(crop))
    if "all_confidences" not in result:
        pytest.skip("Single-output classifier: no list of confidences")
    total = sum(confidence for _, confidence in result["all_confidences"])
    assert total == pytest.approx(1, abs=0.001)


def test_pipeline_classifies_every_detection(classifier, detect):
    results = [detect("coyote_day.jpg"), detect("animal_bird_day.jpg")]
    detection_count = sum(len(result["detections"]) for result in results)
    classifications = classifier.batch_image_classification(det_results=results)
    assert len(classifications) == detection_count


@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason="Known bug in 1.3.0 (F16): ImageFolder does not accept path_head",
)
def test_batch_classification_of_a_folder(classifier, crop):
    results = classifier.batch_image_classification(data_path=str(crop.parent))
    assert len(results) == 1


@pytest.mark.xfail(
    raises=RuntimeError,
    strict=True,
    reason="Known bug in 1.3.0 (F17): grayscale images are not converted to RGB",
)
def test_classification_of_a_grayscale_photo(classifier, crop, tmp_path):
    grayscale = tmp_path / "grayscale.jpg"
    Image.open(crop).convert("L").save(grayscale)
    result = classifier.single_image_classification(str(grayscale))
    assert result["prediction"] in classifier.CLASS_NAMES.values()
