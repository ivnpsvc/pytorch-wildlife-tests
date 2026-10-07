# Findings

Issues found while testing PytorchWildlife 1.3.0 (from PyPI) on Python 3.13,
macOS (Apple M4) and Ubuntu (GitHub Actions).

Categories:

- **Bug:** the code is wrong and can be fixed in code
- **Model limitation:** the code works, but the model's answer is wrong; needs retraining, not a code fix
- **Question:** behavior that may be intentional; needs the maintainers' view

Status: **Reported** (an issue exists), **Fix proposed** (a pull request exists), **Not reported**, or **To verify** (suspected, not yet confirmed by a test).

**On `main`:** result of running this suite against the `main` branch of microsoft/Pytorch-Wildlife
(version 1.3.1, commit `55c5135`, checked 2026-10-07). "Fixed" means the strict xfail test for the
finding unexpectedly passed there. "Still present" means the test result is unchanged; for F1 and F2,
the dependency list in `pyproject.toml` is unchanged.

## Summary

| ID | Finding | Category | Status | On `main` |
|---|---|---|---|---|
| F1 | Fresh install cannot be imported: `soundfile` and `librosa` not declared | Bug | Reported ([#26](https://github.com/microsoft/Pytorch-Wildlife/issues/26)) | Still present |
| F2 | `yolov5` dependency imports `pkg_resources`, removed in setuptools 82 | Bug (dependency) | Reported ([#26](https://github.com/microsoft/Pytorch-Wildlife/issues/26)) | Still present |
| F3 | Model weights are downloaded again on every model creation | Bug | Fix proposed in [PR #23](https://github.com/microsoft/Pytorch-Wildlife/pull/23) (open) | Still present |
| F4 | `MegaDetectorV6()` with the default `version` raises `ValueError` | Bug | Not reported | **Fixed** |
| F5 | `single_image_detection` crashes on a `pathlib.Path` | Bug | Not reported | Still present |
| F6 | `device` argument is ignored; detection always runs on CPU | Bug | To verify | Still present |
| F7 | RT-DETR version runs at the wrong image size with the wrong predictor; it misses clear animals, people and vehicles | Bug | Not reported | Still present |
| F8 | Truncated photos are silently processed; the animal is lost | Question | Not reported | Still present |
| F9 | Single-image detection passes RGB where Ultralytics expects BGR; results differ from batch detection | Bug | Not reported | Still present |
| F10 | One non-image `.jpg` in a folder stops the whole batch | Question | Not reported | Still present |
| F11 | `detection_folder_separation` fails with `SameFileError` on JSON with absolute paths (the default) | Bug | Not reported | Still present |
| F12 | Timelapse JSON writes `max_detection_conf` as an empty string for photos without detections | Question | Not reported | Still present |
| F13 | Folder separation drops animals with confidence exactly at the threshold; detection keeps them | Question | Not reported | Still present |
| F14 | `overwrite=True` empties the whole output folder, including files the library did not create | Question | Not reported | Still present |
| F15 | Array input without `img_path` gets the text `"None"` as `img_id` | Question | Not reported | Still present |
| F16 | `batch_image_classification(data_path=...)` always raises `TypeError` | Bug | Not reported | **Fixed** |
| F17 | Classifiers crash on grayscale images | Bug | Not reported | Still present |
| L1–L5 | Missed or false detections on hard photos | Model limitation | Not reported | Not checked |

## F1: Fresh install cannot be imported

- **Category:** Bug. **Status:** Reported in [#26](https://github.com/microsoft/Pytorch-Wildlife/issues/26).
- **What happens:** after `pip install PytorchWildlife`, `import PytorchWildlife` fails with
  `ModuleNotFoundError: No module named 'soundfile'`.
- **Cause:** `PytorchWildlife/__init__.py` imports `data`, which imports `data/bioacoustics` at
  import time. Those modules import `soundfile` and `librosa`, which are not declared as dependencies.
  This also affects users who only need detection.
- **Workaround:** `pip install soundfile librosa` (in `requirements.txt`).

## F2: `pkg_resources` removed from setuptools

- **Category:** Bug in a dependency. **Status:** Reported in [#26](https://github.com/microsoft/Pytorch-Wildlife/issues/26).
- **What happens:** after F1's workaround, import fails with
  `ModuleNotFoundError: No module named 'pkg_resources'` from `yolov5/utils/general.py`.
- **Cause:** `pkg_resources` was removed in setuptools 82; `setuptools` is unconstrained.
- **Workaround:** `setuptools<82` (in `requirements.txt`). A deprecation warning remains; filtered in `pytest.ini`.

## F3: Model weights are downloaded again on every model creation

- **Category:** Bug. **Status:** fix proposed by another contributor in
  [PR #23](https://github.com/microsoft/Pytorch-Wildlife/pull/23) ("Fix mdv6 model name", open since
  2026-08-26, not yet reviewed). It changes both `MODEL_NAME` values to match the downloaded files.
- **What happens:** every `MegaDetectorV6(version="MDV6-yolov9-c")` downloads the 52 MB weights again,
  and the cache folder fills with copies (`MDV6-yolov9-c.pt`, `MDV6-yolov9-c (1).pt`, ...).
- **Cause:** `yolov8_base.py` checks the cache for `self.MODEL_NAME`, but the downloaded file is named
  after the URL. In `megadetectorv6.py`:

  ```python
  url = "https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1"
  self.MODEL_NAME = "MDV6b-yolov9-c.pt"  # extra "b": never matches the download
  ```

  The same mismatch exists for `MDV6-rtdetr-c` (`MDV6-rtdetr-c.pt` vs `MDV6b-rtdetr-c.pt`).
  The other V6 versions have matching names.
- **Evidence:** two files in `~/.cache/torch/hub/checkpoints` after two runs. Renaming the file to
  `MDV6b-yolov9-c.pt` stops the downloads (verified locally and in CI).
- **Impact:** in CI, test time dropped from 216 s to 21 s after downloading the weights once under the
  expected name. Repeated downloads from Zenodo also risk rate limiting.
- **Workaround:** the "Download model weights" step in `.github/workflows/tests.yml`.
- **Test:** none yet (CI workaround hides it).

## F4: Default `version` is invalid

- **Category:** Bug. **Status:** Not reported.
- **What happens:** `MegaDetectorV6()` raises
  `ValueError: Select a valid model version: MDV6-yolov9-c, MDV6-yolov9-e, ...`.
- **Cause:** the default is `version='yolov9c'`, which is not in the list of accepted versions.
- **On `main`:** fixed; the default is `'MDV6-yolov9-c'`.
- **Test:** `test_model_can_be_created_with_default_version` (xfail, strict).

## F5: `single_image_detection` crashes on a `pathlib.Path`

- **Category:** Bug. **Status:** Not reported.
- **What happens:** passing a `Path` instead of a `str` raises
  `AttributeError: 'PosixPath' object has no attribute 'shape'`.
- **Cause:** `if type(img) == str:` treats anything that is not exactly a `str` as image data.
  `pathlib.Path` is the standard way to handle paths in modern Python.
- **Workaround:** `str(path)`.
- **Test:** none yet.

## F6: `device` argument is ignored

- **Category:** Bug. **Status:** To verify.
- **Observation:** in `yolov8_base.py`, the line that would apply the device is commented out:

  ```python
  # self.predictor.args.device = device # Will uncomment later
  ```

  Ultralytics reports `CPU (Apple M4)` even though MPS is available.
- **To verify:** create the model with `device="mps"` and check which device is used.

## F7: RT-DETR version is effectively broken

- **Category:** Bug. **Status:** Not reported.
- **What happens:** `MegaDetectorV6(version="MDV6-rtdetr-c")` loads, but misses clear photos at
  threshold 0.2: no detection on `coyote_day.jpg`, `person_day.jpg`, `vehicle_car_day.jpg` or
  `animal_mountain_lion_night.jpg`, and many low-confidence boxes elsewhere.
- **Causes (two):**
  1. `megadetectorv6.py` sets `IMAGE_SIZE = 1280` for every V6 version, but the RT-DETR checkpoint
     was trained at 640 px (`train_args["imgsz"] == 640` in the weights file).
  2. `yolov8_base.py` selects the RT-DETR predictor only when
     `self.MODEL_NAME == 'MDV6b-rtdetrl.pt'`, but this version sets `MODEL_NAME = "MDV6b-rtdetr-c.pt"`,
     so the YOLO predictor is used instead.
- **Evidence:** the same weights loaded directly with Ultralytics (`RTDETR(weights)`) at 640 px find
  the coyote (0.98), the person (0.95) and the car (0.96). At 1280 px, Ultralytics gives the same
  poor results as PytorchWildlife.
- **Also:** the weights are downloaded again on every model creation (same name mismatch as F3).
- **Test:** `test_rtdetr_detects_a_clear_animal` in `test_other_models.py` (xfail, strict, slow).

## F8: Truncated photos are silently processed

- **Category:** Question. **Status:** Not reported.
- **What happens:** a photo cut off after its first 20 KB (simulating a failed copy or memory card)
  is processed without an error and returns no detections. The coyote in it is lost, and the photo
  looks empty.
- **Cause:** `data/datasets.py` line 13 sets `ImageFile.LOAD_TRUNCATED_IMAGES = True`, so Pillow fills
  the missing part with gray instead of raising `OSError: image file is truncated`.
- **Side effect:** the setting is global, so importing PytorchWildlife changes Pillow's behavior for
  the whole program.
- **Question for maintainers:** is this intended (to keep batch runs going)? If so, could truncated
  files at least be reported?
- **Test:** `test_truncated_photo_is_processed_without_error` (characterization test).

## F9: Single-image detection uses swapped color channels

- **Category:** Bug. **Status:** Not reported.
- **What happens:** the same photo gives different results with `single_image_detection` and
  `batch_image_detection`:

  | Photo | Single (path) | Batch (folder) |
  |---|---|---|
  | `coyote_day.jpg` | animal 0.594, 0.418 | animal 0.504, 0.422 |
  | `animal_bird_day.jpg` | animal 0.864 | animal 0.880 |
  | `person_day.jpg` | person 0.906 | person 0.912 |
  | `animal_small_distant_day.jpg` | animal 0.404 | **no detection** |
  | `deer_night.jpg` (grayscale) | animal 0.703 | animal 0.703 |

- **Cause:** `single_image_detection` loads the image with
  `np.array(Image.open(img_path).convert("RGB"))` and passes the array to Ultralytics, which treats
  NumPy arrays as BGR (OpenCV convention). The red and blue channels are swapped. Batch detection
  from a folder passes file paths, which Ultralytics loads itself in the correct order.
- **Evidence:** single detection on a path gives the same result as an RGB array; batch detection
  gives the same result as a BGR array (`cv2.imread`). Grayscale night photos are identical in both,
  because their three channels are equal.
- **Also affected:** `batch_image_detection` with a list of arrays. Its docstring says
  "RGB format", but the arrays are treated as BGR.
- **Impact:** single-image results are computed on color-swapped photos. A researcher testing one
  photo and then running a folder can get different answers for the same photo.
- **Measured effect on accuracy** (200 labeled photos, `test_accuracy.py`, MDV6-yolov9-c, threshold
  0.2): batch detection (correct colors) recall 0.93 and false positive rate 0.15; single detection
  (swapped colors) recall 0.92 and false positive rate 0.13. On this sample the overall effect is
  small; individual photos still change. Larger V6 versions showed a bigger effect on empty photos
  (see L5).
- **Note for this suite:** known-answer tests use single detection, so they run on swapped colors.
  `animal_small_distant_day.jpg` is only found with swapped colors.
- **Test:** planned in `test_batch.py` (xfail, strict).

## F10: One non-image `.jpg` stops the whole batch

- **Category:** Question. **Status:** Not reported.
- **What happens:** a folder with valid photos and one text file named `fake.jpg` raises
  `UnidentifiedImageError`, and no results are returned for any photo.
- **Contrast:** truncated photos (F8) are processed silently. One kind of broken file stops the
  batch, the other is hidden.
- **Question for maintainers:** should a batch skip unreadable files and report them, rather than
  fail entirely?
- **Test:** planned in `test_batch.py` (characterization test).

## F11: Folder separation fails on absolute paths

- **Category:** Bug. **Status:** Not reported.
- **What happens:** `save_detection_json` writes absolute paths by default (`img_id` is the full path).
  Passing that JSON to `detection_folder_separation` raises `shutil.SameFileError` on the first photo,
  and no photos are sorted.
- **Cause:** the destination is built as `os.path.join(target_folder, os.path.dirname(img_id))`.
  When `img_id` is absolute, `os.path.join` discards `target_folder` and returns the photo's own
  folder, so the photo is copied onto itself.
- **Works when:** the JSON is saved with `exclude_file_path=<photo folder>`, so `img_id` is relative.
- **Impact:** the default pipeline (detect a folder, save JSON, sort photos) fails at the last step.
- **Test:** planned in `test_post_process.py` (xfail, strict).

## F12: Empty string for `max_detection_conf` in Timelapse JSON

- **Category:** Question. **Status:** Not reported.
- **What happens:** in `save_detection_timelapse_json`, photos without detections get
  `"max_detection_conf": ""`, while other photos get a number.
- **Cause:** `float(max(confidence_list)) if len(confidence_list) > 0 else ''`.
- **Question:** tools reading the file may expect a number (for example `0.0`). Is `""` what
  Timelapse expects?
- **Related observation:** the default `info` is `{"detector": "megadetector_v5"}`, also when the
  results come from MegaDetector V6.

## F13: Threshold boundary differs between detection and folder separation

- **Category:** Question. **Status:** Not reported.
- **What happens:** detection keeps detections with confidence **at or above** the threshold
  (`det_conf_thres`), but `detection_folder_separation` sorts a photo into `Animal/` only when the
  confidence is **above** the threshold (`confidence > confidence_threshold`). An animal with
  confidence exactly 0.2 is detected at threshold 0.2, but sorted into `No_animal/` at threshold 0.2.
- **Impact:** small in practice (an exact match is rare), but the same number means two different
  things in two steps of one pipeline.
- **Test:** planned in `test_post_process.py` (characterization test).

## F14: `overwrite=True` empties the whole output folder

- **Category:** Question. **Status:** Not reported.
- **What happens:** `save_detection_images`, `save_detection_images_dots` and `save_crop_images`
  with `overwrite=True` delete everything in the output folder before saving, including files the
  library did not create. In a test, a `notes.txt` in the output folder was deleted.
- **Cause:** the functions use `supervision.ImageSink(target_dir_path=output_dir, overwrite=overwrite)`,
  which removes the folder's contents when `overwrite=True`.
- **Risk:** the docstring says "Whether overwriting existing image folders". A user who points
  `output_dir` at a folder with other data, or at the photo folder itself, can lose files.
- **Question for maintainers:** is deleting unrelated files intended? A warning in the docstring,
  or overwriting only the files being written, would be safer.
- **Test:** `test_overwrite_true_deletes_everything_in_the_output_folder` (characterization test).

## F15: Array input without `img_path` gets `"None"` as `img_id`

- **Category:** Question. **Status:** Not reported.
- **What happens:** `single_image_detection(array)` without `img_path` returns
  `img_id == "None"`: the word None as text, not Python's `None` value.
- **Cause:** `results_generation` builds the ID with `str(img_id).strip(id_strip)`, and
  `str(None)` is `"None"`.
- **Impact:** small. When several array results are saved to JSON, they all get the same ID
  `"None"`, and later tools cannot tell the photos apart. Batch detection numbers arrays
  `"0"`, `"1"`, ... instead.
- **Test:** `test_array_input_without_img_path_gets_the_text_none_as_img_id`
  (characterization test).

## F16: Batch classification of a folder always fails

- **Category:** Bug. **Status:** Not reported.
- **What happens:** `batch_image_classification(data_path=folder)` raises
  `TypeError: ImageFolder.__init__() got an unexpected keyword argument 'path_head'`, for every
  ResNet classifier (Serengeti, Amazon, Opossum).
- **Cause:** `resnet_base/base_classifier.py` creates `pw_data.ImageFolder(data_path,
  transform=..., path_head='.')`, but `ImageFolder.__init__` only accepts `image_dir` and `transform`.
- **Works:** `batch_image_classification(det_results=...)` (classifying detection crops).
- **On `main`:** fixed; it uses `ClassificationImageFolder` without `path_head`.
- **Test:** `test_batch_classification_of_a_folder` (xfail, strict, slow).

## F17: Classifiers crash on grayscale images

- **Category:** Bug. **Status:** Not reported.
- **What happens:** `single_image_classification` on a grayscale JPEG raises
  `RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224]`.
- **Cause:** the image is opened with `Image.open(img)` but not converted with `.convert("RGB")`,
  unlike in detection. The normalization step expects three color channels.
- **Impact:** some cameras save night photos as true grayscale files.
- **Test:** `test_classification_of_a_grayscale_photo` (xfail, strict, slow).

## L1–L5: Model limitations

The code works in these cases; the model's answer is wrong. Possibly useful as examples for future
training. All at threshold 0.2 with `MDV6-yolov9-c`. Tests: `test_hard_cases.py` (xfail, strict).

| ID | Photo | Expected | Actual |
|---|---|---|---|---|
| L1 | `animal_upside_down_camera.jpg` (camera mounted upside down) | Animal (bird) | No detections |
| L2 | `animal_too_close_blurry.jpg` (animal fills the frame) | Animal | Best confidence 0.14, below threshold |
| L3 | `person_and_dog_day.jpg` (only a person's legs visible) | Person and animal | Animal only |
| L4 | `empty_branch_across_lens.jpg` (branch and rock, no animal) | Nothing | Animal, confidence 0.65 |

| L5 | `empty_hillside_day.jpg` (no animal or vehicle) with the larger V6 versions | Nothing | Vehicle: 0.64 with `MDV6-yolov10-e`, 0.40 with `MDV6-yolov9-e` |

L5 also appears with correct colors (BGR input), so it is not caused by F9. Other false detections
of the larger V6 versions on empty photos disappear with correct colors, which suggests F9 affects
accuracy, not only consistency. MegaDetector V5 finds nothing in any of the empty photos.

Related observation: `vehicle_car_day.jpg` is detected as a vehicle with only 0.22 confidence,
just above the 0.2 threshold. The test uses threshold 0.1 to avoid a flaky result.
