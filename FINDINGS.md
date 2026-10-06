# Findings

Issues found while testing PytorchWildlife 1.3.0 (from PyPI) on Python 3.13,
macOS (Apple M4) and Ubuntu (GitHub Actions).

Categories:

- **Bug:** the code is wrong and can be fixed in code
- **Model limitation:** the code works, but the model's answer is wrong; needs retraining, not a code fix
- **Question:** behavior that may be intentional; needs the maintainers' view

Status: **Reported** (an issue exists), **Not reported**, or **To verify** (suspected, not yet confirmed by a test).

## Summary

| ID | Finding | Category | Status |
|---|---|---|---|
| F1 | Fresh install cannot be imported: `soundfile` and `librosa` not declared | Bug | Reported ([#26](https://github.com/microsoft/Pytorch-Wildlife/issues/26)) |
| F2 | `yolov5` dependency imports `pkg_resources`, removed in setuptools 82 | Bug (dependency) | Reported ([#26](https://github.com/microsoft/Pytorch-Wildlife/issues/26)) |
| F3 | Model weights are downloaded again on every model creation | Bug | Not reported |
| F4 | `MegaDetectorV6()` with the default `version` raises `ValueError` | Bug | Not reported |
| F5 | `single_image_detection` crashes on a `pathlib.Path` | Bug | Not reported |
| F6 | `device` argument is ignored; detection always runs on CPU | Bug | To verify |
| F7 | RT-DETR version may use the wrong predictor | Bug | To verify |
| F8 | Truncated photos are silently processed; the animal is lost | Question | Not reported |
| L1–L4 | Missed or false detections on hard photos | Model limitation | Not reported |

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

- **Category:** Bug. **Status:** Not reported.
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

## F7: RT-DETR version may use the wrong predictor

- **Category:** Bug. **Status:** To verify.
- **Observation:** `yolov8_base.py` selects the RT-DETR predictor only when
  `self.MODEL_NAME == 'MDV6b-rtdetrl.pt'`, but `MDV6-rtdetr-c` sets `MODEL_NAME = "MDV6b-rtdetr-c.pt"`.
  The YOLO predictor is used instead.
- **To verify:** run `MDV6-rtdetr-c` on the known-answer photos (planned in P3, other models).

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

## L1–L4: Model limitations

The code works in these cases; the model's answer is wrong. Possibly useful as examples for future
training. All at threshold 0.2 with `MDV6-yolov9-c`. Tests: `test_hard_cases.py` (xfail, strict).

| ID | Photo | Expected | Actual |
|---|---|---|---|
| L1 | `animal_upside_down_camera.jpg` (camera mounted upside down) | Animal (bird) | No detections |
| L2 | `animal_too_close_blurry.jpg` (animal fills the frame) | Animal | Best confidence 0.14, below threshold |
| L3 | `person_and_dog_day.jpg` (only a person's legs visible) | Person and animal | Animal only |
| L4 | `empty_branch_across_lens.jpg` (branch and rock, no animal) | Nothing | Animal, confidence 0.65 |

Related observation: `vehicle_car_day.jpg` is detected as a vehicle with only 0.22 confidence,
just above the 0.2 threshold. The test uses threshold 0.1 to avoid a flaky result.
