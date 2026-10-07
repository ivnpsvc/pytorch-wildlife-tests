# pytorch-wildlife-tests

[![Tests](https://github.com/ivnpsvc/pytorch-wildlife-tests/actions/workflows/tests.yml/badge.svg)](https://github.com/ivnpsvc/pytorch-wildlife-tests/actions/workflows/tests.yml)
[![Slow tests](https://github.com/ivnpsvc/pytorch-wildlife-tests/actions/workflows/slow-tests.yml/badge.svg)](https://github.com/ivnpsvc/pytorch-wildlife-tests/actions/workflows/slow-tests.yml)

A pytest test suite for [PyTorch-Wildlife](https://github.com/microsoft/Pytorch-Wildlife),
the open source library that runs the [MegaDetector](https://github.com/agentmorris/MegaDetector)
camera trap model.

The suite tests the library from the outside, the way a researcher uses it: install it, detect
animals in camera trap photos, process whole folders, and save, sort and crop the results. It uses
real camera trap photos from the Caltech Camera Traps dataset on [LILA BC](https://lila.science).

- **Version under test:** PytorchWildlife 1.3.0 (PyPI), also checked against `main` (1.3.1)
- **Tests:** 268 (189 fast tests on every push, 79 slow tests every night)
- **Findings:** 17 issues and 5 model limitations, documented in [FINDINGS.md](FINDINGS.md)

## What is tested

| Area | File |
|---|---|
| Detection runs at all (smoke test) | `tests/test_smoke.py` |
| Clear animal, person and empty photos give the expected answer | `tests/test_known_answers.py` |
| Several animals, borderline vehicle, known model limitations | `tests/test_hard_cases.py` |
| Missing, corrupt, tiny and non-image files | `tests/test_edge_cases.py` |
| Result structure on every photo (boxes inside the image, confidences, fields) | `tests/test_result_structure.py` |
| Batch detection of folders and arrays | `tests/test_batch.py` |
| Saving JSON and Timelapse JSON, sorting photos into folders | `tests/test_post_process.py` |
| Annotated images, dot images and crops | `tests/test_output_images.py` |
| File extensions, resizing (letterbox) and transforms | `tests/test_data_utils.py` |
| Other image formats, color modes, portrait photos, array input | `tests/test_input_variants.py` |
| Other MegaDetector versions and species classifiers (slow) | `tests/test_other_models.py` |
| Recall and false positive rate on 200 labeled photos (slow) | `tests/test_accuracy.py` |

The full scope, priorities and test design rules are in [TEST_PLAN.md](TEST_PLAN.md).

## Findings

Testing found bugs and open questions in the library, for example:

- **Single-image and batch detection give different results for the same photo**, because single
  detection swaps the red and blue color channels (F9).
- **Sorting photos into `Animal/` and `No_animal/` folders fails** with the library's default JSON
  output (F11).
- **The RT-DETR model version misses clear animals, people and vehicles**, because it runs at the
  wrong image size (F7).
- **Model weights are downloaded again every time** a model is created (F3).

Each finding has a category (bug, model limitation, or question), evidence, the cause in the source
code where known, and the test that documents it: [FINDINGS.md](FINDINGS.md).

Known bugs are marked with `@pytest.mark.xfail(strict=True)`. When a bug is fixed, its test
"unexpectedly passes" and the suite reports it. Run against `main`, this showed that two of the
findings (F4 and F16) are already fixed there.

## Running the tests

Requirements: Python 3.13 and Git. About 1.5 GB of disk space for PyTorch and model weights.

```bash
git clone https://github.com/ivnpsvc/pytorch-wildlife-tests.git
cd pytorch-wildlife-tests
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Linux without a GPU, add `--extra-index-url https://download.pytorch.org/whl/cpu` to the last
command to install the much smaller CPU version of PyTorch.

`requirements.txt` includes workarounds for two installation bugs in PytorchWildlife 1.3.0
(missing `soundfile` and `librosa`, and `setuptools<82`); see F1 and F2 in [FINDINGS.md](FINDINGS.md).

Then:

| Command | What it runs |
|---|---|
| `pytest` | The fast suite (slow tests are skipped by default) |
| `pytest -m slow` | Only the slow tests: other models (about 1.3 GB of downloads) and the accuracy check (about 80 MB of photos) |
| `pytest -rxX` | The fast suite, with a summary of every known bug and limitation (xfail) |
| `pytest --cov=PytorchWildlife --cov-report=html` | The fast suite with a coverage report in `htmlcov/index.html` |
| `ruff check .` and `ruff format --check .` | Lint and code style checks |

The first run downloads the default MegaDetector weights (about 50 MB). Because of F3, they are
downloaded again on every run unless the file in `~/.cache/torch/hub/checkpoints/` is renamed from
`MDV6-yolov9-c.pt` to `MDV6b-yolov9-c.pt`.

## Continuous integration

- **[Tests](.github/workflows/tests.yml):** on every push and pull request: style checks and the
  fast suite on Ubuntu.
- **[Slow tests](.github/workflows/slow-tests.yml):** every night and on demand: other models and
  the accuracy check. The accuracy report (recall, false positive rate, missed photos) is uploaded
  as a run artifact.

## Project structure

```
images/               20 camera trap photos used by the tests, with sources and license
tests/                the test suite
tests/conftest.py     shared fixtures: the detector, the images folder, cached detections
tests/constants.py    MegaDetector class IDs
tests/data/           the accuracy sample (photo list only; photos are downloaded at test time)
TEST_PLAN.md          scope, priorities and test design rules
FINDINGS.md           issues found, with evidence
```

## Test data

The photos come from the [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps)
dataset on LILA BC, published under the
[Community Data License Agreement, Permissive, Version 1.0](https://cdla.dev/permissive-1-0/).
Citation: Sara Beery, Grant Van Horn, Pietro Perona. Recognition in Terra Incognita. ECCV 2018.
See [images/README.md](images/README.md) and [tests/data/README.md](tests/data/README.md).

## License

The test code and documentation are released under the [MIT License](LICENSE). The photos keep
their original license (CDLA-Permissive-1.0, see "Test data" above).

## About

Built by [Ivan Posavec](https://github.com/ivnpsvc). Camera traps help researchers monitor
wildlife at a scale no person could, and the software behind them should be reliable. This project
tests that software and reports what it finds. Not affiliated with Microsoft or the
PyTorch-Wildlife team.
