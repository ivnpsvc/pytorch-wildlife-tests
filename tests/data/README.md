# Accuracy sample

`accuracy_sample.csv` lists 200 photos from the Caltech Camera Traps dataset on LILA BC:
100 photos with an animal and 100 empty photos. `tests/test_accuracy.py` downloads them at test
time into `~/.cache/pytorch-wildlife-tests/accuracy`; they are not stored in this repository.

- Dataset: https://lila.science/datasets/caltech-camera-traps
- License: Community Data License Agreement - Permissive, Version 1.0
  (https://cdla.dev/permissive-1-0/)
- Citation: Sara Beery, Grant Van Horn, Pietro Perona. Recognition in Terra Incognita. ECCV 2018.

## How the sample was chosen

- **Animal photos:** images with at least one bounding box for an animal, no other image-level label,
  at most two photos per camera location.
- **Empty photos:** images labeled only "empty", at most one photo per camera location.
- Photos already in `images/` were excluded.
- Random selection with a fixed seed (2026), so the sample is the same every time.

The sample covers 14 species (most often coyote, bird, rabbit and deer) from 109 camera locations;
96 of the 200 photos were taken between 19:00 and 06:00.

## Known label errors

"Empty" in this dataset means "no animal labeled", and labels are not perfect. Checked by eye,
at least two photos labeled empty clearly show an animal close to the camera:

- `59328ca6-23d2-11e8-a6a3-ec086b02610b.jpg`
- `59adfd1b-23d2-11e8-a6a3-ec086b02610b.jpg`

They stay in the sample so it is not changed after seeing the results. The measured false positive
rate is therefore an upper bound.
