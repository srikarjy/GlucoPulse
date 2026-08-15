"""
Patient-level train/val/test split, shared by every Phase 4 script (baseline,
training, evaluation) so the same held-out patients are used throughout --
comparing TFT to baseline on different patients would be meaningless.

Split is by patient, not by time within a patient (see docs/STATUS.md) --
generalization to unseen patients is the actual claim, not just unseen weeks
from patients the model has already partly seen.

Reproducible: all 25 AZT1D patient IDs, shuffled with a fixed seed, sliced
18/3/4. Not stratified by anything (data volume, device_mode usage, etc.) --
at N=25 that's not worth the added complexity; see docs/QUESTIONS.md.
"""

import random

ALL_PATIENTS = [str(i) for i in range(1, 26)]

_shuffled = ALL_PATIENTS.copy()
random.Random(42).shuffle(_shuffled)

TRAIN_PATIENTS = sorted(_shuffled[:18], key=int)
VAL_PATIENTS = sorted(_shuffled[18:21], key=int)
TEST_PATIENTS = sorted(_shuffled[21:25], key=int)

assert sorted(TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS, key=int) == ALL_PATIENTS
assert not (set(TRAIN_PATIENTS) & set(VAL_PATIENTS) & set(TEST_PATIENTS))
