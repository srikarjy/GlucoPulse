"""Participant-level split for CGMacros: the same held-out people for every variant,
the baseline and the triage eval. Split by person, not by time, so the claim is
generalization to people the model never saw."""
from __future__ import annotations

import os
import random

# CGMACROS_SPLIT=confirm selects the confirmation split: 9 new held-out people drawn from the primary TRAINING
# people (never seen by anything designed on the primary held-out 12), 5 of the primary held-out people for early
# stopping, and the remaining 31 for training. Unset (default) = the primary split, unchanged.
SCHEME = os.getenv("CGMACROS_SPLIT", "primary")
CONFIRM_SEED, CONFIRM_TEST, CONFIRM_VAL = 2026, 9, 5


def _primary(ids: list[str], seed: int, val_frac: float, test_frac: float):
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_frac))
    n_val = max(1, round(len(shuffled) * val_frac))
    test = sorted(shuffled[:n_test])
    val = sorted(shuffled[n_test:n_test + n_val])
    train = sorted(shuffled[n_test + n_val:])
    return train, val, test


def split_participants(ids: list[str], seed: int = 42, val_frac: float = 0.12,
                       test_frac: float = 0.16) -> tuple[list[str], list[str], list[str]]:
    train, val, test = _primary(ids, seed, val_frac, test_frac)
    if SCHEME == "primary":
        return train, val, test
    if SCHEME != "confirm":
        raise ValueError(f"unknown CGMACROS_SPLIT {SCHEME!r}")
    rng = random.Random(CONFIRM_SEED)
    pool, old_held = list(train), sorted(val + test)
    rng.shuffle(pool)
    rng.shuffle(old_held)
    new_test = sorted(pool[:CONFIRM_TEST])
    new_val = sorted(old_held[:CONFIRM_VAL])
    new_train = sorted(pool[CONFIRM_TEST:] + old_held[CONFIRM_VAL:])
    return new_train, new_val, new_test
