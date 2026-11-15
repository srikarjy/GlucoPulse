"""Participant-level split for CGMacros: the same held-out people for every variant,
the baseline and the triage eval. Split by person, not by time, so the claim is
generalization to people the model never saw."""
from __future__ import annotations

import random


def split_participants(ids: list[str], seed: int = 42, val_frac: float = 0.12,
                       test_frac: float = 0.16) -> tuple[list[str], list[str], list[str]]:
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_frac))
    n_val = max(1, round(len(shuffled) * val_frac))
    test = sorted(shuffled[:n_test])
    val = sorted(shuffled[n_test:n_test + n_val])
    train = sorted(shuffled[n_test + n_val:])
    return train, val, test
