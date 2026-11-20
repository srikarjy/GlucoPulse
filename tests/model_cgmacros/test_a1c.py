"""Pure helpers of model/cgmacros/a1c.py, on synthetic numbers only."""
import numpy as np
import pandas as pd
import pytest

from model.cgmacros import a1c as A


def test_gmi_matches_published_table():
    # Bergenstal 2018 Table 1: 150 mg/dL -> 6.9 %, 200 -> 8.1 %
    got = A.gmi(pd.DataFrame({"mean": [150.0, 200.0]}))
    assert np.round(got, 1).tolist() == [6.9, 8.1]


def test_ifcc_to_ngsp():
    assert A.mmol_mol_to_pct(48.0) == pytest.approx(6.54, abs=0.02)      # 48 mmol/mol is the 6.5 % cut point


def test_unit_guard_rejects_unconverted_values():
    with pytest.raises(ValueError):
        A.unit_guard(48.0, "x")


def test_cgm_features_need_five_days():
    assert A.cgm_features(np.full(288 * 4, 120.0), 5.0) is None
    f = A.cgm_features(np.r_[np.full(288 * 5, 100.0), np.full(288, 200.0)], 5.0)
    assert f["gt180"] == pytest.approx(1 / 6) and f["days"] == pytest.approx(6.0)


def test_hr_features_from_synthetic_days():
    ts = pd.date_range("2026-01-01", periods=288 * 4, freq="5min")
    hr = np.where(ts.hour < 6, 55.0, 75.0)
    f = A.hr_features(pd.Series(ts), pd.Series(hr))
    assert f["night"] == pytest.approx(55.0) and f["day_night"] == pytest.approx(20.0) and f["hr_days"] == 4
