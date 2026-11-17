"""Hybrid meal what-if (model/cgmacros/hybrid.py): the properties the design guarantees.

More carbohydrate never forecasts lower glucose; a meal never forecasts below no meal; the effect curve is the
Hovorka gut-absorption shape (zero at the meal, rising through the hour, all absorbed in the long run).
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from model.cgmacros.hybrid import (  # noqa: E402
    HORIZON, K_GRID, effect_curve, fit_gain, hybrid_forecast, meal_basis,
)


def test_effect_curve_shape():
    e = effect_curve(0.0, minutes=2000)
    assert e[0] == 0.0
    assert np.all(np.diff(e[1:61]) > 0)                # still rising during the first hour with no clearance
    assert abs(e[-1] - 1.0) < 1e-2                       # all of the meal absorbed eventually
    assert effect_curve(0.04)[60] < effect_curve(0.0)[60]


@pytest.mark.parametrize("k", K_GRID)
def test_more_carbs_never_forecasts_lower_and_meal_never_below_no_meal(k):
    rng = np.random.default_rng(1)
    n = 200
    no_meal = 120 + rng.normal(0, 20, (n, HORIZON))
    X = rng.normal(0, 1, (n, 7))
    params = rng.normal(0, 1, 8)
    zero, c30, c60 = (np.zeros((n, HORIZON)) for _ in range(3))
    c30[:, 0], c60[:, 0] = 30.0, 60.0
    f0, f30, f60 = (hybrid_forecast(no_meal, X, c, params, k) for c in (zero, c30, c60))
    assert np.allclose(f0, no_meal)
    assert np.all(f30 >= f0) and np.all(f60 >= f30)
    assert np.all(f60[:, -1] > f30[:, -1])               # strictly larger by the end of the hour
    personal = hybrid_forecast(no_meal, X, c60, params, k, scale=rng.uniform(0, 3, n))
    assert np.all(personal >= no_meal)


def test_meal_later_in_the_hour_has_no_effect_before_it():
    c = np.zeros((1, HORIZON))
    c[0, 6] = 50.0
    b = meal_basis(c, 0.01)[0]
    assert np.all(b[:7] == 0) and np.all(b[7:] > 0)


def test_fit_recovers_a_known_gain():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(0, 1, (n, 2))
    carbs = np.zeros((n, HORIZON))
    carbs[:, 0] = rng.uniform(10, 90, n)
    basis = meal_basis(carbs, 0.01)
    true = np.array([np.log(1.5), 0.3, 0.0])
    resid = np.exp(true[0] + X @ true[1:])[:, None] * basis + rng.normal(0, 1, (n, HORIZON))
    p, _ = fit_gain(X, resid, basis)
    assert abs(p[0] - true[0]) < 0.1 and abs(p[1] - 0.3) < 0.1
