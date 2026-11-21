"""The reconstructed kinetic model must reproduce the published worked numbers (Xu et al., DOM 2022, doi:10.1111/dom.14763)."""
import pytest

from model.cgmacros import a1c_kinetic as K


@pytest.mark.parametrize("agr,expected", [(65.1, 7.0), (70.0, 7.5), (80.0, 8.5)])
def test_published_anchors_at_154(agr, expected):
    assert K.steady_a1c(154, agr) == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("target,agr,expected", [(6.0, 60, 5.5), (6.5, 60, 6.0), (7.0, 60, 6.5), (7.0, 75, 8.0),
                                                 (8.0, 80, 9.7), (6.0, 80, 7.2)])
def test_table2_rows(target, agr, expected):
    # Table 2: A1c at the same glucose under another AGR (targets set at reference AGR 65.1); paper rounds to 0.1
    assert K.steady_a1c(K.glucose_for_a1c(target), agr) == pytest.approx(expected, abs=0.1)


def test_inverse_round_trip():
    for g in (90.0, 154.0, 300.0):
        assert K.glucose_for_a1c(K.steady_a1c(g)) == pytest.approx(g, rel=1e-6)


def test_simulation_settles_at_steady_state_and_lags():
    start, end = float(K.steady_a1c(260)), float(K.steady_a1c(140))
    path = K.simulate([140.0] * 1000, start)      # time constant ~100 days
    assert path[-1] == pytest.approx(end, abs=0.02)
    assert path[9] > start - 0.5 * (start - end)          # 10 days after the drop: far less than halfway down
