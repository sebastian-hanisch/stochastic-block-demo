"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = q, n = 2, keine Kanten, Gleichstände, inklusive Grenzen."""

import numpy as np
import pytest

import sbm_algorithm as A
import sbm_constants as C
import sbm_evaluation as ev
from conftest import edges_of

EMPTY = (np.zeros(0, dtype=int), np.zeros(0, dtype=int))


def test_smallest_networks_sample_without_error():
    for q in (2, 3, 4):
        labels, u, v = A.sample_sbm(q, q, A.symmetric_cmat(q, 3.0, 1.0), 1)      # n = q: eine Gruppe je Knoten
        assert labels.tolist() == list(range(q)) and np.all(u < v)
    labels, u, v = A.sample_sbm(2, 2, A.symmetric_cmat(2, 100.0, 100.0), 1)       # n = 2: höchstens eine Kante
    assert len(u) <= 1


def test_networks_without_edges_have_defined_results():
    u, v = EMPTY
    assert A.mean_degree(5, u) == 0.0 and A.mean_degree(0, u) == 0.0
    em = A.em_parameters(5, u, v, 2, 1)
    assert em.marg.shape == (5, 2) and em.nvec.tolist() == [0.5, 0.5] and em.iterations == 0
    assert A.dense_adjacency(5, u, v).sum() == 0


def test_nmi_and_overlap_of_empty_inputs():
    assert A.nmi([], []) == 0.0 and A.overlap([], [], 2) == 0.0 and A.best_permutation_accuracy([], [], 2) == 0.0
    assert A.nmi([0, 1], [0]) == 0.0            # ungleiche Längen


def test_signal_limits_are_inclusive():
    """Die Grenzwerte selbst sind zulässig: c_out = 0 bei λ = √c, c_in = 0 bei λ = −√c / (q − 1)."""
    lo, hi = C.lam_limits(2, 6)
    assert A.affinities(2, 6, hi)[2] == pytest.approx(hi) and A.affinities(2, 6, lo)[2] == pytest.approx(lo)
    assert A.affinities(2, 6, hi + 1e-6)[2] == pytest.approx(hi)       # knapp darüber wird begrenzt


def test_best_permutation_with_ties_picks_the_maximum():
    """Zwei gleich gute Umbenennungen: das Maximum zählt, nicht die erste."""
    assert A.best_permutation_accuracy([0, 1, 0, 1], [0, 0, 1, 1], 2) == pytest.approx(0.5)
    assert A.best_permutation_accuracy([0, 0, 1], [1, 1, 0], 2) == pytest.approx(1.0)


def test_analysis_of_the_smallest_slider_corner_does_not_fail():
    for mode in C.MODES:
        a = ev.analyse(ev.Settings(C.Q_MAX, C.N_MIN, C.C_MIN, C.LAM_MAX, 1, mode))
        assert a.inst.n == C.N_MIN and 0.0 <= a.nmi_infer <= 1.0 and 0.0 <= a.nmi_leiden <= 1.0
        b = ev.analyse(ev.Settings(C.Q_MIN, C.N_MAX, C.C_MAX, C.LAM_MIN, 1, mode))
        assert b.inst.n == C.N_MAX


def test_experiments_skip_signals_outside_the_feasible_range():
    rows = ev.detect_curve(2, 100, 3, lams=(1.5, 2.4), seeds=(1,))
    assert [r["lam"] for r in rows] == [1.5]                          # λ = 2,4 > √3 = 1,73 entfällt
    assert ev.disassortative_rows(100, 3, lams=(-2.0, -1.0), seeds=(1,))[0]["lam"] == -1.0


def test_pure_noise_has_no_meaningful_groups_for_bp():
    cm = A.symmetric_cmat(3, 6.0, 6.0)
    labels, u, v = A.sample_sbm(150, 3, cm, 2)
    res = A.bp_marginals(150, u, v, 3, cm, np.full(3, 1 / 3), 2)
    assert np.abs(res.marg - 1 / 3).max() < 0.2           # bei gleichen Affinitäten bleibt jede Gruppe gleich wahrscheinlich
