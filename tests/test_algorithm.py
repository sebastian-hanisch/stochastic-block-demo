"""Kern gegen Handrechnung auf der Mini-Instanz (DEMO-PLAYBOOK §1b): je Einheit eine Aussage, von Hand gerechnet."""

import math

import numpy as np
import pytest

import sbm_algorithm as A
import sbm_constants as C


def test_affinities_by_hand():
    """q = 2, c = 6, λ = 1: Δ = λ q √c = 2 √6 = 4,899; c_in = 6 + Δ/2 = 8,449, c_out = 6 − Δ/2 = 3,551; mittlerer Grad (c_in + c_out) / 2 = 6."""
    cin, cout, lam = A.affinities(2, 6, 1.0)
    assert (cin, cout, lam) == pytest.approx((8.449, 3.551, 1.0), abs=1e-3)
    assert (cin + cout) / 2 == pytest.approx(6.0)
    assert A.signal_strength(cin, cout, 2, 6) == pytest.approx(1.0)


def test_affinities_keep_the_mean_degree_for_q_3_and_negative_signal():
    cin, cout, lam = A.affinities(3, 6, -1.0)
    assert lam == pytest.approx(-1.0) and cout > cin
    assert (cin + 2 * cout) / 3 == pytest.approx(6.0)


def test_affinities_clip_the_signal_to_the_feasible_range():
    """c_out >= 0 gibt λ <= √c; c_in >= 0 gibt λ >= −√c / (q − 1)."""
    cin, cout, lam = A.affinities(2, 6, 3.0)
    assert lam == pytest.approx(math.sqrt(6)) and cout == pytest.approx(0.0, abs=1e-9)
    cin, cout, lam = A.affinities(3, 6, -3.0)
    assert lam == pytest.approx(-math.sqrt(6) / 2) and cin == pytest.approx(0.0, abs=1e-9)
    assert C.lam_limits(2, 6) == pytest.approx((-math.sqrt(6), math.sqrt(6)))


def test_group_labels_split_equally_with_the_rest_in_the_last_group():
    assert A.group_labels(10, 3).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    assert A.group_labels(4, 2).tolist() == [0, 0, 1, 1]


def test_sampler_shapes_order_and_determinism():
    labels, u, v = A.sample_sbm(60, 2, A.symmetric_cmat(2, 8.0, 2.0), 5)
    assert len(labels) == 60 and np.all(u < v) and len(u) == len(v) > 0
    labels2, u2, v2 = A.sample_sbm(60, 2, A.symmetric_cmat(2, 8.0, 2.0), 5)
    assert np.array_equal(u, u2) and np.array_equal(v, v2)
    assert not np.array_equal(u, A.sample_sbm(60, 2, A.symmetric_cmat(2, 8.0, 2.0), 6)[1]) or len(u) != len(A.sample_sbm(60, 2, A.symmetric_cmat(2, 8.0, 2.0), 6)[1])


def test_dense_adjacency_is_symmetric_with_zero_diagonal(two_triangles):
    n, u, v = two_triangles
    a = A.dense_adjacency(n, u, v)
    assert np.array_equal(a, a.T) and a.trace() == 0 and a.sum() == 12
    assert A.mean_degree(n, u) == pytest.approx(2.0)


# --- Überlappungsmaße -------------------------------------------------------------------------------------------------------------------------------


def test_nmi_by_hand():
    """Gleiche Einteilung: 1; unabhängige (0,0,1,1) gegen (0,1,0,1): 0; (0,0,1,1) gegen (0,0,0,1): MI 0,21576 / Mittel der Entropien (0,6931 + 0,5623) / 2 = 0,3437."""
    assert A.nmi([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert A.nmi([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0, abs=1e-12)
    assert A.nmi([0, 0, 1, 1], [0, 0, 0, 1]) == pytest.approx(0.3437, abs=1e-4)


def test_nmi_trivial_partitions():
    assert A.nmi([0, 0, 0], [5, 5, 5]) == 1.0           # beide trivial
    assert A.nmi([0, 0, 1, 1], [0, 0, 0, 0]) == pytest.approx(0.0)   # nur eine Seite trivial


def test_best_permutation_accuracy_and_overlap():
    assert A.best_permutation_accuracy([0, 0, 1, 1], [1, 1, 0, 0], 2) == 1.0
    assert A.best_permutation_accuracy([0, 0, 1, 1], [0, 1, 0, 1], 2) == pytest.approx(0.5)
    assert A.overlap([0, 0, 1, 1], [0, 0, 1, 0], 2) == pytest.approx(0.5)             # Genauigkeit 0,75, größte Gruppe 0,5
    assert A.overlap([0, 0, 0, 1], [0, 0, 0, 0], 2) == pytest.approx(0.0)             # alles in die größte Gruppe: Raten
    assert A.overlap([0, 0, 0, 0], [0, 0, 0, 0], 2) == 0.0                            # nur eine geplante Gruppe


# --- Belief Propagation --------------------------------------------------------------------------------------------------------------------------


def test_bp_on_one_edge_by_hand(one_edge):
    """Gruppenanteile 0,7 / 0,3, c_aa = 3, c_ab = 1: Gewichte (0,0) 0,49·3 = 1,47, (0,1) 0,21, (1,0) 0,21, (1,1) 0,09·3 = 0,27, Summe 2,16;
    Randverteilung beider Knoten P(Gruppe 0) = (1,47 + 0,21) / 2,16 = 7/9."""
    n, u, v = one_edge
    cm = np.array([[3.0, 1.0], [1.0, 3.0]])
    res = A.bp_marginals(n, u, v, 2, cm, np.array([0.7, 0.3]), 1, iters=50, damping=0.0, use_field=False)
    assert res.marg[:, 0] == pytest.approx([7 / 9, 7 / 9], abs=1e-9)
    assert res.marg.sum(1) == pytest.approx([1.0, 1.0])
    assert res.converged


def test_bp_without_edges_returns_the_prior():
    res = A.bp_marginals(5, np.zeros(0, dtype=int), np.zeros(0, dtype=int), 2, np.eye(2), np.array([0.6, 0.4]), 1)
    assert res.iterations == 0 and res.converged and res.marg.shape == (5, 2) and res.marg[0].tolist() == [0.6, 0.4]


def test_bp_field_changes_the_marginals():
    """Zweig-Test: das Außenfeld (Beitrag der Nicht-Kanten) ändert die Randverteilungen auf einem Netz mit ungleichen Gruppen."""
    labels, u, v = A.sample_sbm(120, 2, A.symmetric_cmat(2, 8.0, 2.0), 4)
    cm = A.symmetric_cmat(2, 8.0, 2.0)
    with_field = A.bp_marginals(120, u, v, 2, cm, np.array([0.6, 0.4]), 4, use_field=True)
    without = A.bp_marginals(120, u, v, 2, cm, np.array([0.6, 0.4]), 4, use_field=False)
    assert np.abs(with_field.marg - without.marg).max() > 1e-3


def test_bp_finds_strong_groups_and_nothing_in_a_random_net():
    cm = A.symmetric_cmat(2, 10.9, 1.1)
    labels, u, v = A.sample_sbm(400, 2, cm, 3)
    res = A.bp_marginals(400, u, v, 2, cm, np.full(2, 0.5), 3)
    assert A.nmi(labels, res.labels) > 0.7
    cm0 = A.symmetric_cmat(2, 6.0, 6.0)
    labels0, u0, v0 = A.sample_sbm(400, 2, cm0, 3)
    res0 = A.bp_marginals(400, u0, v0, 2, cm0, np.full(2, 0.5), 3)
    assert A.nmi(labels0, res0.labels) < 0.1


# --- EM ---------------------------------------------------------------------------------------------------------------------------------------------


def test_learned_contrast_by_hand():
    """c_aa = 10, c_ab = 2, c = 6, q = 2: |10 − 2| / (2 √6) = 1,633."""
    assert A.learned_contrast(np.array([[10.0, 2.0], [2.0, 10.0]]), 6.0) == pytest.approx(1.633, abs=1e-3)


def test_em_start_is_assortative_and_above_the_threshold():
    rng = np.random.default_rng(1)
    cm = A.em_start(2, 6.0, 0.0, rng)
    assert cm[0, 0] > cm[0, 1] and A.signal_strength(cm[0, 0], cm[0, 1], 2, 6.0) == pytest.approx(C.EM_START_LAMBDA, abs=0.05)


def test_em_update_on_one_edge_with_equal_messages():
    """Eine Kante, gleich verteilte Nachrichten (0,5 / 0,5) und c_aa = c_ab: die Kante liegt zu 1/4 in jedem der vier Paare; erwartete Kantenzahlen E_aa = 0,25, E_ab = 0,5."""
    psi = np.full((2, 2), 0.5)
    e = A.em_update(2, np.array([0]), np.array([1]), 2, psi, np.ones((2, 2)))
    assert e[0, 0] == pytest.approx(0.25) and e[1, 1] == pytest.approx(0.25) and e[0, 1] == pytest.approx(0.5) and e[1, 0] == pytest.approx(0.5)


# --- Modularität und Leiden --------------------------------------------------------------------------------------------------------------------------


def test_modularity_of_two_triangles_by_hand(two_triangles):
    """Zwei Dreiecke: Q = 2 · (3/6 − (6/12)²) = 0,5 für die Zerlegung in die Dreiecke; 0 für alles in einer Gruppe."""
    n, u, v = two_triangles
    a = A.dense_adjacency(n, u, v)
    assert A.modularity(a, [0, 0, 0, 1, 1, 1]) == pytest.approx(0.5)
    assert A.planted_modularity(a, np.array([0, 0, 0, 0, 0, 0])) == pytest.approx(0.0)


def test_leiden_finds_the_two_triangles(two_triangles):
    n, u, v = two_triangles
    lab = A.leiden_labels(A.dense_adjacency(n, u, v), 1)
    assert lab.max() + 1 == 2 and len({lab[0], lab[1], lab[2]}) == 1 and lab[0] != lab[3]
