"""Orakel-Tests stochastic-block-demo (DEMO-PLAYBOOK §12): jede Kernroutine gegen einen anderen Rechenweg.

- Belief Propagation: Randverteilungen gegen die Aufzählung aller Einteilungen auf Zufallsbäumen (dort ist BP ohne Außenfeld exakt)
- Modularität: networkx.community.modularity; NMI: scikit-learn; beste Umbenennung: Hungarian (scipy)
- Sampler: Erwartungswert der Kantenzahl gegen Messung; EM: Wiederherstellung der Parameter; Erkennbarkeit als Eigenschaftstest mit Band
"""

import itertools

import numpy as np
import pytest

import sbm_algorithm as A
import sbm_constants as C
from conftest import edges_of

nx = pytest.importorskip("networkx")
sk_metrics = pytest.importorskip("sklearn.metrics")
scipy_opt = pytest.importorskip("scipy.optimize")


def _random_tree(n, rng):
    """Zufallsbaum: Knoten i hat einen Elternteil unter 0 bis i - 1."""
    return [(int(rng.integers(0, i)), i) for i in range(1, n)]


def test_bp_without_field_matches_enumeration_on_random_trees():
    """BP ohne Außenfeld ist auf Bäumen exakt: P(t) ∝ ∏ n_t ∏_{Kanten} c_{t t'}, Randverteilungen durch Aufzählung aller q^n Einteilungen."""
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(30):
        n = int(rng.integers(3, 9))
        q = int(rng.integers(2, 4))
        pairs = _random_tree(n, rng)
        u, v = edges_of(pairs)
        cm = rng.random((q, q)) * 5 + 0.2
        cm = (cm + cm.T) / 2
        nvec = rng.random(q) + 0.3
        nvec /= nvec.sum()
        res = A.bp_marginals(n, u, v, q, cm, nvec, int(rng.integers(1000)), iters=80, damping=0.0, noise=0.3, use_field=False)
        exact = np.zeros((n, q))
        total = 0.0
        for t in itertools.product(range(q), repeat=n):
            w = np.prod([nvec[x] for x in t]) * np.prod([cm[t[a], t[b]] for a, b in pairs])
            total += w
            for i, x in enumerate(t):
                exact[i, x] += w
        exact /= total
        worst = max(worst, float(np.abs(exact - res.marg).max()))
    assert worst < 1e-8


def test_modularity_matches_networkx_on_random_partitions():
    rng = np.random.default_rng(6)
    for _ in range(30):
        n = int(rng.integers(8, 40))
        labels_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.2]
        if not labels_pairs:
            continue
        u, v = edges_of(labels_pairs)
        labels = rng.integers(0, 4, size=n)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(labels_pairs)
        comms = [set(np.where(labels == k)[0].tolist()) for k in range(4) if (labels == k).any()]
        assert A.modularity(A.dense_adjacency(n, u, v), labels) == pytest.approx(nx.community.modularity(g, comms), abs=1e-9)


def test_nmi_matches_scikit_learn():
    rng = np.random.default_rng(7)
    for _ in range(100):
        n = int(rng.integers(2, 60))
        a = rng.integers(0, int(rng.integers(1, 6)), size=n)
        b = rng.integers(0, int(rng.integers(1, 6)), size=n)
        assert A.nmi(a, b) == pytest.approx(sk_metrics.normalized_mutual_info_score(a, b), abs=1e-9)


def test_best_permutation_accuracy_matches_the_hungarian_method():
    rng = np.random.default_rng(8)
    for _ in range(40):
        q = int(rng.integers(2, 5))
        n = int(rng.integers(5, 50))
        true = rng.integers(0, q, size=n)
        pred = rng.integers(0, q, size=n)
        table = np.zeros((q, q))
        np.add.at(table, (true, pred), 1)
        rows, cols = scipy_opt.linear_sum_assignment(-table)
        assert A.best_permutation_accuracy(true, pred, q) == pytest.approx(table[rows, cols].sum() / n)


def test_sampler_edge_count_matches_the_expectation():
    """E[Kanten] = Σ_{i<j} c_{g(i) g(j)} / n; Mittel über 40 Netze bis auf 3 %."""
    n, q = 80, 2
    cm = A.symmetric_cmat(q, 9.0, 3.0)
    labels = A.group_labels(n, q)
    expected = sum(cm[labels[i], labels[j]] for i in range(n) for j in range(i + 1, n)) / n
    counts = [len(A.sample_sbm(n, q, cm, s)[1]) for s in range(40)]
    assert np.mean(counts) == pytest.approx(expected, rel=0.03)
    # Kanten innerhalb der Gruppen: Erwartung (Σ innen c_in) / n
    inside = sum(cm[0, 0] for i in range(n) for j in range(i + 1, n) if labels[i] == labels[j]) / n
    in_counts = []
    for s in range(40):
        _, u, v = A.sample_sbm(n, q, cm, s)
        in_counts.append(int(np.sum(labels[u] == labels[v])))
    assert np.mean(in_counts) == pytest.approx(inside, rel=0.04)


def test_em_recovers_the_parameters_of_a_clear_signal():
    cin, cout, _ = A.affinities(2, 6, 2.0)
    labels, u, v = A.sample_sbm(600, 2, A.symmetric_cmat(2, cin, cout), 11)
    em = A.em_parameters(600, u, v, 2, 11)
    assert A.nmi(labels, em.labels) > 0.7
    assert sorted([em.cmat[0, 0], em.cmat[1, 1]])[0] == pytest.approx(cin, abs=1.0)
    assert em.cmat[0, 1] == pytest.approx(cout, abs=1.0)
    assert em.nvec.sum() == pytest.approx(1.0) and np.all(em.nvec > 0.3)


def test_detectability_is_a_band_not_a_point():
    """Eigenschaftstest (q = 2, n = 600, c = 6, sechs Netze je Stufe): unterhalb der Grenze höchstens 0,05, bei 1,5-facher Grenze mindestens 0,4, bei doppelter mindestens 0,7."""
    medians = {}
    for lam in (0.8, 1.5, 2.0):
        cin, cout, lam_eff = A.affinities(2, 6, lam)
        values = []
        for seed in range(6):
            labels, u, v = A.sample_sbm(600, 2, A.symmetric_cmat(2, cin, cout), 500 + seed)
            res = A.bp_marginals(600, u, v, 2, A.symmetric_cmat(2, cin, cout), np.full(2, 0.5), seed)
            values.append(A.nmi(labels, res.labels))
        medians[lam] = float(np.median(values))
    assert medians[0.8] <= 0.05 and medians[1.5] >= 0.4 and medians[2.0] >= 0.7


def test_disassortative_groups_are_found_by_bp_and_not_by_leiden():
    cin, cout, _ = A.affinities(2, 6, -2.0)
    cm = A.symmetric_cmat(2, cin, cout)
    labels, u, v = A.sample_sbm(400, 2, cm, 21)
    bp = A.bp_marginals(400, u, v, 2, cm, np.full(2, 0.5), 21)
    lab = A.leiden_labels(A.dense_adjacency(400, u, v), 21)
    assert A.nmi(labels, bp.labels) > 0.6 and A.nmi(labels, lab) < 0.1
    assert A.modularity(A.dense_adjacency(400, u, v), labels) < 0          # die wahre Einteilung hat negative Modularität


def test_leiden_modularity_is_high_even_without_any_community():
    """Reines Zufallsnetz (c = 6, n = 300): Leiden meldet Modularität über 0,3 und mehrere Gemeinschaften."""
    labels, u, v = A.sample_sbm(300, 2, A.symmetric_cmat(2, 6.0, 6.0), 31)
    adj = A.dense_adjacency(300, u, v)
    lab = A.leiden_labels(adj, 31)
    assert A.modularity(adj, lab) > 0.3 and lab.max() + 1 >= 5
