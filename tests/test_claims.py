"""Jede Zahl aus der README wird hier über dieselben Auswertungsfunktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Netze der Messreihen: die acht festen Netze `SWEEP_SEEDS` je Zelle, q = 2, n = 400, c = 6 (wo nichts anderes steht); BP und Leiden laufen mit festen Seeds und sind plattformstabil,
Kennzahlen werden mit Band geprüft. NMI = normalisierte gegenseitige Information zur geplanten Einteilung."""

from functools import lru_cache

import numpy as np
import pytest

import sbm_algorithm as A
import sbm_constants as C
import sbm_evaluation as ev

S = C.SWEEP_SEEDS


@lru_cache(maxsize=None)
def _detect(q, n, c, lams=None):
    return ev.detect_curve(q, n, c, **({} if lams is None else {"lams": lams}), seeds=S)


def _by_lam(rows):
    return {round(r["lam"], 3): r for r in rows}


def test_detectability_curve_q2_c6():
    rows = _by_lam(_detect(2, 400, 6))
    assert max(rows[lam]["bp"] for lam in (0.0, 0.5, 0.8, 1.0, 1.2)) <= 0.01            # bis 1,2-fach: nichts zu finden
    assert rows[1.5]["bp"] == pytest.approx(0.607, abs=0.02) and rows[2.0]["bp"] == pytest.approx(0.911, abs=0.02) and rows[2.4]["bp"] == pytest.approx(0.989, abs=0.02)
    assert rows[1.5]["em"] == pytest.approx(0.367, abs=0.03) and rows[2.0]["em"] == pytest.approx(0.902, abs=0.02) and rows[2.4]["em"] == pytest.approx(0.977, abs=0.02)
    assert rows[1.5]["leiden"] == pytest.approx(0.117, abs=0.02) and rows[2.0]["leiden"] == pytest.approx(0.322, abs=0.03) and rows[2.4]["leiden"] == pytest.approx(0.670, abs=0.03)
    assert max(rows[lam]["leiden"] for lam in (0.0, 0.5, 0.8, 1.0, 1.2)) <= 0.07


def test_detectability_curve_q2_c3_and_q3():
    rows = _by_lam(_detect(2, 400, 3))
    assert rows[1.0]["bp"] <= 0.03 and rows[1.2]["bp"] == pytest.approx(0.22, abs=0.03) and rows[1.5]["bp"] == pytest.approx(0.663, abs=0.03)
    assert rows[1.2]["leiden"] == pytest.approx(0.118, abs=0.02) and rows[1.5]["leiden"] == pytest.approx(0.233, abs=0.03)
    assert 1.8 not in rows and 2.4 not in rows                                         # über √3 = 1,73 nicht zulässig
    q3 = _by_lam(_detect(3, 600, 6, lams=(0.5, 1.0, 1.3, 1.6, 2.0)))
    assert q3[1.0]["bp"] <= 0.02 and q3[1.3]["bp"] == pytest.approx(0.411, abs=0.03) and q3[1.3]["em"] == pytest.approx(0.108, abs=0.04) and q3[1.3]["leiden"] == pytest.approx(0.12, abs=0.03)
    assert q3[1.6]["bp"] == pytest.approx(0.715, abs=0.03) and q3[1.6]["em"] == pytest.approx(0.721, abs=0.03) and q3[1.6]["leiden"] == pytest.approx(0.252, abs=0.03)
    assert q3[2.0]["bp"] == pytest.approx(0.923, abs=0.02) and q3[2.0]["em"] == pytest.approx(0.917, abs=0.02) and q3[2.0]["leiden"] == pytest.approx(0.598, abs=0.04)


@lru_cache(maxsize=None)
def _random():
    return ev.random_modularity_rows(seeds=S)


def test_modularity_in_pure_random_networks():
    rows = {(r["n"], r["c"]): r for r in _random()}
    expect = {(100, 3): (0.576, 16.0), (100, 6): (0.368, 7.5), (100, 10): (0.268, 7.0), (300, 3): (0.628, 32.0), (300, 6): (0.400, 11.0), (300, 10): (0.286, 11.0),
              (600, 3): (0.635, 50.5), (600, 6): (0.399, 16.0), (600, 10): (0.282, 13.5)}
    for key, (q_med, k_med) in expect.items():
        assert rows[key]["q_median"] == pytest.approx(q_med, abs=0.01), key
        assert rows[key]["k_median"] == pytest.approx(k_med, abs=1.5), key
    assert min(r["q_median"] for r in rows.values()) > 0.26 and max(r["q_median"] for r in rows.values()) < 0.64


@lru_cache(maxsize=None)
def _planted():
    return ev.planted_vs_found(2, 400, 6, seeds=S)


def test_modularity_of_the_found_against_the_planted_partition():
    rows = _by_lam(_planted())
    assert rows[0.0]["q_found"] == pytest.approx(0.390, abs=0.01) and rows[0.0]["q_planted"] == pytest.approx(0.0, abs=0.01) and rows[0.0]["k"] == pytest.approx(14.5, abs=1.5)
    assert rows[1.0]["q_found"] == pytest.approx(0.395, abs=0.01) and rows[1.0]["q_planted"] == pytest.approx(0.203, abs=0.02)
    assert rows[2.0]["q_found"] == pytest.approx(0.438, abs=0.01) and rows[2.0]["q_planted"] == pytest.approx(0.408, abs=0.01)
    assert rows[2.0]["q_found"] > rows[2.0]["q_planted"] and rows[2.0]["nmi_leiden"] == pytest.approx(0.322, abs=0.03) and rows[2.0]["nmi_bp"] == pytest.approx(0.911, abs=0.02)
    assert rows[2.4]["q_found"] == pytest.approx(0.492, abs=0.01) and rows[2.4]["q_planted"] == pytest.approx(0.491, abs=0.01) and rows[2.4]["k"] == pytest.approx(4.5, abs=1.0)
    assert abs(rows[1.0]["q_found"] - rows[0.0]["q_found"]) < 0.02                       # Rauschen und Signal an der Grenze: derselbe Wert


def test_disassortative_groups():
    rows = _by_lam(ev.disassortative_rows(400, 6, seeds=S))
    assert rows[-1.0]["bp"] <= 0.03 and rows[-1.5]["bp"] == pytest.approx(0.575, abs=0.03) and rows[-2.0]["bp"] == pytest.approx(0.902, abs=0.02)
    assert max(r["leiden"] for r in rows.values()) <= 0.01 and max(r["em"] for r in rows.values()) <= 0.001


def test_em_invents_structure_without_a_signal():
    rows = {r["lam"]: r for r in ev.em_invention_rows(400, 6, seeds=S)}
    assert rows[0.0]["runs"] == 8 and round(rows[0.0]["share_above"] * 8) == 5 and rows[0.0]["hat_median"] == pytest.approx(1.63, abs=0.1)
    assert rows[0.0]["imbalance"] == pytest.approx(0.355, abs=0.03)
    assert round(rows[0.5]["share_above"] * 8) == 5 and rows[0.5]["hat_median"] == pytest.approx(1.40, abs=0.1)
    assert round(rows[1.0]["share_above"] * 8) == 5 and rows[1.0]["hat_median"] == pytest.approx(1.30, abs=0.1) and rows[1.0]["imbalance"] == pytest.approx(0.175, abs=0.03)


def test_stability_over_seeds_is_no_diagnosis():
    rows = {r["lam"]: r["nmi"] for r in ev.stability_rows(2, 400, 6, seeds=S)}
    assert rows[0.0] == pytest.approx(0.279, abs=0.03) and rows[1.0] == pytest.approx(0.284, abs=0.03) and rows[2.0] == pytest.approx(0.348, abs=0.04) and rows[2.4] == pytest.approx(0.534, abs=0.05)
    assert abs(rows[2.0] - rows[0.0]) < 0.1


def test_zachary_karate_club_margin_note():
    nx = pytest.importorskip("networkx")
    g = nx.karate_club_graph()
    n = g.number_of_nodes()
    labels = np.array([0 if g.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)])
    u = np.array([min(a, b) for a, b in g.edges()])
    v = np.array([max(a, b) for a, b in g.edges()])
    lab = A.leiden_labels(A.dense_adjacency(n, u, v), 0)
    em = A.em_parameters(n, u, v, 2, 0)
    assert (n, len(u)) == (34, 78) and int(lab.max()) + 1 == 4
    assert A.nmi(labels, lab) == pytest.approx(0.49, abs=0.02)
    assert A.nmi(labels, em.labels) <= 0.01


def test_example_default_network_numbers():
    """Standardnetz (q = 2, n = 400, c = 6, λ = 1,5, Seed 35): die Zahlen der Voreinstellung."""
    a = ev.analyse(ev.Settings())
    assert a.inst.m == 1189 and a.k_found == 12 and a.infer_iterations == 98 and a.infer_converged
    assert a.nmi_infer == pytest.approx(0.60, abs=0.02) and a.nmi_leiden == pytest.approx(0.14, abs=0.02)
    assert a.q_found == pytest.approx(0.40, abs=0.01) and a.q_planted == pytest.approx(0.31, abs=0.01)
