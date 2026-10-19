"""Netze mit geplanter Einteilung: gültige Netze, Determinismus (auch über Prozesse), Positionen, Begrenzung der Signalstärke."""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sbm_algorithm as A
import sbm_constants as C
import sbm_scenario as S

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("q", [2, 3, 4])
@pytest.mark.parametrize("n,c,lam", [(C.N_MIN, C.C_MIN, C.LAM_MIN), (C.N_MIN, C.C_MAX, C.LAM_MAX), (250, 6, 0.0), (C.N_MAX, 6, 1.5)])
def test_every_corner_gives_a_valid_network(q, n, c, lam):
    inst = S.make_instance(q, n, c, lam, 35)
    assert inst.n == n and len(inst.labels) == n and np.all(inst.u < inst.v) and len(inst.pos) == n
    assert set(np.unique(inst.labels).tolist()) == set(range(q))
    lo, hi = C.lam_limits(q, c)
    assert lo - 1e-9 <= inst.lam <= hi + 1e-9
    for x, y in inst.pos:
        assert math.isfinite(x) and math.isfinite(y) and -1e-9 <= x <= 1 + 1e-9 and -1e-9 <= y <= 1 + 1e-9
    pairs = set(zip(inst.u.tolist(), inst.v.tolist()))
    assert len(pairs) == inst.m                                      # keine Doppelkanten


def test_instances_are_deterministic_per_seed_and_differ_between_seeds():
    a = S.make_instance(2, 120, 6, 1.5, 5)
    b = S.make_instance(2, 120, 6, 1.5, 5)
    assert np.array_equal(a.u, b.u) and np.array_equal(a.v, b.v) and a.pos == b.pos
    c = S.make_instance(2, 120, 6, 1.5, 6, layout=False)
    assert not (len(c.u) == a.m and np.array_equal(c.u, a.u))


def test_the_signal_is_clipped_and_reported():
    inst = S.make_instance(2, 100, 3, 3.0, 1, layout=False)
    assert inst.lam == pytest.approx(math.sqrt(3)) and inst.cout == pytest.approx(0.0, abs=1e-9)
    assert S.make_instance(2, 100, 6, 1.0, 1, layout=False).lam == pytest.approx(1.0)


def test_mean_degree_follows_c_independent_of_the_signal():
    for lam in (0.0, 1.0, 2.0, -1.5):
        inst = S.make_instance(2, 400, 6, lam, 7, layout=False)
        assert inst.mean_degree == pytest.approx(6.0, rel=0.1), lam


def test_the_signal_moves_edges_between_inside_and_outside():
    assort = S.make_instance(2, 400, 6, 2.0, 9, layout=False)
    disassort = S.make_instance(2, 400, 6, -2.0, 9, layout=False)
    inside = lambda inst: float(np.mean(inst.labels[inst.u] == inst.labels[inst.v]))
    assert inside(assort) > 0.8 and inside(disassort) < 0.2


def test_layout_without_edges_and_with_one_node():
    assert S.layout_force(1, np.zeros(0, dtype=int), np.zeros(0, dtype=int), 1) == ((0.5, 0.5),)
    pos = S.layout_force(5, np.zeros(0, dtype=int), np.zeros(0, dtype=int), 1)
    assert len(pos) == 5 and all(0 <= x <= 1 and 0 <= y <= 1 for x, y in pos)


def test_adjacency_lists_are_symmetric():
    inst = S.make_instance(2, 60, 6, 1.0, 3, layout=False)
    adj = S.adjacency_lists(inst.n, inst.u, inst.v)
    for a, nb in enumerate(adj):
        for b in nb:
            assert a in adj[b]


def test_results_do_not_depend_on_the_hash_seed():
    """Dasselbe Szenario in zwei Prozessen mit verschiedenem PYTHONHASHSEED rechnet dasselbe."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import sbm_evaluation as ev, sbm_constants as C\n"
        "out = []\n"
        "for q, lam, mode in ((2, 1.5, C.MODE_KNOWN), (3, 2.0, C.MODE_LEARNED), (2, -2.0, C.MODE_KNOWN)):\n"
        "    a = ev.analyse(ev.Settings(q, 150, 6, lam, 7, mode))\n"
        "    out.append((a.inst.m, round(a.nmi_infer, 9), round(a.nmi_leiden, 9), a.k_found, round(a.q_found, 9)))\n"
        "print(repr(out))\n"
    ) % str(ROOT)
    outputs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONIOENCODING="utf-8")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1]
