"""Netz mit geplanter Einteilung (Blockmodell) und Anordnung für die Anzeige."""

import math
from dataclasses import dataclass

import numpy as np

import sbm_algorithm as A


@dataclass(frozen=True, eq=False)
class Instance:
    q: int
    n: int
    labels: np.ndarray      # geplante Gruppe je Knoten
    u: np.ndarray           # Kanten (u < v)
    v: np.ndarray
    cin: float
    cout: float
    lam: float              # tatsächlich verwendete Signalstärke (auf den zulässigen Bereich begrenzt)
    pos: tuple              # (x, y) in [0, 1] für die Anzeige

    @property
    def m(self) -> int:
        return len(self.u)

    @property
    def cmat(self) -> np.ndarray:
        return A.symmetric_cmat(self.q, self.cin, self.cout)

    @property
    def mean_degree(self) -> float:
        return A.mean_degree(self.n, self.u)


def adjacency_lists(n, u, v):
    adj = [[] for _ in range(n)]
    for a, b in zip(u.tolist(), v.tolist()):
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _normalise(pos):
    lo = pos.min(axis=0)
    span = max(float((pos.max(axis=0) - lo).max()), 1e-9)
    out = (pos - lo) / span
    return out + (1 - out.max(axis=0)) / 2


def layout_force(n, u, v, seed, iterations=60):
    """Fruchterman-Reingold, vektorisiert, mit Schwerkraft zur Mitte; deterministisch je Seed. Ausreißer werden auf einen Kreis zurückgeholt. Nur für die Anzeige."""
    if n == 1:
        return ((0.5, 0.5),)
    rng = np.random.default_rng(int(seed))
    pos = rng.random((n, 2))
    edges = np.stack([u, v], axis=1) if len(u) else np.zeros((0, 2), dtype=int)
    k = math.sqrt(1.0 / n)
    t = 0.1
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.sqrt((delta ** 2).sum(-1)), 1e-6)
        disp = (delta / dist[..., None] * (k * k / dist)[..., None]).sum(axis=1)
        if len(edges):
            d = pos[edges[:, 0]] - pos[edges[:, 1]]
            length = np.maximum(np.sqrt((d ** 2).sum(1)), 1e-6)
            f = (d / length[:, None]) * (length * length / k)[:, None]
            np.add.at(disp, edges[:, 0], -f)
            np.add.at(disp, edges[:, 1], f)
        disp -= 0.6 * (pos - pos.mean(axis=0))
        length = np.maximum(np.sqrt((disp ** 2).sum(1)), 1e-9)
        pos = pos + disp / length[:, None] * np.minimum(length, t)[:, None]
        t = max(0.005, t * 0.95)
    center = np.median(pos, axis=0)
    d = pos - center
    r = np.sqrt((d ** 2).sum(1))
    cap = max(1.6 * float(np.percentile(r, 90)), 1e-9)
    pos = center + d * np.minimum(1.0, cap / np.maximum(r, 1e-12))[:, None]
    return tuple((float(x), float(y)) for x, y in _normalise(pos))


def make_instance(q, n, c, lam, seed, layout=True) -> Instance:
    """Blockmodell mit q gleich großen Gruppen, n Knoten, mittlerem Grad c und Signalstärke λ (auf den zulässigen Bereich begrenzt).
    `layout=False` lässt die Kräfte-Anordnung weg (alle Knoten auf (0,5; 0,5)); Sweeps brauchen nur das Netz."""
    cin, cout, lam_eff = A.affinities(q, c, lam)
    labels, u, v = A.sample_sbm(n, q, A.symmetric_cmat(q, cin, cout), seed)
    pos = layout_force(n, u, v, int(seed) % 1_000_003 + 3) if layout else ((0.5, 0.5),) * n
    return Instance(q, n, labels, u, v, cin, cout, lam_eff, pos)
