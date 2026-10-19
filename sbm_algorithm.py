"""Kern: Blockmodell (SBM) mit Belief Propagation (BP) und EM, Modularität und Leiden (Kopie aus leiden-demo), Überlappungsmaße.

Alle Einheiten sind eigene Funktionen ohne versteckten Zustand; Kanten sind zwei Arrays (u, v) mit u < v, Knoten 0 bis n - 1. Zufall nur über `numpy.random.default_rng` mit übergebenem Seed.
Das Modell (Decelle, Krzakala, Moore, Zdeborová 2011, dünner Grenzfall): q gleich große Gruppen, Kante zwischen Gruppe a und b mit Wahrscheinlichkeit c_ab / n.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np

import sbm_constants as C
import sbm_leiden as L


# --- Affinitäten ---------------------------------------------------------------------------------------------------------------------------------


def symmetric_cmat(q, cin, cout):
    """Affinitätsmatrix: c_in auf der Diagonale, c_out daneben."""
    m = np.full((q, q), float(cout))
    np.fill_diagonal(m, float(cin))
    return m


def signal_strength(cin, cout, q, c):
    """λ = (c_in - c_out) / (q √c): das Vielfache der Kesten-Stigum-Schwelle (Erkennbarkeit im dünnen symmetrischen Fall ab |λ| > 1)."""
    return (cin - cout) / (q * math.sqrt(c))


def affinities(q, c, lam):
    """(c_in, c_out, λ_eff) zu Gruppenzahl q, mittlerem Grad c und gewünschter Signalstärke λ; λ wird auf den zulässigen Bereich begrenzt (c_in, c_out >= 0).
    Mittlerer Grad: (c_in + (q - 1) c_out) / q = c."""
    lo, hi = C.lam_limits(q, c)
    lam_eff = min(max(float(lam), lo), hi)
    delta = lam_eff * q * math.sqrt(c)
    cin = c + (q - 1) * delta / q
    cout = c - delta / q
    return max(cin, 0.0), max(cout, 0.0), lam_eff


# --- Sampler -----------------------------------------------------------------------------------------------------------------------------------------


def group_labels(n, q):
    """Gleich große Gruppen (die letzte nimmt den Rest), zusammenhängend nummeriert."""
    sizes = [n // q] * q
    sizes[-1] += n - sum(sizes)
    return np.repeat(np.arange(q), sizes)


def sample_sbm(n, q, cmat, seed):
    """Blockmodell: Knoten i, j (i < j) sind mit Wahrscheinlichkeit c_{g(i) g(j)} / n verbunden. Gibt (labels, u, v) mit u < v zurück."""
    rng = np.random.default_rng(int(seed))
    labels = group_labels(n, q)
    prob = cmat[np.ix_(labels, labels)] / n
    adjacency = np.triu(rng.random((n, n)) < prob, 1)
    u, v = np.nonzero(adjacency)
    return labels, u, v


def dense_adjacency(n, u, v):
    a = np.zeros((n, n))
    a[u, v] = 1.0
    a[v, u] = 1.0
    return a


def mean_degree(n, u):
    return 2.0 * len(u) / n if n else 0.0


# --- Überlappungsmaße -------------------------------------------------------------------------------------------------------------------------------


def nmi(a, b):
    """Normalisierte gegenseitige Information (arithmetischer Mittelwert der Entropien, natürlicher Logarithmus); 1,0, wenn beide Einteilungen trivial sind."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    table = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(table, (ia, ib), 1.0)
    pa = table.sum(1) / n
    pb = table.sum(0) / n
    ha = -float(np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    hb = -float(np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    if ha == 0.0 and hb == 0.0:
        return 1.0
    pab = table / n
    nz = pab > 0
    mi = float(np.sum(pab[nz] * np.log(pab[nz] / np.outer(pa, pb)[nz])))
    return mi / ((ha + hb) / 2)


def best_permutation_accuracy(true, pred, q):
    """Größter Anteil richtig zugeordneter Knoten über alle Umbenennungen der q Gruppen (q <= 4: höchstens 24 Permutationen)."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    if len(true) == 0:
        return 0.0
    best = 0.0
    for perm in itertools.permutations(range(q)):
        mapped = np.asarray(perm)[pred]
        best = max(best, float(np.mean(mapped == true)))
    return best


def overlap(true, pred, q):
    """Überlappung (acc - max_a n_a) / (1 - max_a n_a) mit dem größten Gruppenanteil n_a: 0 = Raten, 1 = alles richtig."""
    true = np.asarray(true)
    if len(true) == 0:
        return 0.0
    biggest = float(np.max(np.bincount(true, minlength=q))) / len(true)
    if biggest >= 1.0:
        return 0.0
    return (best_permutation_accuracy(true, pred, q) - biggest) / (1.0 - biggest)


# --- Belief Propagation -------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BPResult:
    marg: np.ndarray        # (n, q) Randverteilungen
    psi: np.ndarray         # (2m, q) Nachrichten je gerichteter Kante (erste m: u -> v, zweite m: v -> u)
    iterations: int
    converged: bool

    @property
    def labels(self):
        return self.marg.argmax(1)


def bp_marginals(n, u, v, q, cmat, nvec, seed, iters=C.BP_ITERS, damping=C.BP_DAMPING, noise=C.BP_NOISE, use_field=True) -> BPResult:
    """Belief Propagation für das Blockmodell mit bekannten Parametern (Decelle u. a. 2011):
    ψ^{i→j}_t ∝ n_t e^{-h_t} ∏_{k ∈ ∂i \\ j} Σ_s c_{st} ψ^{k→i}_s, Randverteilung mit allen Nachbarn. Das Außenfeld h_t = (1/n) Σ_k Σ_s c_{st} ψ^k_s (Beitrag der
    Nicht-Kanten) wird mit gedämpften Randverteilungen gerechnet (`use_field=False` lässt es weg: dann ist BP auf Bäumen exakt). Start: fast gleichverteilt mit kleinem Rauschen."""
    n = int(n)
    nvec = np.asarray(nvec, dtype=float)
    cmat = np.asarray(cmat, dtype=float)
    m = len(u)
    marg = np.tile(nvec, (n, 1))
    if m == 0:
        return BPResult(marg, np.zeros((0, q)), 0, True)
    rng = np.random.default_rng(int(seed))
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    rev = np.concatenate([np.arange(m, 2 * m), np.arange(0, m)])
    psi = (1.0 / q) * (1.0 + noise * (rng.random((2 * m, q)) - 0.5) * 2.0)
    psi /= psi.sum(1, keepdims=True)
    marg_h = marg.copy()
    logn = np.log(nvec)
    converged = False
    it = 0
    for it in range(1, iters + 1):
        msg = psi @ cmat                                   # Σ_s ψ_s c_{st} aus Sicht des Empfängers
        logm = np.log(msg + 1e-300)
        s_in = np.zeros((n, q))
        for t in range(q):
            s_in[:, t] = np.bincount(dst, weights=logm[:, t], minlength=n)
        h = (marg_h @ cmat).sum(0) / n if use_field else np.zeros(q)
        logmarg = logn - h + s_in
        marg_new = np.exp(logmarg - logmarg.max(1, keepdims=True))
        marg_new /= marg_new.sum(1, keepdims=True)
        out = logn - h + s_in[src] - logm[rev]
        new = np.exp(out - out.max(1, keepdims=True))
        new /= new.sum(1, keepdims=True)
        delta = float(np.abs(new - psi).max())
        psi = damping * psi + (1.0 - damping) * new
        marg = marg_new
        marg_h = C.FIELD_DAMPING * marg_h + (1.0 - C.FIELD_DAMPING) * marg_new
        if delta < C.BP_TOL:
            converged = True
            break
    return BPResult(marg, psi, it, converged)


# --- EM: Parameter lernen -------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EMResult:
    marg: np.ndarray
    cmat: np.ndarray        # gelernte Affinitäten
    nvec: np.ndarray        # gelernte Gruppenanteile
    iterations: int

    @property
    def labels(self):
        return self.marg.argmax(1)


def em_start(q, c, perturbation, rng):
    """Startaffinitäten: assortativ mit Kontrast λ = EM_START_LAMBDA und kleiner symmetrischer Störung (sie bricht die Gleichheit der Gruppen)."""
    delta = C.EM_START_LAMBDA * q * math.sqrt(max(c, 1e-9))
    cm = symmetric_cmat(q, c + (q - 1) * delta / q, max(0.1, c - delta / q))
    cm = cm + perturbation * rng.random((q, q))
    return (cm + cm.T) / 2


def em_update(n, u, v, q, psi, cmat):
    """Neue Gruppenanteile und Affinitäten aus den Kantenrandverteilungen ψ^{ij}_{ab} ∝ c_ab ψ^{i→j}_a ψ^{j→i}_b:
    c_ab = E_ab / (n n_a n_b) für a ≠ b und c_aa = 2 E_aa / (n n_a^2) mit den erwarteten Kantenzahlen E."""
    m = len(u)
    a = psi[:m]
    b = psi[m:]
    joint = a[:, :, None] * b[:, None, :] * cmat[None, :, :]
    joint /= joint.sum((1, 2), keepdims=True)
    js = joint.sum(0)
    expected = js + js.T
    np.fill_diagonal(expected, np.diag(js))
    return expected


def em_parameters(n, u, v, q, seed, outer=C.EM_OUTER, inner=C.EM_INNER) -> EMResult:
    """EM: abwechselnd BP mit den aktuellen Parametern und Parameter-Update aus den Kantenrandverteilungen. Start assortativ (siehe `em_start`)."""
    m = len(u)
    c = mean_degree(n, u)
    rng = np.random.default_rng(int(seed) + 17)
    cmat = em_start(q, c, 0.1, rng)
    nvec = np.full(q, 1.0 / q)
    if m == 0:
        return EMResult(np.tile(nvec, (n, 1)), cmat, nvec, 0)
    marg = np.tile(nvec, (n, 1))
    for _ in range(outer):
        res = bp_marginals(n, u, v, q, cmat, nvec, seed, iters=inner)
        marg = res.marg
        nvec = np.maximum(marg.mean(0), 1e-6)
        nvec = nvec / nvec.sum()
        expected = em_update(n, u, v, q, res.psi, cmat)
        new = np.zeros((q, q))
        for x in range(q):
            for y in range(q):
                new[x, y] = 2 * expected[x, x] / (n * nvec[x] ** 2) if x == y else expected[x, y] / (n * nvec[x] * nvec[y])
        cmat = np.maximum(new, 1e-3)
    return EMResult(marg, cmat, nvec, outer)


def learned_contrast(cmat, c):
    """Gelernter Kontrast λ̂ = |c_aa - c_ab| / (q √c) aus den ersten beiden Gruppen (nur für gleich große Gruppen eine Schwellenaussage)."""
    q = len(cmat)
    return abs(cmat[0, 0] - cmat[0, 1]) / (q * math.sqrt(max(c, 1e-9)))


# --- Modularität und Leiden (Kopie aus leiden-demo) ---------------------------------------------------------------------------------------------


def modularity(adjacency, labels):
    return L.modularity(adjacency, labels)


def leiden_labels(adjacency, seed):
    """Leiden-Zerlegung (Modularität, γ = 1) mit dem Code aus leiden-demo; Etiketten 0 bis k - 1."""
    passes = L._run_passes(adjacency, 1.0, int(seed), True)
    return np.asarray(passes[-1].partition, dtype=int)


def planted_modularity(adjacency, labels):
    return L.modularity(adjacency, np.asarray(labels))
