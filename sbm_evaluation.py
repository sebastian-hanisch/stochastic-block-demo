"""Kennzahlen, Sweeps und Verdikte der Blockmodell-Demo. Verdikte sind reine Funktionen (Art, Text); jede Zahl im Text kommt aus der Rechnung."""

import itertools
from dataclasses import dataclass
from statistics import median

import numpy as np

import sbm_algorithm as A
import sbm_constants as C
from sbm_scenario import Instance, make_instance


def de(x, digits=1) -> str:
    """Dezimalzahl mit Komma (deutsche Texte)."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    q: int = C.DEFAULT_Q
    n: int = C.DEFAULT_N
    c: int = C.DEFAULT_C
    lam: float = C.DEFAULT_LAM
    seed: int = C.DEFAULT_SEED
    mode: str = C.MODE_KNOWN


@dataclass(frozen=True, eq=False)
class Analysis:
    inst: Instance
    mode: str
    infer_labels: np.ndarray      # Einteilung des Blockmodells (bekannte oder gelernte Parameter)
    infer_iterations: int
    infer_converged: object       # True/False bei BP, None beim EM
    infer_cmat: np.ndarray        # verwendete bzw. gelernte Affinitäten
    infer_nvec: np.ndarray
    leiden_labels: np.ndarray
    k_found: int                  # Zahl der Gemeinschaften von Leiden
    q_found: float                # Modularität der Leiden-Zerlegung
    q_planted: float              # Modularität der geplanten Einteilung
    nmi_infer: float
    nmi_leiden: float
    overlap_infer: float


def analyse(settings: Settings) -> Analysis:
    inst = make_instance(settings.q, settings.n, settings.c, settings.lam, settings.seed)
    adj = A.dense_adjacency(inst.n, inst.u, inst.v)
    if settings.mode == C.MODE_KNOWN:
        nvec = np.full(inst.q, 1.0 / inst.q)
        bp = A.bp_marginals(inst.n, inst.u, inst.v, inst.q, inst.cmat, nvec, settings.seed)
        labels, its, conv, cmat = bp.labels, bp.iterations, bp.converged, inst.cmat
    else:
        em = A.em_parameters(inst.n, inst.u, inst.v, inst.q, settings.seed)
        labels, its, conv, cmat, nvec = em.labels, em.iterations, None, em.cmat, em.nvec
    lab = A.leiden_labels(adj, settings.seed)
    return Analysis(inst, settings.mode, labels, its, conv, cmat, nvec, lab, int(lab.max()) + 1 if len(lab) else 0,
                    A.modularity(adj, lab), A.planted_modularity(adj, inst.labels),
                    A.nmi(inst.labels, labels), A.nmi(inst.labels, lab), A.overlap(inst.labels, labels, inst.q))


def above_threshold(lam_eff) -> bool:
    """Liegt die Signalstärke über der Erkennbarkeitsgrenze (|λ| > 1, dünner symmetrischer Grenzfall)?"""
    return abs(lam_eff) > 1.0


# --- Hauptverdikt -----------------------------------------------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    """Verdikt der Kernfrage: findet ein Verfahren hier eine Gemeinschaft, die es wirklich gibt?"""
    name = "Blockmodell (Parameter bekannt)" if a.mode == C.MODE_KNOWN else "Blockmodell (Parameter gelernt)"
    if a.inst.m == 0:
        return "warning", "Das Netz hat keine Kanten: es gibt nichts zu finden."
    lam = a.inst.lam
    found = (f"Leiden zerlegt das Netz in {a.k_found} Gemeinschaften mit Modularität {de(a.q_found, 2)} (die geplante Einteilung hat {de(a.q_planted, 2)}), "
             f"Überlappung mit den geplanten Gruppen NMI {de(a.nmi_leiden, 2)}; das {name} erreicht NMI {de(a.nmi_infer, 2)}.")
    if not above_threshold(lam):
        return "warning", (f"Die Signalstärke λ = {de(lam, 2)} liegt **unter der Erkennbarkeitsgrenze 1**: für kein Verfahren gibt es hier eine Einteilung zu finden. " + found +
                           " Die Gemeinschaften, die Leiden meldet, sind Zufall.")
    if a.nmi_infer >= 0.5 and a.nmi_infer > a.nmi_leiden + 0.1:
        return "success", (f"Die Signalstärke λ = {de(lam, 2)} liegt über der Grenze, und das {name} findet die geplanten Gruppen (NMI {de(a.nmi_infer, 2)}), Leiden deutlich schlechter (NMI {de(a.nmi_leiden, 2)}): "
                           f"{a.k_found} Gemeinschaften mit Modularität {de(a.q_found, 2)}, die geplante Einteilung hat {de(a.q_planted, 2)}.")
    return "info", f"Die Signalstärke λ = {de(lam, 2)} liegt über der Grenze. " + found


# --- E1: Erkennbarkeit -------------------------------------------------------------------------------------------------------------------------------


def _med(values):
    return float(median(values)) if values else 0.0


def detect_curve(q, n, c, lams=C.DETECT_LAMBDAS, seeds=C.LIVE_SEEDS) -> list[dict]:
    """NMI zur geplanten Einteilung über die Signalstärke: Blockmodell mit bekannten Parametern (BP), mit gelernten (EM) und Leiden, Median über die festen Netze.
    Signalstärken außerhalb des zulässigen Bereichs (c_in, c_out >= 0) entfallen."""
    lo, hi = C.lam_limits(q, c)
    rows = []
    for lam in lams:
        if not lo - 1e-9 <= lam <= hi + 1e-9:
            continue
        bp_v, em_v, ld_v = [], [], []
        for seed in seeds:
            inst = make_instance(q, n, c, lam, seed, layout=False)
            if inst.m == 0:
                continue
            bp = A.bp_marginals(n, inst.u, inst.v, q, inst.cmat, np.full(q, 1.0 / q), seed)
            em = A.em_parameters(n, inst.u, inst.v, q, seed)
            lab = A.leiden_labels(A.dense_adjacency(n, inst.u, inst.v), seed)
            bp_v.append(A.nmi(inst.labels, bp.labels))
            em_v.append(A.nmi(inst.labels, em.labels))
            ld_v.append(A.nmi(inst.labels, lab))
        rows.append({"lam": lam, "bp": _med(bp_v), "em": _med(em_v), "leiden": _med(ld_v)})
    return rows


def verdict_detect(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E1: ab welcher Signalstärke findet das Blockmodell etwas, was erreicht Leiden dort, und was gilt unterhalb der Grenze?"""
    hit = [r for r in rows if r["bp"] >= 0.3]
    below = [r["bp"] for r in rows if r["lam"] <= 1.0]
    if not hit:
        return "warning", "In diesem Bereich der Signalstärke findet auch das Blockmodell die Gruppen nicht (NMI unter 0,3); stärkere Signale oder größere Netze wählen."
    first = hit[0]
    low = max(below) if below else 0.0
    best = rows[-1]
    return "info", (f"Das Blockmodell (bekannte Parameter) findet die Gruppen ab λ = {de(first['lam'], 1)} (NMI {de(first['bp'], 2)}); bis λ = 1,0 liegt es bei höchstens {de(low, 2)}. "
                    f"Leiden erreicht bei λ = {de(first['lam'], 1)} nur {de(first['leiden'], 2)} und bei λ = {de(best['lam'], 1)} {de(best['leiden'], 2)} (Blockmodell {de(best['bp'], 2)}).")


# --- E2: Modularität im Zufallsnetz und gegen die Wahrheit -----------------------------------------------------------------------------------------


def random_modularity_rows(sizes=C.RANDOM_SIZES, degrees=C.RANDOM_DEGREES, seeds=C.LIVE_SEEDS) -> list[dict]:
    """Beste Leiden-Modularität in **reinen** Zufallsnetzen (Signalstärke 0) je Größe und mittlerem Grad: Median, Minimum, Maximum und Zahl der Gemeinschaften über die festen Netze."""
    rows = []
    for n in sizes:
        for c in degrees:
            qs, ks = [], []
            for seed in seeds:
                inst = make_instance(2, n, c, 0.0, seed, layout=False)
                adj = A.dense_adjacency(n, inst.u, inst.v)
                lab = A.leiden_labels(adj, seed)
                qs.append(A.modularity(adj, lab))
                ks.append(int(lab.max()) + 1)
            rows.append({"n": n, "c": c, "q_median": _med(qs), "q_min": min(qs), "q_max": max(qs), "k_median": _med(ks)})
    return rows


def verdict_random(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E2a: wie hoch wird die Modularität in Netzen ohne jede Gemeinschaft?"""
    lo = min(r["q_median"] for r in rows)
    hi = max(r["q_median"] for r in rows)
    top = max(rows, key=lambda r: r["q_median"])
    return "warning", (f"In Zufallsnetzen **ohne jede Gemeinschaft** erreicht die Modularität im Median {de(lo, 2)} bis {de(hi, 2)}, am höchsten bei n = {top['n']} und mittlerem Grad {top['c']} "
                       f"({de(top['q_median'], 2)} mit etwa {top['k_median']:.0f} Gemeinschaften). Ein hoher Wert beweist keine Struktur.")


def planted_vs_found(q, n, c, lams=C.PLANTED_LAMBDAS, seeds=C.LIVE_SEEDS) -> list[dict]:
    """Modularität der Leiden-Zerlegung gegen die der geplanten Einteilung, dazu Zahl der Gemeinschaften und NMI beider Verfahren, Median über die festen Netze."""
    lo, hi = C.lam_limits(q, c)
    rows = []
    for lam in lams:
        if not lo - 1e-9 <= lam <= hi + 1e-9:
            continue
        qf, qp, ks, nl, nb = [], [], [], [], []
        for seed in seeds:
            inst = make_instance(q, n, c, lam, seed, layout=False)
            if inst.m == 0:
                continue
            adj = A.dense_adjacency(n, inst.u, inst.v)
            lab = A.leiden_labels(adj, seed)
            bp = A.bp_marginals(n, inst.u, inst.v, q, inst.cmat, np.full(q, 1.0 / q), seed)
            qf.append(A.modularity(adj, lab))
            qp.append(A.planted_modularity(adj, inst.labels))
            ks.append(int(lab.max()) + 1)
            nl.append(A.nmi(inst.labels, lab))
            nb.append(A.nmi(inst.labels, bp.labels))
        rows.append({"lam": lam, "q_found": _med(qf), "q_planted": _med(qp), "k": _med(ks), "nmi_leiden": _med(nl), "nmi_bp": _med(nb)})
    return rows


def verdict_planted(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E2b: gibt es eine Signalstärke, bei der Leiden eine Zerlegung mit höherer Modularität als die wahre findet, obwohl sie ihr kaum ähnelt?"""
    worse = [r for r in rows if r["q_found"] > r["q_planted"] + 1e-9 and r["nmi_leiden"] < 0.5 and r["nmi_bp"] >= 0.5]
    zero = next((r for r in rows if r["lam"] == 0.0), None)
    one = next((r for r in rows if r["lam"] == 1.0), None)
    note = ""
    if zero is not None and one is not None:
        note = f" Rauschen (λ = 0) kommt auf {de(zero['q_found'], 3)}, ein Signal an der Grenze (λ = 1,0) auf {de(one['q_found'], 3)}: der Wert allein unterscheidet beide nicht."
    if worse:
        r = worse[0]
        return "warning", (f"Bei λ = {de(r['lam'], 1)} findet Leiden eine Zerlegung mit **höherer Modularität ({de(r['q_found'], 3)}) als die geplante ({de(r['q_planted'], 3)})**, "
                           f"die der Wahrheit nur zu NMI {de(r['nmi_leiden'], 2)} ähnelt; das Blockmodell findet sie zu {de(r['nmi_bp'], 2)}. Das Modularitätsoptimum ist nicht die Wahrheit." + note)
    return "info", "In diesem Bereich liegt die Modularität der geplanten Einteilung nirgends unter der von Leiden gefundenen." + note


# --- E3: Lieferstufen -------------------------------------------------------------------------------------------------------------------------------


def disassortative_rows(n, c, lams=C.DIS_LAMBDAS, seeds=C.LIVE_SEEDS) -> list[dict]:
    """Zwei Gruppen mit Kanten vorwiegend zwischen den Gruppen (negative Signalstärke): NMI von Blockmodell (bekannt), EM (assortativ gestartet) und Leiden."""
    return detect_curve(2, n, c, lams, seeds)


def verdict_disassortative(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E3: findet Modularität Lieferstufen, findet es das Blockmodell?"""
    if not rows:
        return "warning", "Für diesen mittleren Grad gibt es keine zulässige negative Signalstärke."
    strong = rows[-1]
    top_leiden = max(r["leiden"] for r in rows)
    return "info", (f"Bei λ = {de(strong['lam'], 1)} findet das Blockmodell die Stufen (NMI {de(strong['bp'], 2)}), Leiden nicht (NMI {de(strong['leiden'], 2)}; über alle Zeilen höchstens {de(top_leiden, 2)}): "
                    f"Modularität sucht dichte Gruppen und sieht Kanten **zwischen** Gruppen nicht als Gemeinschaft. Das EM, assortativ gestartet, scheitert ebenfalls (NMI {de(strong['em'], 2)}).")


# --- E4: Parameter lernen ---------------------------------------------------------------------------------------------------------------------------


def em_invention_rows(n, c, lams=C.EM_LAMBDAS, seeds=C.LIVE_SEEDS) -> list[dict]:
    """Was lernt das EM unterhalb der Schwelle? Gelernter Kontrast λ̂ und Gruppenungleichgewicht |n_1 - 0,5| (q = 2); Anteil der Läufe mit λ̂ > 1."""
    rows = []
    for lam in lams:
        hats, imbalance = [], []
        for seed in seeds:
            inst = make_instance(2, n, c, lam, seed, layout=False)
            if inst.m == 0:
                continue
            em = A.em_parameters(n, inst.u, inst.v, 2, seed)
            hats.append(A.learned_contrast(em.cmat, inst.mean_degree))
            imbalance.append(abs(float(em.nvec[0]) - 0.5))
        rows.append({"lam": lam, "hat_median": _med(hats), "share_above": sum(1 for h in hats if h > 1.0) / len(hats) if hats else 0.0,
                     "runs": len(hats), "imbalance": _med(imbalance)})
    return rows


def verdict_em(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E4: erfindet das EM Struktur im Netz ohne Gemeinschaft?"""
    zero = next((r for r in rows if r["lam"] == 0.0), None)
    if zero is None:
        return "info", "Für dieses Experiment fehlt die Zeile ohne Signal."
    count = round(zero["share_above"] * zero["runs"])
    return "warning", (f"Im Netz **ohne** Gemeinschaft lernt das EM in {count} von {zero['runs']} Läufen einen Kontrast über der Schwelle (Median λ̂ = {de(zero['hat_median'], 2)}, Gruppenungleichgewicht {de(zero['imbalance'], 2)}): "
                       "es erfindet eine kleine dichte Gruppe. Ohne Modellwahl überanpasst eine lernende Inferenz.")


# --- E5: Stabilität über Seeds ---------------------------------------------------------------------------------------------------------------------


def stability_rows(q, n, c, lams=C.STABILITY_LAMBDAS, seeds=C.LIVE_SEEDS, runs=C.STABILITY_RUNS) -> list[dict]:
    """Mittlere paarweise NMI zwischen `runs` Leiden-Läufen mit verschiedenem Seed auf demselben Netz, Median über die festen Netze."""
    lo, hi = C.lam_limits(q, c)
    rows = []
    for lam in lams:
        if not lo - 1e-9 <= lam <= hi + 1e-9:
            continue
        per_net = []
        for seed in seeds:
            inst = make_instance(q, n, c, lam, seed, layout=False)
            adj = A.dense_adjacency(n, inst.u, inst.v)
            labs = [A.leiden_labels(adj, 1000 * seed + r) for r in range(runs)]
            per_net.append(median(A.nmi(a, b) for a, b in itertools.combinations(labs, 2)))
        rows.append({"lam": lam, "nmi": _med(per_net)})
    return rows


def verdict_stability(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E5: zeigt die Stabilität über Seeds, ob Struktur da ist?"""
    if len(rows) < 2:
        return "info", "Für dieses Experiment fehlen Signalstärken."
    first, last = rows[0], rows[-1]
    detail = ", ".join(f"λ = {de(r['lam'], 1)}: {de(r['nmi'], 2)}" for r in rows)
    return "info", f"Mittlere NMI zwischen {C.STABILITY_RUNS} Leiden-Läufen: {detail}. Die Läufe unterscheiden sich im Netz ohne Gemeinschaft ({de(first['nmi'], 2)}) kaum anders als bei erkennbarer Struktur; erst bei sehr starkem Signal steigt der Wert ({de(last['nmi'], 2)}). Stabilität über Seeds ist keine Diagnose."
