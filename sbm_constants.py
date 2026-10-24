"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

# --- Netz ---------------------------------------------------------------------------------------------------------------------------------------
Q_MIN, Q_MAX, DEFAULT_Q = 2, 4, 2                  # Zahl der Gruppen; q >= 5 hat eine "schwere" Phase und ist nicht Teil der Demo
N_MIN, N_MAX, DEFAULT_N = 100, 600, 400            # Slider-Maximum: Sampler O(n^2), Layout O(n^2 * 60), Leiden mit dichter Matrix O(n^2) je Sweep
C_MIN, C_MAX, DEFAULT_C = 3, 10, 6                 # mittlerer Grad
LAM_MIN, LAM_MAX, DEFAULT_LAM = -3.0, 3.0, 1.5     # Signalstärke in Vielfachen der Schwelle; negativ = Lieferstufen (disassortativ)
LAM_STEP = 0.1
SEED_MAX, DEFAULT_SEED = 999999, 35
SWEEP_SEEDS = tuple(range(100000, 100008))         # feste Netze für alle Sweeps: nie ein einzelner Seed als Befund
LIVE_SEEDS = SWEEP_SEEDS[:4]                       # so viele feste Netze je Zelle rechnet die App live (die README-Zahlen nutzen alle acht)

MODE_KNOWN = "Parameter bekannt"
MODE_LEARNED = "Parameter gelernt (EM, assortativ gestartet)"
MODES = (MODE_KNOWN, MODE_LEARNED)

# --- Inferenz -------------------------------------------------------------------------------------------------------------------------------------
BP_ITERS, BP_DAMPING, BP_NOISE = 300, 0.5, 0.05
FIELD_DAMPING = 0.8       # das Außenfeld h wird mit gedämpften Randverteilungen berechnet, sonst kippen alle Knoten synchron in eine Gruppe
BP_TOL = 1e-7
EM_OUTER, EM_INNER = 12, 60
EM_START_LAMBDA = 1.6     # Start des EM mit Kontrast deutlich über der Schwelle (aus dem unstrukturierten Start bleibt es hängen)
CALL_LIMIT = 2_000_000

# --- Experimente ----------------------------------------------------------------------------------------------------------------------------------
DETECT_LAMBDAS = (0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.4)
DIS_LAMBDAS = (-0.5, -1.0, -1.5, -2.0)
PLANTED_LAMBDAS = (0.0, 1.0, 2.0, 2.4)
EM_LAMBDAS = (0.0, 0.5, 1.0)
STABILITY_LAMBDAS = (0.0, 1.0, 2.0, 2.4)
STABILITY_RUNS = 8
RANDOM_SIZES = (100, 300, 600)
RANDOM_DEGREES = (3, 6, 10)

# --- Darstellung ---------------------------------------------------------------------------------------------------------------------------------
GROUP_COLORS = ("#4c78a8", "#e45756", "#54a24b", "#f58518", "#b279a2", "#72b7b2", "#eeca3b", "#9d755d", "#ff9da6", "#bab0ac")
COLOR_EDGE = "rgba(120,120,120,0.28)"
COLOR_BP = "#2F6B65"
COLOR_LEIDEN = "#c4623b"
COLOR_EM = "#6b5b95"
COLOR_LINE = "#4c78a8"


def lam_limits(q, c):
    """Zulässiger Bereich der Signalstärke bei Gruppenzahl q und mittlerem Grad c: c_out >= 0 gibt λ <= √c, c_in >= 0 gibt λ >= -√c / (q - 1)."""
    return -math.sqrt(c) / (q - 1), math.sqrt(c)


PRESETS = {
    "Knapp über der Schwelle (Voreinstellung)": {"q": DEFAULT_Q, "n": DEFAULT_N, "c": DEFAULT_C, "lam": 1.5, "seed": DEFAULT_SEED, "mode": MODE_KNOWN},
    "Reiner Zufall": {"q": 2, "n": DEFAULT_N, "c": DEFAULT_C, "lam": 0.0, "seed": DEFAULT_SEED, "mode": MODE_KNOWN},
    "Knapp unter der Schwelle": {"q": 2, "n": DEFAULT_N, "c": DEFAULT_C, "lam": 0.8, "seed": DEFAULT_SEED, "mode": MODE_KNOWN},
    "Starke Verbünde": {"q": 2, "n": DEFAULT_N, "c": DEFAULT_C, "lam": 2.4, "seed": DEFAULT_SEED, "mode": MODE_KNOWN},
    "Lieferstufen (disassortativ)": {"q": 2, "n": DEFAULT_N, "c": DEFAULT_C, "lam": -2.0, "seed": DEFAULT_SEED, "mode": MODE_KNOWN},
    "Drei Verbünde": {"q": 3, "n": DEFAULT_N, "c": DEFAULT_C, "lam": 2.0, "seed": DEFAULT_SEED, "mode": MODE_KNOWN},
    "Parameter unbekannt (EM)": {"q": 2, "n": DEFAULT_N, "c": DEFAULT_C, "lam": 1.5, "seed": DEFAULT_SEED, "mode": MODE_LEARNED},
}
PRESET_HELP = {
    "Knapp über der Schwelle (Voreinstellung)": "Zwei Verbünde, Signalstärke 1,5 (das 1,5-fache der Schwelle): das Blockmodell erreicht NMI rund 0,6, die Modularität rund 0,1.",
    "Reiner Zufall": "Kein Verbund im Netz (Signalstärke 0): trotzdem findet die Modularität Gemeinschaften mit hoher Modularität, das Blockmodell findet nichts.",
    "Knapp unter der Schwelle": "Signalstärke 0,8: unterhalb der Erkennbarkeitsgrenze ist nach der Theorie (dünner Grenzfall) keine Einteilung zu finden.",
    "Starke Verbünde": "Signalstärke 2,4 (das Maximum bei diesem mittleren Grad): das Blockmodell findet die Gruppen fast vollständig (NMI 0,99), die Modularität teilweise (NMI 0,67) und zerlegt sie oft in mehr Teile.",
    "Lieferstufen (disassortativ)": "Zulieferer arbeiten mit Herstellern, nicht untereinander: die Kanten liegen zwischen den Gruppen. Die Modularität sieht das in den gemessenen Fällen nicht als Gemeinschaft.",
    "Drei Verbünde": "Drei Gruppen, Signalstärke 2,0: die Schwelle gilt mit dem Faktor q, die Modularität liegt zurück (README: bei n = 600 NMI 0,60 gegen 0,92 für BP).",
    "Parameter unbekannt (EM)": "Wie die Voreinstellung, aber das Blockmodell lernt seine Parameter selbst (EM). Funktioniert bei Verbünden mit klarem Signal, nicht bei Lieferstufen und nicht knapp über der Schwelle.",
}
