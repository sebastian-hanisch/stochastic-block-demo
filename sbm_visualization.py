"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import itertools

import numpy as np
import plotly.graph_objects as go

import sbm_algorithm as A
import sbm_constants as C
from sbm_evaluation import de


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, bottom=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=bottom), legend=dict(orientation="h", y=-0.28), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def aligned_labels(true, pred, q):
    """Etiketten des Blockmodells so umbenannt, dass sie zu den geplanten Gruppen passen (beste Umbenennung), damit die Farben vergleichbar sind."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    best, best_perm = -1.0, tuple(range(q))
    for perm in itertools.permutations(range(q)):
        acc = float(np.mean(np.asarray(perm)[pred] == true)) if len(true) else 0.0
        if acc > best:
            best, best_perm = acc, perm
    return np.asarray(best_perm)[pred]


def build_network(inst, labels, height=270):
    """Netz mit festen Positionen, Knoten nach `labels` gefärbt (Farbtabelle wiederholt sich ab zehn Gruppen); gleicher Maßstab auf beiden Achsen."""
    n = inst.n
    xs, ys = [], []
    for a, b in zip(inst.u.tolist(), inst.v.tolist()):
        xs += [inst.pos[a][0], inst.pos[b][0], None]
        ys += [inst.pos[a][1], inst.pos[b][1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_EDGE, width=0.8), hoverinfo="skip", showlegend=False))
    size = 5 if n > 300 else 7
    colors = [C.GROUP_COLORS[int(g) % len(C.GROUP_COLORS)] for g in labels]
    fig.add_trace(go.Scatter(x=[p[0] for p in inst.pos], y=[p[1] for p in inst.pos], mode="markers",
                             marker=dict(size=size, color=colors, line=dict(color="rgba(40,40,40,0.45)", width=0.4)),
                             hovertext=[f"Knoten {i}: Gruppe {int(g)}" for i, g in enumerate(labels)], hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[-0.04, 1.04], y=[-0.04, 1.04], mode="markers", marker=dict(size=1, color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, autorange=True)
    fig.update_yaxes(visible=False, autorange=True, scaleanchor="x", scaleratio=1)
    return _base(fig, height=height)


def _threshold_lines(fig, lams):
    for x in (-1.0, 1.0):
        if min(lams) - 1e-9 <= x <= max(lams) + 1e-9:
            fig.add_vline(x=x, line=dict(color="rgba(120,120,120,0.9)", dash="dash"), annotation_text="Schwelle" if x > 0 or min(lams) >= 0 else "", annotation_position="top")


def build_detect(rows):
    """NMI zur geplanten Einteilung über die Signalstärke; gestrichelt die Erkennbarkeitsgrenze |λ| = 1."""
    xs = [r["lam"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["bp"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_BP, width=3), name="Blockmodell, Parameter bekannt (BP)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["em"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_EM, width=2, dash="dot"), name="Blockmodell, Parameter gelernt (EM)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["leiden"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_LEIDEN, width=3), name="Modularität (Leiden)"))
    _threshold_lines(fig, xs)
    fig.update_xaxes(title_text="Signalstärke λ")
    fig.update_yaxes(title_text="NMI zur geplanten Einteilung", range=[-0.02, 1.05])
    return _base(fig, height=360, bottom=60)


def build_random_q(rows):
    """Beste Leiden-Modularität in reinen Zufallsnetzen je Größe, gruppiert nach mittlerem Grad (Balken = Median, Strich = Minimum bis Maximum)."""
    sizes = sorted({r["n"] for r in rows})
    degrees = sorted({r["c"] for r in rows})
    shades = ["#9ecae1", "#4c78a8", "#1f3b5c"]
    fig = go.Figure()
    for i, c in enumerate(degrees):
        sel = [r for r in rows if r["c"] == c]
        fig.add_trace(go.Bar(x=[f"n = {r['n']}" for r in sel], y=[r["q_median"] for r in sel], name=f"mittlerer Grad {c}", marker_color=shades[i % len(shades)],
                             error_y=dict(type="data", symmetric=False, array=[r["q_max"] - r["q_median"] for r in sel], arrayminus=[r["q_median"] - r["q_min"] for r in sel]),
                             text=[de(r["q_median"], 2) for r in sel], textposition="outside", hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Modularität", range=[0, 0.85])
    return _base(fig, height=330, bottom=50)


def build_planted(rows):
    """Modularität der Leiden-Zerlegung gegen die der geplanten Einteilung über die Signalstärke."""
    xs = [r["lam"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["q_found"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_LEIDEN, width=3), name="Zerlegung von Leiden"))
    fig.add_trace(go.Scatter(x=xs, y=[r["q_planted"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_BP, width=3), name="geplante Einteilung"))
    fig.update_xaxes(title_text="Signalstärke λ")
    fig.update_yaxes(title_text="Modularität")
    return _base(fig, height=330, bottom=60)


def build_em(rows):
    """Gelernter Kontrast λ̂ des EM unterhalb der Schwelle (Median) mit der Grenze 1."""
    fig = go.Figure(go.Bar(x=[f"λ = {de(r['lam'], 1)}" for r in rows], y=[r["hat_median"] for r in rows], marker_color=C.COLOR_EM,
                           text=[f"{de(r['hat_median'], 2)} ({100 * r['share_above']:.0f} % über 1)" for r in rows], textposition="outside"))
    fig.add_hline(y=1.0, line=dict(color="rgba(120,120,120,0.9)", dash="dash"), annotation_text="Schwelle 1", annotation_position="top left")
    fig.update_yaxes(title_text="gelernter Kontrast λ̂ (Median)", rangemode="tozero")
    return _base(fig, height=320, bottom=30)


def build_stability(rows):
    """Mittlere paarweise NMI zwischen Leiden-Läufen mit verschiedenem Seed."""
    fig = go.Figure(go.Bar(x=[f"λ = {de(r['lam'], 1)}" for r in rows], y=[r["nmi"] for r in rows], marker_color=C.COLOR_LEIDEN,
                           text=[de(r["nmi"], 2) for r in rows], textposition="outside"))
    fig.update_yaxes(title_text="paarweise NMI der Läufe", range=[0, 1.15])
    return _base(fig, height=300, bottom=30)
