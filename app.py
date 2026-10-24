"""Blockmodell: wann ist eine Gemeinschaft real? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweites Stück der Reihe „Netzwerkanalyse“: ein Netz mit geplanter Einteilung (Blockmodell), die Modularität (Leiden aus leiden-demo) gegen die Inferenz
(Belief Propagation mit bekannten Parametern, EM mit gelernten) und die Frage, ab welcher Signalstärke überhaupt etwas zu finden ist.
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import sbm_algorithm as A
import sbm_constants as C
from sbm_evaluation import (
    Settings,
    analyse,
    de,
    detect_curve,
    disassortative_rows,
    em_invention_rows,
    planted_vs_found,
    random_modularity_rows,
    stability_rows,
    verdict_detect,
    verdict_disassortative,
    verdict_em,
    verdict_main,
    verdict_planted,
    verdict_random,
    verdict_stability,
)
from sbm_layout import dunkles_design, kopf_css
from sbm_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from sbm_visualization import (
    aligned_labels,
    build_detect,
    build_em,
    build_network,
    build_planted,
    build_random_q,
    build_stability,
)

st.set_page_config(page_title="Blockmodell – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _detect(q, n, c):
    return detect_curve(q, n, c)


@st.cache_data(show_spinner=False)
def _random():
    return random_modularity_rows()


@st.cache_data(show_spinner=False)
def _planted(q, n, c):
    return planted_vs_found(q, n, c)


@st.cache_data(show_spinner=False)
def _dis(n, c):
    return disassortative_rows(n, c)


@st.cache_data(show_spinner=False)
def _em(n, c):
    return em_invention_rows(n, c)


@st.cache_data(show_spinner=False)
def _stability(q, n, c):
    return stability_rows(q, n, c)


def experiment(name, label, key, compute, show, hint):
    """Teure Zusatzansicht erst auf Klick; das Ergebnis bleibt nur für genau diese Einstellungen stehen (Session State)."""
    state_key = f"exp_{name}"
    if st.button(label, key=f"run_{name}", help=hint):
        with st.spinner("Rechne …"):
            result = compute()
        st.session_state[state_key] = (key, result)
    state = st.session_state.get(state_key)
    if state is None:
        st.caption(f"Noch nicht gerechnet. {hint}")
        return
    if state[0] != key:
        st.info("Die Einstellungen haben sich seit der letzten Rechnung geändert. Bitte neu starten.")
        return
    show(state[1])


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in sbm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧱 Blockmodell: wann ist eine Gemeinschaft real?")
st.markdown(
    """
**Zweites Stück** der Reihe „Netzwerkanalyse“, im Struktur-Arm die **Fortsetzung der [Leiden-Demo](https://sebastianhanisch-leiden-demo.streamlit.app/)**: Leiden optimiert die Modularität und liefert immer eine Zerlegung, aber ob dahinter eine Gemeinschaft steckt, sagt die Zahl nicht. Hier gibt es ein Netz von Subunternehmern und Lieferanten mit **geplanter** Einteilung (die Wahrheit ist bekannt), und ein **Blockmodell** (SBM, Belief Propagation nach Decelle, Krzakala, Moore und Zdeborová) tritt gegen die Modularität an. Anders als die Fall-Demos zeigt diese Demo **ein** Verfahren an einem wachsenden Beispiel.

Die Kernfrage: **Wann ist eine Gemeinschaft real, und wann zeigt ein Verfahren nur, was Zufall ohnehin hergibt?** Der Regler **Signalstärke λ** misst, wie deutlich die Gruppen sind, in Vielfachen der Erkennbarkeitsgrenze: unter 1 ist (im dünnen Grenzfall mit gleich großen, symmetrischen Gruppen) keine Einteilung zu finden.

Weiter unten: fünf Experimente (**🔬**), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Blockmodell-Inferenz", expanded=True):
    st.markdown(
        """
1. **Geplante Einteilung:** *q* gleich große Gruppen; zwei Betriebe sind mit Wahrscheinlichkeit *c*<sub>in</sub>/*n* verbunden, wenn sie in derselben Gruppe liegen, sonst mit *c*<sub>out</sub>/*n*. Der mittlere Grad *c* bleibt fest, die **Signalstärke** λ = (*c*<sub>in</sub> − *c*<sub>out</sub>) / (*q*·√*c*) verschiebt Kanten in die Gruppen (λ > 0, Verbünde) oder zwischen die Gruppen (λ < 0, **Lieferstufen**).
2. **Modularität (Leiden):** sucht die Zerlegung mit der größten Modularität, ohne eine Wahrheit zu kennen; sie meldet immer Gemeinschaften, auch im Zufallsnetz.
3. **Blockmodell mit bekannten Parametern (BP):** Belief Propagation schätzt für jeden Knoten die Wahrscheinlichkeit jeder Gruppe aus den Nachrichten seiner Nachbarn. Ist das Signal unter der Grenze, liefert das Verfahren **kein Ergebnis über Zufall** (NMI nahe 0), statt etwas zu erfinden; das EM mit gelernten Parametern kann dagegen Struktur erfinden.
4. **Blockmodell mit gelernten Parametern (EM):** die Parameter sind unbekannt und werden abwechselnd mit der Einteilung geschätzt. Der Start ist assortativ (Kanten innerhalb der Gruppen).
5. **Überlappung:** die normalisierte gegenseitige Information (**NMI**) zwischen gefundener und geplanter Einteilung, 0 = Raten, 1 = identisch.
        """,
        unsafe_allow_html=True,
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Netz zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    q = st.slider("Gruppen q", *bounds("q_slider"), key="q_slider", help="Zahl der gleich großen Gruppen. Bei q ≥ 5 gibt es eine schwere Phase, die diese Demo nicht zeigt.")
    n = st.slider("Knoten n", *bounds("n_slider"), key="n_slider", help="Zahl der Betriebe.")
    c = st.slider("Mittlerer Grad c", *bounds("c_slider"), key="c_slider", help="Wie viele Partner ein Betrieb im Mittel hat (bleibt bei jeder Signalstärke gleich).")
    lam = st.slider("Signalstärke λ", *bounds("lam_slider"), step=C.LAM_STEP, key="lam_slider",
                    help="Vielfaches der Erkennbarkeitsgrenze. Positiv: Verbünde (Kanten innerhalb der Gruppen). Negativ: Lieferstufen (Kanten zwischen den Gruppen). Auf den zulässigen Bereich begrenzt.")
    seed = st.number_input("Zufalls-Seed des Netzes", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Netz würfeln", width="stretch", on_click=randomize_seed)
    st.markdown("**Inferenz**")
    mode = st.selectbox("Blockmodell", C.MODES, key="mode_select",
                        help="Bekannte Parameter: BP kennt die wahren Affinitäten (der Idealfall). Gelernt: das EM schätzt sie, startet aber assortativ und kann Lieferstufen nicht finden.")
    cin, cout, lam_eff = A.affinities(int(q), int(c), float(lam))
    lo, hi = C.lam_limits(int(q), int(c))
    st.caption(f"Wirksam: λ = {de(lam_eff, 2)}, c_in = {de(cin, 2)}, c_out = {de(cout, 2)}"
               + (f" (begrenzt auf [{de(lo, 2)}; {de(hi, 2)}], weil c_in und c_out nicht negativ sein können)" if abs(lam_eff - float(lam)) > 1e-9 else ""))

sync_query_params({"q_slider": int(q), "n_slider": int(n), "c_slider": int(c), "lam_slider": float(lam), "seed_input": int(seed), "mode_select": mode})

settings = Settings(int(q), int(n), int(c), float(lam), int(seed), mode)
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Drei Blicke auf dasselbe Netz")
m1, m2, m3 = st.columns(3)
m1.metric("Knoten / Kanten", f"{a.inst.n} / {a.inst.m}")
m2.metric("Signalstärke λ (wirksam)", de(a.inst.lam, 2), delta="über der Grenze 1" if abs(a.inst.lam) > 1 else "unter der Grenze 1", delta_color="off")
m3.metric("Tatsächlicher mittlerer Grad", de(a.inst.mean_degree, 2))
columns = st.columns(3)
columns[0].markdown("**Geplante Einteilung** (die Wahrheit)")
columns[0].plotly_chart(build_network(a.inst, a.inst.labels), width="stretch", key="net_planted")
columns[1].markdown(f"**Modularität (Leiden)**: {a.k_found} Gemeinschaften")
columns[1].plotly_chart(build_network(a.inst, a.leiden_labels), width="stretch", key="net_leiden")
columns[2].markdown(f"**Blockmodell** ({'bekannt' if a.mode == C.MODE_KNOWN else 'gelernt'})")
columns[2].plotly_chart(build_network(a.inst, aligned_labels(a.inst.labels, a.infer_labels, a.inst.q)), width="stretch", key="net_infer")
k1, k2, k3, k4 = st.columns(4)
k1.metric("NMI Leiden", de(a.nmi_leiden, 2))
k2.metric("NMI Blockmodell", de(a.nmi_infer, 2))
k3.metric("Q Leiden / geplant", f"{de(a.q_found, 2)} / {de(a.q_planted, 2)}")
k4.metric("Blockmodell-Lauf", "EM (12 × 60 Iterationen)" if a.infer_converged is None else f"BP, {a.infer_iterations} Iterationen", delta=None if a.infer_converged is None else ("konvergiert" if a.infer_converged else "nicht konvergiert"), delta_color="off")
st.caption("Farben: links die geplanten Gruppen, in der Mitte je Gemeinschaft von Leiden eine eigene Farbe, rechts die Gruppen des Blockmodells, passend zu den geplanten Farben umbenannt. Die Anordnung der Knoten ist ein reiner Kräfte-Entwurf des Netzes.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Findet ein Verfahren hier eine Gemeinschaft, die es wirklich gibt?")
kind, text = verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Ab welcher Signalstärke findet ein Verfahren etwas?")
st.caption(f"Median über {len(C.LIVE_SEEDS)} feste Netze je Signalstärke, q = {settings.q}, n = {settings.n}, c = {settings.c} (die Einstellungen der Seitenleiste). Die gestrichelte Linie ist die Erkennbarkeitsgrenze |λ| = 1.")


def show_detect(rows):
    st.plotly_chart(build_detect(rows), width="stretch", key="detect_chart")
    kind_, text_ = verdict_detect(rows)
    getattr(st, kind_)(text_)


experiment("detect", "Kurve rechnen", (settings.q, settings.n, settings.c), lambda: _detect(settings.q, settings.n, settings.c), show_detect,
           "Rechnet für acht Signalstärken je vier Netze mit drei Verfahren, einige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 2: Wie hoch wird die Modularität ohne jede Gemeinschaft, und ist sie die Wahrheit?")
st.caption("Teil a: Netze ohne jede Gemeinschaft (λ = 0), beste Leiden-Modularität je Größe und mittlerem Grad. Teil b: bei steigender Signalstärke die Modularität der Leiden-Zerlegung gegen die der geplanten Einteilung.")


def show_random(rows):
    st.plotly_chart(build_random_q(rows), width="stretch", key="random_chart")
    kind_, text_ = verdict_random(rows)
    getattr(st, kind_)(text_)


def show_planted(rows):
    st.plotly_chart(build_planted(rows), width="stretch", key="planted_chart")
    st.dataframe([{"λ": de(r["lam"], 1), "Q Leiden": de(r["q_found"], 3), "Q geplant": de(r["q_planted"], 3), "Gemeinschaften (Leiden)": f"{r['k']:.0f}",
                   "NMI Leiden": de(r["nmi_leiden"], 2), "NMI Blockmodell": de(r["nmi_bp"], 2)} for r in rows], hide_index=True, width="stretch")
    kind_, text_ = verdict_planted(rows)
    getattr(st, kind_)(text_)


experiment("random", "Teil a rechnen", "random", _random, show_random, "Rechnet neun Kombinationen aus Größe und Grad mit je vier Netzen, einige Sekunden.")
experiment("planted", "Teil b rechnen", (settings.q, settings.n, settings.c), lambda: _planted(settings.q, settings.n, settings.c), show_planted,
           "Rechnet vier Signalstärken mit je vier Netzen.")

st.markdown("---")

st.subheader("🔬 Experiment 3: Lieferstufen – wenn die Kanten zwischen den Gruppen liegen")
st.caption(f"Zwei Gruppen mit Kanten vorwiegend zwischen den Gruppen (negative Signalstärke), n = {settings.n}, c = {settings.c}. Das EM startet assortativ und kann sie nicht finden.")


def show_dis(rows):
    if rows:
        st.plotly_chart(build_detect(rows), width="stretch", key="dis_chart")
    kind_, text_ = verdict_disassortative(rows)
    getattr(st, kind_)(text_)


experiment("dis", "Lieferstufen rechnen", (settings.n, settings.c), lambda: _dis(settings.n, settings.c), show_dis, "Rechnet bis zu vier negative Signalstärken mit je vier Netzen.")

st.markdown("---")

st.subheader("🔬 Experiment 4: Erfindet das EM Struktur, wo keine ist?")
st.caption(f"Zwei Gruppen, n = {settings.n}, c = {settings.c}: was das EM unterhalb und an der Grenze lernt (gelernter Kontrast λ̂ über 1 hieße „Gruppen erkannt“).")


def show_em(rows):
    st.plotly_chart(build_em(rows), width="stretch", key="em_chart")
    kind_, text_ = verdict_em(rows)
    getattr(st, kind_)(text_)


experiment("em", "EM-Lauf rechnen", (settings.n, settings.c), lambda: _em(settings.n, settings.c), show_em, "Rechnet drei Signalstärken mit je vier Netzen.")

st.markdown("---")

st.subheader("🔬 Experiment 5: Zeigt die Stabilität über Zufalls-Seeds, ob Struktur da ist?")
st.caption(f"Mittlere paarweise NMI zwischen {C.STABILITY_RUNS} Leiden-Läufen mit verschiedenem Seed auf demselben Netz (q = {settings.q}, n = {settings.n}, c = {settings.c}).")


def show_stability(rows):
    st.plotly_chart(build_stability(rows), width="stretch", key="stability_chart")
    kind_, text_ = verdict_stability(rows)
    getattr(st, kind_)(text_)


experiment("stability", "Stabilität rechnen", (settings.q, settings.n, settings.c), lambda: _stability(settings.q, settings.n, settings.c), show_stability,
           "Rechnet vier Signalstärken mit je vier Netzen und acht Läufen.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gleich große Gruppen, symmetrische Affinitäten** | Die Erkennbarkeitsgrenze (Betrag von λ gleich 1) gilt nur für diesen Fall im dünnen Grenzfall (n → ∞, alle Grade von der Größe des Mittels). Bei ungleichen Gruppen oder Affinitätsmatrizen gibt es andere Grenzen; das EM lernt ungleiche Gruppen und **erfindet** dabei Struktur (Experiment 4). | – |
| **Bekannte Parameter und bekannte Gruppenzahl** | Der Idealfall BP kennt die wahren Affinitäten und q. In der Praxis fehlen beide; das EM schätzt die Parameter, aber nur assortativ gestartet, und scheitert bei q = 3 nahe der Grenze. Eine Modellwahl (wie viele Gruppen?) gibt es hier nicht. | – |
| **Endliche Größe** | Die Grenze ist bei n ≤ 600 verschmiert: nahe λ = 1 streuen die Ergebnisse von Netz zu Netz stark, und BP braucht dort die meisten Iterationen oder erreicht die Grenze von 300 nicht. | – |
| **Kein Grad-korrigiertes Modell** | Hubs (wenige Betriebe mit sehr vielen Partnern) stören das einfache Blockmodell: es würde sie als eigene Gruppe lesen. | spätere Stücke der Reihe |
| **Nur q ≤ 4** | Ab q = 5 gibt es eine „schwere“ Phase: eine Einteilung ist im Prinzip erkennbar, aber von einem zufälligen Start aus kaum zu finden. Sie ist nicht Teil der Demo. | – |
| **Synthetische Netze mit bekannter Wahrheit** | In echten Netzen gibt es keine „wahre“ Einteilung; Metadaten (z. B. Vereinszugehörigkeit) sind nicht die Wahrheit (Peel u. a. 2017). Randnotiz Karate-Club: siehe unten. | – |
"""
)
st.markdown(
    "*Randnotiz Zachary-Karate-Club (34 Knoten, zwei Vereine):* "
    "Die Modularität zerlegt das Netz in vier Gemeinschaften (NMI zu den Vereinen 0,49); das Blockmodell mit gelernten Parametern (q = 2) findet eine Einteilung, die mit den Vereinen nichts zu tun hat (NMI 0,003). "
    "Das Netz ist zu klein und zu dicht für die dünne Näherung; Vereine sind Metadaten und nicht „die“ Wahrheit, jede Antwort gilt für ihre Frage."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Blockmodell.** Jeder der $n$ Knoten gehört zu einer von $q$ Gruppen mit Anteil $n_a$; zwei Knoten der Gruppen $a$ und $b$ sind mit Wahrscheinlichkeit $c_{ab}/n$ verbunden. Im symmetrischen Fall ist $c_{aa} = c_\text{in}$, $c_{ab} = c_\text{out}$ für $a \ne b$ und der mittlere Grad

$$c = \frac{c_\text{in} + (q-1)\, c_\text{out}}{q}$$

**Signalstärke und Erkennbarkeitsgrenze** (Decelle, Krzakala, Moore, Zdeborová 2011, dünner Grenzfall, gleich große Gruppen): mit

$$\lambda = \frac{c_\text{in} - c_\text{out}}{q \sqrt{c}}$$

ist die geplante Einteilung für BP auffindbar, wenn

$$|c_\text{in} - c_\text{out}| > q \sqrt{c} \iff |\lambda| > 1$$

**Belief Propagation.** Für jede gerichtete Kante $i \to j$ gilt die Nachricht

$$\psi^{i \to j}_{t} \propto n_t \, e^{-h_t} \prod_{k \in \partial i \setminus j} \sum_{s} c_{st} \, \psi^{k \to i}_{s}$$

mit dem Außenfeld

$$h_t = \frac{1}{n} \sum_{k} \sum_{s} c_{st} \, \psi^{k}_{s}$$

und die Randverteilung eines Knotens enthält alle Nachbarn. Auf Bäumen ist BP ohne Außenfeld exakt.

**EM.** Mit den Kantenrandverteilungen $\psi^{ij}_{ab} \propto c_{ab} \, \psi^{i \to j}_a \, \psi^{j \to i}_b$ und den erwarteten Kantenzahlen $E_{ab}$ zwischen den Gruppen sind die neuen Parameter

$$c_{ab} = \frac{E_{ab}}{n \, n_a n_b} \quad (a \ne b), \qquad c_{aa} = \frac{2 E_{aa}}{n \, n_a^2}$$

**Modularität.** Für eine Zerlegung $g$ ist

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(g_i, g_j)$$

**Überlappung (NMI)** zwischen den Einteilungen $X$ und $Y$:

$$\mathrm{NMI}(X, Y) = \frac{I(X; Y)}{\tfrac{1}{2} \left( H(X) + H(Y) \right)}$$

**Literatur.**
- Decelle, A., Krzakala, F., Moore, C., Zdeborová, L. (2011): *Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications.* Physical Review E 84, 066106.
- Guimerà, R., Sales-Pardo, M., Amaral, L. A. N. (2004): *Modularity from fluctuations in random graphs and complex networks.* Physical Review E 70, 025101.
- Peixoto, T. P. (2021/2023): *Descriptive vs. inferential community detection: pitfalls, myths and half-truths.* arXiv:2112.00183, Cambridge Elements 2023.
- Peel, L., Larremore, D. B., Clauset, A. (2017): *The ground truth about metadata and community detection in networks.* Science Advances 3(5), e1602548.
- Traag, V. A., Waltman, L., van Eck, N. J. (2019): der Leiden-Algorithmus; hier der Code aus der Leiden-Demo (Angaben zur Originalarbeit dort).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
