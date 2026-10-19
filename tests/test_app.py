"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Slider an seine Grenzen,
prüft Permalink, Modus-Umschalter, Begrenzung der Signalstärke und alle fünf Experimente; es darf kein Python-Fehler auftreten."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import sbm_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
EXPERIMENTS = ("detect", "random", "planted", "dis", "em", "stability")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_app_loads_without_exception():
    at = _fresh_app()
    assert _metric(at, "Signalstärke λ (wirksam)") == "1,50" and _metric(at, "NMI Blockmodell") == "0,60"
    assert not at.error


def test_every_preset_click_does_not_raise():
    for name in C.PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            slider.set_value(int(value) if isinstance(slider.value, int) else value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_all_sliders_at_their_minimum_and_all_at_their_maximum_do_not_raise():
    for edge in ("min", "max"):
        at = _fresh_app()
        for index in range(len(at.slider)):
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            slider.set_value(int(value) if isinstance(slider.value, int) else value)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("mode", C.MODES)
def test_both_inference_modes_render(mode):
    at = _fresh_app()
    at.selectbox(key="mode_select").set_value(mode).run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Blockmodell-Lauf" == m.label for m in at.metric)
    expected = "EM (12 × 60 Iterationen)" if mode == C.MODE_LEARNED else None
    value = _metric(at, "Blockmodell-Lauf")
    assert (value == expected) if expected else value.startswith("BP, ")


def test_the_signal_is_clipped_and_the_sidebar_says_so():
    at = _fresh_app()
    at.slider(key="c_slider").set_value(3).run(timeout=TIMEOUT)
    at.slider(key="lam_slider").set_value(3.0).run(timeout=TIMEOUT)
    assert not at.exception
    assert _metric(at, "Signalstärke λ (wirksam)") == "1,73"
    assert any("begrenzt auf" in c.value for c in at.sidebar.caption)


def test_permalink_values_are_clamped_and_unknown_modes_ignored():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n"] = "5"
    at.query_params["q"] = "9"
    at.query_params["lam"] = "-7"
    at.query_params["modus"] = "gibt es nicht"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="n_slider").value == C.N_MIN and at.slider(key="q_slider").value == C.Q_MAX and at.slider(key="lam_slider").value == C.LAM_MIN
    assert at.selectbox(key="mode_select").value == C.MODE_KNOWN


def test_permalink_selects_the_learned_mode():
    at = AppTest.from_file(APP_PATH)
    at.query_params["modus"] = C.MODE_LEARNED
    at.run(timeout=TIMEOUT)
    assert not at.exception and at.selectbox(key="mode_select").value == C.MODE_LEARNED


def test_every_experiment_runs_and_is_invalidated_by_a_setting_change():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    at.slider(key="n_slider").set_value(C.N_MIN).run(timeout=TIMEOUT)         # kleines Netz: schnell
    for name in EXPERIMENTS:
        at.button(key=f"run_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
    texts = " ".join(x.value for x in list(at.info) + list(at.warning) + list(at.success))
    for fragment in ("Blockmodell (bekannte Parameter) findet", "ohne jede Gemeinschaft", "der Wert allein unterscheidet beide nicht", "Stufen", "erfindet", "keine Diagnose"):
        assert fragment in texts, fragment
    at.slider(key="c_slider").set_value(7).run(timeout=TIMEOUT)
    assert not at.exception
    assert sum("neu starten" in i.value or "Neu starten" in i.value for i in at.info) >= 5      # alle von c abhängigen Ergebnisse sind ungültig
    assert any("ohne jede Gemeinschaft" in w.value for w in at.warning)                         # Teil a hängt nicht von c der Seitenleiste ab und bleibt stehen
