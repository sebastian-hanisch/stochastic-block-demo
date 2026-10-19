"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen."""

import sbm_constants as C
import sbm_evaluation as ev
import sbm_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert C.Q_MIN <= p["q"] <= C.Q_MAX and C.N_MIN <= p["n"] <= C.N_MAX and C.C_MIN <= p["c"] <= C.C_MAX, name
        assert C.LAM_MIN <= p["lam"] <= C.LAM_MAX and 0 <= p["seed"] <= C.SEED_MAX and p["mode"] in C.MODES, name


def test_every_preset_signal_is_feasible_without_clipping():
    for name, p in C.PRESETS.items():
        lo, hi = C.lam_limits(p["q"], p["c"])
        assert lo - 1e-9 <= p["lam"] <= hi + 1e-9, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Knapp über der Schwelle (Voreinstellung)"]) == ev.Settings()
    assert next(iter(C.PRESETS)) == "Knapp über der Schwelle (Voreinstellung)"


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_cover_both_signs_every_q_and_both_modes():
    ps = C.PRESETS.values()
    assert any(p["lam"] < 0 for p in ps) and any(p["lam"] == 0 for p in ps) and {p["q"] for p in ps} >= {2, 3}
    assert {p["mode"] for p in ps} == set(C.MODES)
    assert any(0 < p["lam"] < 1 for p in ps) and any(1 < p["lam"] < 2 for p in ps)       # unter und über der Schwelle
