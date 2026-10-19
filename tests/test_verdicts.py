"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import dataclasses

import sbm_constants as C
import sbm_evaluation as ev


def _analysis(lam=1.5, nmi_infer=0.6, nmi_leiden=0.14, k=12, q_found=0.4, q_planted=0.31, mode=C.MODE_KNOWN, edges=True):
    a = ev.analyse(ev.Settings(2, C.N_MIN, 6, lam, 3, mode))
    inst = a.inst
    if not edges:
        inst = dataclasses.replace(inst, u=inst.u[:0], v=inst.v[:0])
    inst = dataclasses.replace(inst, lam=lam)
    return dataclasses.replace(a, inst=inst, nmi_infer=nmi_infer, nmi_leiden=nmi_leiden, k_found=k, q_found=q_found, q_planted=q_planted)


def test_de_uses_a_decimal_comma():
    assert ev.de(1.26, 1) == "1,3" and ev.de(0.9487, 3) == "0,949" and ev.de(3, 0) == "3"


def test_above_threshold_is_strict_at_one():
    assert not ev.above_threshold(1.0) and ev.above_threshold(1.01) and ev.above_threshold(-1.5) and not ev.above_threshold(-0.99)


def test_main_verdict_success_when_the_block_model_is_clearly_better():
    kind, text = ev.verdict_main(_analysis())
    assert kind == "success"
    assert "λ = 1,50" in text and "(Parameter bekannt)" in text and "NMI 0,60" in text and "NMI 0,14" in text
    assert "12 Gemeinschaften mit Modularität 0,40" in text and "geplante Einteilung hat 0,31" in text


def test_main_verdict_thresholds_for_success():
    assert ev.verdict_main(_analysis(nmi_infer=0.5, nmi_leiden=0.39))[0] == "success"         # genau 0,5 und Abstand über 0,1
    assert ev.verdict_main(_analysis(nmi_infer=0.49, nmi_leiden=0.1))[0] == "info"             # unter 0,5
    assert ev.verdict_main(_analysis(nmi_infer=0.8, nmi_leiden=0.71))[0] == "info"             # Abstand nicht über 0,1


def test_main_verdict_below_the_threshold_is_a_warning_that_the_groups_are_chance():
    kind, text = ev.verdict_main(_analysis(lam=0.8, nmi_infer=0.0, nmi_leiden=0.02, k=14, q_found=0.39, q_planted=0.0))
    assert kind == "warning" and "unter der Erkennbarkeitsgrenze 1" in text and "λ = 0,80" in text and "sind Zufall" in text and "14 Gemeinschaften" in text
    assert ev.verdict_main(_analysis(lam=1.0, nmi_infer=0.0, nmi_leiden=0.0))[0] == "warning"      # genau 1 gilt noch als unter der Grenze
    assert ev.verdict_main(_analysis(lam=-0.5, nmi_infer=0.0, nmi_leiden=0.0))[0] == "warning"


def test_main_verdict_names_the_learned_mode_and_handles_a_network_without_edges():
    assert "Blockmodell (Parameter gelernt)" in ev.verdict_main(_analysis(mode=C.MODE_LEARNED))[1]
    kind, text = ev.verdict_main(_analysis(edges=False))
    assert kind == "warning" and "keine Kanten" in text


def test_detect_verdict_names_first_hit_leiden_and_the_floor():
    rows = [{"lam": 0.0, "bp": 0.0, "em": 0.0, "leiden": 0.01}, {"lam": 1.0, "bp": 0.02, "em": 0.0, "leiden": 0.03},
            {"lam": 1.5, "bp": 0.55, "em": 0.5, "leiden": 0.08}, {"lam": 2.0, "bp": 0.89, "em": 0.9, "leiden": 0.31}]
    kind, text = ev.verdict_detect(rows)
    assert kind == "info"
    assert "ab λ = 1,5 (NMI 0,55)" in text and "höchstens 0,02" in text and "bei λ = 1,5 nur 0,08" in text and "bei λ = 2,0 0,31 (Blockmodell 0,89)" in text


def test_detect_verdict_without_a_hit_is_a_warning():
    kind, text = ev.verdict_detect([{"lam": 0.0, "bp": 0.0, "em": 0.0, "leiden": 0.0}, {"lam": 0.8, "bp": 0.29, "em": 0.0, "leiden": 0.0}])
    assert kind == "warning" and "unter 0,3" in text
    assert ev.verdict_detect([{"lam": 1.5, "bp": 0.3, "em": 0.0, "leiden": 0.0}])[0] == "info"       # genau 0,3 zählt als Treffer


def test_random_verdict_names_range_and_maximum():
    rows = [{"n": 100, "c": 3, "q_median": 0.576, "q_min": 0.5, "q_max": 0.6, "k_median": 16.0}, {"n": 300, "c": 3, "q_median": 0.628, "q_min": 0.5, "q_max": 0.7, "k_median": 32.0},
            {"n": 300, "c": 10, "q_median": 0.286, "q_min": 0.2, "q_max": 0.3, "k_median": 11.0}]
    kind, text = ev.verdict_random(rows)
    assert kind == "warning" and "0,29 bis 0,63" in text and "n = 300 und mittlerem Grad 3 (0,63 mit etwa 32 Gemeinschaften)" in text


def _planted_rows():
    return [{"lam": 0.0, "q_found": 0.39, "q_planted": -0.003, "k": 14.5, "nmi_leiden": 0.012, "nmi_bp": 0.0},
            {"lam": 1.0, "q_found": 0.395, "q_planted": 0.203, "k": 13.0, "nmi_leiden": 0.031, "nmi_bp": 0.001},
            {"lam": 2.0, "q_found": 0.438, "q_planted": 0.408, "k": 9.5, "nmi_leiden": 0.322, "nmi_bp": 0.911},
            {"lam": 2.4, "q_found": 0.492, "q_planted": 0.491, "k": 4.5, "nmi_leiden": 0.67, "nmi_bp": 0.989}]


def test_planted_verdict_finds_the_case_where_leiden_beats_the_truth_in_modularity():
    kind, text = ev.verdict_planted(_planted_rows())
    assert kind == "warning"
    assert "Bei λ = 2,0" in text and "höherer Modularität (0,438) als die geplante (0,408)" in text and "NMI 0,32" in text and "zu 0,91" in text
    assert "Rauschen (λ = 0) kommt auf 0,390" in text and "λ = 1,0) auf 0,395" in text


def test_planted_verdict_without_such_a_case_is_info():
    rows = [r for r in _planted_rows() if r["lam"] in (0.0, 1.0)]
    kind, text = ev.verdict_planted(rows)
    assert kind == "info" and "nirgends" in text


def test_disassortative_verdict_names_the_strongest_row():
    rows = [{"lam": -1.5, "bp": 0.575, "em": 0.0, "leiden": 0.004}, {"lam": -2.0, "bp": 0.902, "em": 0.0, "leiden": 0.002}]
    kind, text = ev.verdict_disassortative(rows)
    assert kind == "info" and "Bei λ = -2,0" in text and "NMI 0,90" in text and "NMI 0,00; über alle Zeilen höchstens 0,00" in text
    assert ev.verdict_disassortative([])[0] == "warning"


def test_em_verdict_counts_the_runs_above_the_threshold():
    rows = [{"lam": 0.0, "hat_median": 1.63, "share_above": 0.625, "runs": 8, "imbalance": 0.355}, {"lam": 0.5, "hat_median": 1.4, "share_above": 0.5, "runs": 8, "imbalance": 0.3}]
    kind, text = ev.verdict_em(rows)
    assert kind == "warning" and "in 5 von 8 Läufen" in text and "λ̂ = 1,63" in text and "Gruppenungleichgewicht 0,35" in text
    assert ev.verdict_em([rows[1]])[0] == "info"


def test_stability_verdict_lists_all_signal_strengths():
    rows = [{"lam": 0.0, "nmi": 0.279}, {"lam": 1.0, "nmi": 0.284}, {"lam": 2.4, "nmi": 0.534}]
    kind, text = ev.verdict_stability(rows)
    assert kind == "info" and "λ = 0,0: 0,28, λ = 1,0: 0,28, λ = 2,4: 0,53" in text and "(0,28)" in text and "steigt der Wert (0,53)" in text
    assert ev.verdict_stability(rows[:1])[0] == "info"
