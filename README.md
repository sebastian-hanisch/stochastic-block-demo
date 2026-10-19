# Blockmodell: wann ist eine Gemeinschaft real? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-stochastic-block-demo.streamlit.app/)** -->

Zweites Stück der **Netzwerkanalyse-Reihe** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, im Struktur-Arm die **Fortsetzung der [Leiden-Demo](https://sebastianhanisch-leiden-demo.streamlit.app/)** (Kante `fix`; der Nullmodell-Gedanke kommt aus der [Demo zu Strukturkennzahlen und Nullmodellen](https://sebastianhanisch-strukturkennzahlen-demo.streamlit.app/)): Leiden optimiert die Modularität und liefert immer eine Zerlegung, aber ob dahinter eine Gemeinschaft steckt, sagt die Zahl nicht. Anders als die Fall-Demos zeigt diese Demo **ein** Verfahren an einem wachsenden Beispiel: ein Netz von Subunternehmern und Lieferanten (synthetisch) mit **geplanter** Einteilung, an dem die Modularität gegen ein **Blockmodell** (stochastic block model, SBM) mit Belief Propagation antritt. Zwei Ausprägungen: **Verbünde** (Kanten innerhalb der Gruppen) und **Lieferstufen** (Zulieferer arbeiten mit Herstellern, nicht untereinander: Kanten zwischen den Gruppen).

Der Regler **Signalstärke λ** = (c_in − c_out) / (q·√c) misst, wie deutlich die Gruppen sind, in Vielfachen der **Erkennbarkeitsgrenze** (Decelle, Krzakala, Moore, Zdeborová 2011, dünner Grenzfall, gleich große Gruppen): bei |λ| ≤ 1 gibt es für kein Verfahren eine Einteilung zu finden.

## Kernfrage

**Wann ist eine Gemeinschaft real, und wann zeigt ein Verfahren nur, was Zufall ohnehin hergibt?** Die Demo vergleicht auf demselben Netz drei Antworten: die Modularität (Leiden), das Blockmodell mit bekannten Parametern (Belief Propagation, BP) und das Blockmodell mit gelernten Parametern (EM), und misst jede an der bekannten Wahrheit (NMI, normalisierte gegenseitige Information: 0 = Raten, 1 = identisch).

## Befunde und Korrekturen gegenüber dem Plan

- **Konsens-Clustering und γ gestrichen (Nutzerentscheidung 2026-10-19).** Der Gesamtplan wollte sie als Gegenmittel gegen Überanpassung messen; sie bleiben bei `leiden-demo`.
- **Leiden statt Louvain, andere Größe.** Die Vormessung rechnete mit `networkx.louvain_communities` bei n = 1 000; die Demo nutzt den Leiden-Code aus `leiden-demo` (Kopie in `sbm_leiden.py`) bei n = 400 und höchstens n = 600. Die Zahlen unten sind die der Demo und weichen von der Vormessung ab (z. B. Leiden bei λ = 2,0: 0,32 gegen 0,31 bei Louvain; Stabilität bei der stärksten zulässigen Signalstärke 2,4 gleich 0,53, in der Vormessung bei 3,0 gleich 1,0: bei c = 6 ist λ = 3 nicht zulässig).
- **Karate-Club: andere Zahlen als geplant.** Die Vormessung nannte Louvain 0,60 und „Nabe gegen Rest“ (0,006); die endgültige Rechnung mit Leiden und dem EM der Demo ergibt vier Gemeinschaften (NMI 0,49) und eine Einteilung ohne Bezug zu den Vereinen (NMI 0,003). Die Aussage „Metadaten sind nicht die Wahrheit“ bleibt, die Randnotiz nennt die neuen Zahlen. Die Karate-Daten liegen nicht im Repo (Test nur mit `networkx`).
- **EM-Vorzeichen als Grenze benannt (Nutzerentscheidung).** Das EM startet assortativ und findet Lieferstufen nicht; es gibt keinen zweiten Start.
- **Plan-Vermutung nicht eingetreten:** „das Inferenzverfahren ist auf kleinen Netzen nicht besser als Modularität“. Bei n = 400 ist BP oberhalb der Schwelle deutlich besser.
- **Zwei Fallen beim Bau:** BP kippte zunächst alle Knoten synchron in eine Gruppe (NMI 0), weil das Außenfeld h mit den ungedämpften Randverteilungen gerechnet wurde; es wird jetzt mit gedämpften gerechnet. Und das EM bleibt aus einem unstrukturierten Start im unstrukturierten Zustand hängen; es startet mit einem Kontrast von λ = 1,6.

## Modell

q gleich große Gruppen, zwei Knoten sind mit Wahrscheinlichkeit c_ab / n verbunden (symmetrisch: c_in innerhalb, c_out zwischen den Gruppen), mittlerer Grad c = (c_in + (q − 1)·c_out) / q bleibt bei jeder Signalstärke gleich. Bei vorgegebener Signalstärke wird λ auf den zulässigen Bereich begrenzt (c_in, c_out ≥ 0: −√c / (q − 1) ≤ λ ≤ √c; die App zeigt die wirksamen Werte). **BP** schätzt je Knoten die Wahrscheinlichkeit jeder Gruppe aus den Nachrichten seiner Nachbarn (mit Außenfeld für die Nicht-Kanten); **EM** schätzt Parameter und Einteilung abwechselnd (12 äußere Läufe zu je 60 BP-Iterationen). Formeln: 📐-Expander der App.

## Methodik

| Baustein | Inhalt |
|---|---|
| **Sampler** | Blockmodell mit gleich großen Gruppen, feste Zufalls-Seeds, Positionen für die Anzeige per Kräfte-Entwurf |
| **Leiden** | Kopie aus `leiden-demo` (dichte Matrix), Modularität mit γ = 1 |
| **BP** | Belief Propagation mit bekannten Parametern, Dämpfung 0,5, Außenfeld mit gedämpften Randverteilungen, höchstens 300 Iterationen |
| **EM** | Parameter lernen, assortativ gestartet (λ = 1,6), bis zu q = 4 Gruppen |
| **Experiment 1** | NMI über die Signalstärke für BP, EM und Leiden |
| **Experiment 2** | beste Modularität in Zufallsnetzen ohne Gemeinschaft; Modularität der Leiden-Zerlegung gegen die der geplanten Einteilung |
| **Experiment 3** | Lieferstufen (negative Signalstärke) |
| **Experiment 4** | was das EM unterhalb der Grenze lernt |
| **Experiment 5** | Stabilität von Leiden über Zufalls-Seeds |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Netze der Messreihen: die acht festen Netze `SWEEP_SEEDS` je Zelle, q = 2, n = 400, c = 6, wo nichts anderes steht; NMI zur geplanten Einteilung, Median.

| Frage | Befund |
|---|---|
| **Ab wann findet das Blockmodell etwas?** (q = 2, n = 400, c = 6) | Bis zum **1,2-Fachen der Grenze** liegt BP bei höchstens 0,01 (kein Verfahren findet etwas). **λ = 1,5: BP 0,61**, EM 0,37, Leiden 0,12; **λ = 2,0: BP 0,91**, EM 0,90, Leiden 0,32; **λ = 2,4: BP 0,99**, EM 0,98, Leiden 0,67. Bei c = 3: λ = 1,2 BP 0,22, Leiden 0,12; λ = 1,5 BP 0,66, Leiden 0,23 |
| **Gilt die Grenze auch bei drei Gruppen?** (q = 3, n = 600, c = 6) | BP bis λ = 1,0 höchstens 0,02, bei 1,3: **0,41** (EM nur **0,11**, Leiden 0,12), bei 1,6: 0,72 (EM 0,72, Leiden 0,25), bei 2,0: 0,92 (EM 0,92, Leiden 0,60). Das EM verfehlt die Gruppen nahe der Grenze |
| **Wie hoch wird die Modularität ohne jede Gemeinschaft?** (beste Leiden-Modularität, Zufallsnetze, vier bis fünf Mal acht Netze je Zelle) | Median **0,27 bis 0,64**: n = 100: 0,58 / 0,37 / 0,27, n = 300: 0,63 / 0,40 / 0,29, n = 600: 0,64 / 0,40 / 0,28 bei mittlerem Grad 3 / 6 / 10; die Zahl der Gemeinschaften wächst mit n (bis 50 bei n = 600, c = 3) |
| **Unterscheidet der Wert Q Rauschen von Signal?** | Nein: Q der Leiden-Zerlegung **0,390 bei λ = 0** und **0,395 bei λ = 1,0** (die geplante Einteilung hat dort −0,003 bzw. 0,203) |
| **Ist das Modularitätsoptimum die Wahrheit?** | Nein: bei **λ = 2,0** findet Leiden eine Zerlegung mit **höherer Modularität (0,438) als die geplante (0,408)**, die der Wahrheit nur zu NMI 0,32 ähnelt, das Blockmodell findet sie zu 0,91. Bei λ = 2,4 stimmen beide überein (0,492 gegen 0,491, etwa 4,5 Gemeinschaften, NMI 0,67) |
| **Finden Verfahren Lieferstufen?** (q = 2, c = 6) | **BP: 0,58 bei λ = −1,5, 0,90 bei −2,0** (bis −1,0 höchstens 0,03). **Leiden findet sie nie** (höchstens 0,01 bei allen Stärken), das EM auch nicht (assortativ gestartet) |
| **Erfindet das EM Struktur?** (Zwei Gruppen, ohne und mit schwachem Signal) | Im Netz **ohne** Gemeinschaft lernt das EM in **5 von 8** Läufen einen Kontrast über der Schwelle (Median λ̂ 1,63, Gruppenungleichgewicht 0,36: eine kleine dichte Gruppe); ebenso 5 von 8 bei λ = 0,5 (Median 1,40) und bei λ = 1,0 (Median 1,30) |
| **Zeigt die Stabilität über Seeds, ob Struktur da ist?** (mittlere paarweise NMI von acht Leiden-Läufen) | Nein: 0,28 bei λ = 0, 0,28 bei 1,0, 0,35 bei 2,0 (dort ist die Einteilung für BP zu 0,91 erkennbar), erst bei 2,4 steigt der Wert auf 0,53 |
| **Karate-Club** (34 Knoten, 78 Kanten, Metadaten: Vereine) | Leiden: **4 Gemeinschaften**, NMI zu den Vereinen **0,49**; Blockmodell mit gelernten Parametern (q = 2): NMI **0,003**. Das Netz ist zu klein und zu dicht für die dünne Näherung; Vereine sind Metadaten, nicht „die“ Wahrheit |
| **Ist BP korrekt?** | Auf 30 Zufallsbäumen (bis 8 Knoten, q = 2 und 3) stimmen die Randverteilungen ohne Außenfeld bis auf 10⁻⁸ mit der Aufzählung aller Einteilungen überein (`tests/test_oracle_blockmodell.py`) |
| **Standardnetz** (q = 2, n = 400, c = 6, λ = 1,5, Seed 35) | 1 189 Kanten; BP konvergiert in 98 Iterationen, NMI 0,60; Leiden findet 12 Gemeinschaften mit Modularität 0,40 (geplante Einteilung 0,31), NMI 0,14 |

## Ehrliche Grenzen

- **Die Grenze |λ| = 1 gilt für gleich große Gruppen mit symmetrischen Affinitäten im dünnen Grenzfall** (n → ∞). Bei n ≤ 600 ist sie verschmiert: nahe λ = 1 streuen die Ergebnisse von Netz zu Netz stark, und BP erreicht dort oft die 300 Iterationen nicht. Die Messreihen zeigen deshalb Kurven mit Streuung, keinen Schwellenpunkt.
- **BP kennt die wahren Parameter und q.** Das ist der Idealfall; in der Praxis fehlt beides. Das EM schätzt die Parameter, aber nur assortativ gestartet, und scheitert bei q = 3 nahe der Grenze und bei Lieferstufen. Eine Modellwahl (wie viele Gruppen?) gibt es nicht.
- **Das EM erfindet Struktur**, wo keine ist (Experiment 4); der Test ist ein Befund für diese Rechnung (feste Läufe, acht Netze), keine allgemeine Aussage über Inferenz.
- **Kein Grad-korrigiertes Modell** (Hubs würden als eigene Gruppe gelesen) und **nur q ≤ 4**: ab q = 5 gibt es eine „schwere“ Phase, die die Demo nicht zeigt.
- **Synthetische Netze mit bekannter Wahrheit.** In echten Netzen gibt es keine „wahre“ Einteilung (Peel u. a. 2017).
- **NMI und die Zahl der Läufe:** Mediane über acht (Experimente live vier) feste Netze; einzelne Netze streuen.

## Verifikation

- **Kern:** Hand-Rechnung (Affinitäten, NMI, beste Umbenennung, BP auf einer Kante: 7/9, EM-Update, Modularität zweier Dreiecke), Randfälle (n = q, Netze ohne Kanten, Grenzen inklusive) in `tests/test_algorithm.py` und `tests/test_edge_cases.py`.
- **Orakel** (`tests/test_oracle_blockmodell.py`): BP gegen die Aufzählung aller Einteilungen auf Zufallsbäumen, Modularität gegen `networkx`, NMI gegen `scikit-learn`, beste Umbenennung gegen die Hungarian-Methode (`scipy`), Sampler-Erwartungswerte, Parameterwiederherstellung des EM, Erkennbarkeit als Eigenschaftstest mit Band.
- **Netze** (`tests/test_scenario.py`): gültige Netze in allen Ecken der Regler, Determinismus je Seed, Ergebnisse unabhängig vom Hash-Seed (zwei Prozesse), Begrenzung der Signalstärke.
- **Verdikte** als reine Funktionen (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an handgesetzten Werten; Dezimalkomma.
- **Presets und Layout:** vollständig, gültige Werte, Standard-Preset = Standardeinstellung; Schnellstart oben, aktives Preset hervorgehoben (`tests/test_presets.py`, `tests/test_layout.py`).
- **Oberfläche:** AppTest (jeder Preset-Knopf, jeder Regler an Minimum und Maximum, alle Minima und alle Maxima, beide Modi, Begrenzung der Signalstärke, Permalink, alle Experimente mit Ungültigwerden bei geänderten Einstellungen), Footer- und Laufzeit-Import-Test.
- **Claims:** jede Zahl dieser README in `tests/test_claims.py`.
- **Mutationsprüfung** (`tools/mutate_demo.py`, Zustand in `tests/mutation_state.json`) für den Kern (Sampler, BP, EM, Überlappungsmaße), die Auswertung und die Netze; die wortgleiche Kopie `sbm_leiden.py` ist ausgenommen (Tests im Ursprung `leiden-demo`): von 107 Mutationsstellen 100 geprüft und gefangen, keine überlebte, 7 führen zu Endlosschleifen oder Zeitüberschreitung (zählt nicht als gefangen). Verdikte haben zusätzlich eigene Tests mit handgesetzten Werten.

## Tests

Siehe `tests/`; alle Tests laufen mit `pytest tests/ -v` (die Claims-Tests rechnen rund anderthalb Minuten).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: drei Netzbilder, Kernfrage, fünf Experimente, Grenzen, Formeln |
| `sbm_algorithm.py` | Kern: Sampler, BP, EM, NMI, beste Umbenennung, Modularität, Leiden-Anbindung |
| `sbm_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `sbm_evaluation.py` | Analyse, Sweeps, Verdikte (reine Funktionen) |
| `sbm_layout.py` | Fester Kopf (Stylesheet) |
| `sbm_leiden.py` | Leiden-Algorithmus, wortgleiche Kopie aus `leiden-demo` (Tests im Ursprung) |
| `sbm_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `sbm_scenario.py` | Netz mit geplanter Einteilung und Anordnung für die Anzeige |
| `sbm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx, scikit-learn (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Hand-Rechnung, Orakel, Randfälle, Netze, Verdikte, Presets, Layout, AppTest, Claims, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Konsens-Clustering und γ** (bleibt bei `leiden-demo`).
- **Zweiter EM-Start** für Lieferstufen und Modellwahl über die Zahl der Gruppen.
- **q ≥ 5** (schwere Phase) und ungleich große Gruppen.
- **Grad-korrigiertes Blockmodell** (Karrer–Newman): spätere Stücke der Reihe.
- **Echte Netze** und Karate-Club-Daten im Repo (nur als Randnotiz mit Test über `networkx`).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
