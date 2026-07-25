# Design-Spec: `tabular-agent` — MCP-Server für agentengetriebene Tabular-ML

- **Datum:** 2026-07-21
- **Status:** Entwurf zur Abnahme
- **Inspiration:** W&B-Report *„Tabular ML didn't die. It was just waiting for agents"* (MLE-bench / NOMAD)
- **Autor:** Ulf Mertens (mit Claude)

---

## 1. Kontext & Ziel

Der W&B-Report zeigt, dass ein Frontier-LLM mit einem **strukturierten Werkzeugkasten** autonom durch Feature-Engineering-, Trainings- und Evaluations-Schleifen iterieren und dabei Kaggle-Gold-Niveau erreichen kann. Die zentrale Erkenntnis des Reports ist **nicht** „der Agent ist schlau", sondern: **Der Wert liegt in disziplinierten, typisierten Tools, die reproduzierbare Ergebnisse erzwingen.** Der Report belegt das negativ — ihr Agent missbrauchte einen freien Python-Escape-Hatch, um Modelle unkontrolliert zu trainieren, und musste per Prompt zurückgepfiffen werden.

**Ziel dieses Projekts:** Ein **wiederverwendbares Werkzeug**, das andere auf ihre **eigenen** Tabellendaten (nicht Kaggle) loslassen können. Nicht ein einmaliger Reproduktions-Benchmark, sondern ein Tool mit Nutzwert.

**Form:** Ein **MCP-Server** (Python). Wir liefern die Tools; den Agent-Loop stellt der Nutzer mit seinem eigenen MCP-fähigen Agenten (Claude Code, Codex, Claude Desktop, …). Wir bauen **kein** Agent-Framework nach — das können bestehende Coding-Agenten bereits besser (Retries, Kontext-Management, Kostenkontrolle).

**Leitprinzip (zieht durch das ganze Design):**
> **Falsche Zustände unmöglich machen, statt sie zu verbieten.** Constraints gehören in Architektur und Typen, nicht in Bitten an das Modell. Prompting steuert *Führung*, Struktur erzwingt *Grenzen*.

---

## 2. Non-Goals (bewusst außen vor)

- **Kein eigener Agent-/LLM-Loop.** Kein Pydantic AI / LangChain als Orchestrierung. Der Nutzer-Agent ist der Loop.
- **Kein SaaS-Zwang, kein „phone home".** Local-first.
- **Keine Kaggle-Spezifika** (kein `sample_submission`-Zwang, kein verstecktes Grading).
- **Kein NN/AutoML in v1** (TabPFN, FT-Transformer, AutoML-Wrapper). `model_type` bleibt aber ein steckbarer Parameter für später.
- **Keine zentrale Telemetrie-Sammlung.** Logfire-Cloud ist ein optionaler Opt-in-Seam, nicht Teil von v1.
- **Keine microVM-/gVisor-Isolation.** Bedrohungsmodell ist „eigene Maschine, eigene Daten", nicht Multi-Tenant-Fremdcode.

---

## 3. Architektur auf einen Blick

```
Nutzer-Agent (Claude Code / Codex / Claude Desktop)   ← fährt den Loop, nicht wir
        │  MCP (stdio)
┌───────┴───────────────────────────────────────────────────────┐
│  MCP-Server "tabular-agent"  (Python)                          │
│                                                                │
│  Tools (stateless, dateibasiert — „alles ist ein Pfad"):       │
│   • inspect_data        Profil, Typen, Missing, Ziel-Analyse    │
│   • transform_features  Escape-Hatch: df→df, ZUSTANDSLOS,       │
│                         Subprozess + Import-Whitelist,          │
│                         kann strukturell nicht trainieren       │
│   • fit_encoding        strukturiert & ZUSTANDSBEHAFTET         │
│                         (target-encode / categorical-encode),   │
│                         fit-pro-Fold, Parameter                 │
│                         persistiert → leakage-sicher            │
│   • train_tabular_model LGBM / XGB / CatBoost + Baseline,       │
│                         k-fold CV, task-aware, schreibt         │
│                         metrics + fold-table + model-artifacts  │
│   • blend_models        nicht-negative Gewichte auf OOF         │
│   • predict             Pipeline nachspielen: zustandslose      │
│                         Transforms neu + gefittete Params +     │
│                         Modell                                  │
│   • finalize_run        run_report.md + Prädiktor-Bundle        │
│                                                                │
│  Querschnitt: OTel-Spans am Tool-Rand (local-first,            │
│               Logfire opt-in via LOGFIRE_TOKEN)                │
│  Persistenz:  run_dir/ (metrics.json, folds.parquet,          │
│               oof.parquet, model/, pipeline.json, report.md)  │
│  Server-Instructions: empfohlener Workflow für den Agenten     │
└────────────────────────────────────────────────────────────────┘
```

Zwei Ausgabe-Schichten, sauber getrennt:

| Schicht | Mechanismus | Zweck |
|---|---|---|
| **Reproduzierbare Artefakte** | Dateien im `run_dir/` (metrics, fold-table, OOF, Modelle, Pipeline, Report) | „Ich kann den Lauf exakt nachvollziehen und der Prädiktor ist deploybar" |
| **Observability / Traces** | OTel-Spans am Tool-Rand | „Was tat welches Tool, wie lange, warum langsam" — Debugging |

---

## 4. Zustandsmodell: stateless & dateibasiert

Jedes Tool ist **zustandslos**: es nimmt Eingabepfad(e) + Parameter, schreibt Ausgabedatei(en) ins `run_dir/`, gibt eine kompakte Zusammenfassung **+ Pfade** zurück. Der Agent orchestriert, indem er Pfade weiterreicht. Die Feature-Matrix ist eine Parquet-Datei.

Begründung: Die Artefakte *sind* der Zustand (auf Platte) → deckt sich mit dem Reproduzierbarkeits-Ziel, ist robust gegen das kurzlebige Prozessmodell von MCP-stdio-Servern, absturzsicher, keine Session-/Nebenläufigkeits-Verwaltung.

---

## 5. Der zustandslos/zustandsbehaftet-Split (Kern-Entscheidung)

Feature-Transforms zerfallen in zwei Klassen, und die Trennung verhindert **Data-Leakage** und **Train/Serve-Skew** per Konstruktion:

### 5.1 Zustandslose Transforms → Escape-Hatch (`transform_features`)
Zeilen-lokale Mathematik ohne gelernten Parameter: `dist = √(x²+y²)`, Verhältnisse, Zählungen, Aggregationen pro Zeile. (Exakt die Feature-Typen aus dem NOMAD-Beispiel des Reports.)

- Der Agent schreibt beliebigen Python-Code mit Signatur **`df → df`**.
- Ausführung in **Subprozess + Import-Whitelist** (siehe §6).
- Der Code wird als **geordneter Pipeline-Schritt** persistiert und bei `predict` **einfach neu ausgeführt** — weil zustandslos, ist er trivial reproduzierbar, kein Skew.
- Strukturell **unfähig zu trainieren** (keine Modell-Libs in der Whitelist) und **unfähig, gelernten Zustand zu erzeugen**.

### 5.2 Zustandsbehaftete Transforms → strukturierte Tools (`fit_encoding`)
Alles, was einen Parameter aus dem Trainingsset lernt **und den ein Baum nicht selbst aus den Rohdaten ableiten kann**. Für die GBDT-only-v1 ist das genau **zwei** Encoder (v1-Set A):

- **Target-Encoding** — Kategorie → ziel-informierter Mittelwert. Nicht-monoton, leakage-kritisch.
- **Categorical-/Frequency-Encoding** — Kategorie → Häufigkeit/Count (+ Ordinal-Fallback), gelernte Vokabular-Tabelle, Unknown-Kategorie-Fallback. Macht Strings auch für XGBoost nutzbar.

**Bewusst *nicht* im v1-Set:** Skalierung (StandardScaler/MinMax/Robust) und Power-Transforms (Box-Cox/Yeo-Johnson) sind für Bäume **monotone No-Ops** — ein GBDT splittet auf Reihenfolge, nicht auf Werten, also ändert eine monotone Ein-Feature-Transformation nichts am Modell. Sie kommen erst mit einem nicht-Baum-`model_type` (§16). Imputation ist marginal, weil LightGBM/XGBoost/CatBoost `NaN` **nativ** behandeln. Rein *zustandslose* Transforms (Log, Differencing, Sin/Cos, Fourier, Verhältnisse/Interaktionen) gehören ohnehin in den Escape-Hatch (§5.1), nicht hierher.

- Werden **nicht** als freier Code erlaubt, sondern über typisierte Tool-Aufrufe.
- **Fit pro CV-Fold**: Die gelernte Statistik wird nur aus den Trainingszeilen des jeweiligen Folds gebildet → keine Leakage in die Validierung.
- **Gefittete Parameter werden als Datei persistiert** (Teil von `pipeline.json` / `pipeline/`), sodass `predict` nur noch *anwendet* und nichts neu berechnet.
- Enthält **Fallbacks für unbekannte Kategorien** zur `predict`-Zeit.

### 5.3 Warum (der Footgun, den das verhindert)
Beispiel Target-Encoding als freier Code:
- **Train/Serve-Skew:** `predict` bekommt neue Rohdaten ohne Zielspalte → `groupby(...)['ziel'].mean()` crasht oder rechnet auf neuen Daten Unsinn. Modell wurde auf `Berlin=340` trainiert, bekommt jetzt `NaN`/Müll. Stille Fehlvorhersage.
- **CV-Leakage:** Fit auf dem ganzen Trainingsset → eine seltene Kategorie sieht im Validierungs-Fold ihren *eigenen* Zielwert. CV-Metrik glänzt (z. B. 0.021), echtes Holdout katastrophal (0.094). Der Agent wählt Modelle/Blend-Gewichte nach der CV-Zahl → optimiert auf eine Fantasie, liefert einen Prädiktor, der im Betrieb dreimal schlechter ist als der Report behauptet.

Der Split macht **beide** Fehler unmöglich: Der Escape-Hatch kann Leakage nicht erzeugen (nichts Zustandsbehaftetes erlaubt), und Skew nicht (nichts zum Re-Fitten).

---

## 6. Sandbox-Design

**Default-Tier: Subprozess + Import-Whitelist + Ressourcen-Limits (B1).**

Ein frischer Python-Subprozess pro `transform_features`-Aufruf:

| Mechanismus | Effekt |
|---|---|
| Eigener Prozess | Crash im Transform killt nicht den Server |
| `timeout` | Runaway-/Endlos-Code wird hart beendet |
| Ressourcen-Limits (`resource`/`ulimit`) | RAM-/CPU-Deckel |
| Eigenes Temp-`cwd` | Sieht per Default nur sein Arbeitsverzeichnis |
| **Import-Whitelist** | Kein `lightgbm`/`xgboost`/`catboost`/`sklearn.ensemble` → **strukturell kein Inline-Training**; erlaubt sind `pandas`, `numpy`, `scipy`, Standard-Lib-Mathematik |
| `df→df`-Vertrag validiert | Output muss ein DataFrame sein; sonst Fehler |

**Ehrliche Einschränkung:** Subprozess ist ein *Isolations*-, kein harter *Sicherheits*-Boundary. Der Kindprozess läuft mit denselben OS-Rechten; echtes „Netzwerk aus" braucht OS-Features. Gegen Runaway/Buggy-Code + Disziplin schützt er zuverlässig — das passt zum Bedrohungsmodell „eigene Maschine, eigene Daten".

**Seams für später (nicht v1):** Alle Sandbox-Tiers hinter *einem* Sandbox-Interface (gleicher `df→df`-Vertrag), sodass steckbar ist:
- `--sandbox=wasm` → **Pydantics `mcp-run-python`** (Deno + Pyodide; echte FS/Netz-Isolation ohne Docker). *Nicht* LangChains Klon (deprecated).
- `--sandbox=docker` → Container für stärkere Isolation, wo Docker vorhanden.

**Fehlertext lehrt:** Wirft die Whitelist einen `ImportError`, sagt die Message dem Agenten, was er stattdessen tun soll (`→ nutze train_tabular_model` bzw. `fit_encoding`), damit er nicht im Kreis probiert.

---

## 7. Task-Typ & Metrik: Auto-Detect mit Override

Das Tool unterstützt **Regression, binäre und mehrklassige Klassifikation**.

- **Task-Spec (eine Quelle der Wahrheit):** Der Nutzer gibt vorne einen optionalen Task-Spec rein — das Gegenstück zur „task description" des Reports:
  ```
  target: <spaltenname>            # erforderlich
  task_type: regression|binary|multiclass   # optional
  metric: rmse|rmsle|mae|logloss|auc|accuracy|...  # optional (Selektionsmetrik)
  positive_class: <label>          # optional (binär)
  notes: <freitext>                # optional, Domänen-Hinweise für den Agenten
  ```
- **Auto-Detect füllt Lücken:** Fehlt `task_type`/`metric`, leitet das Tool sie aus der Zielspalte (Dtype, Kardinalität) ab und **meldet die Annahme transparent zurück** („erkannt: binary, Default-Metrik logloss").
- **Zwei Metrik-Rollen:** *Selektionsmetrik* (eine; CV + Blend minimieren sie) vs. *berichtetes Panel* (mehrere zur Ansicht, z. B. binär: AUC + LogLoss + Accuracy). Override betrifft nur die Selektionsmetrik.
- **Korrektheits-Wächter:** RMSLE braucht nicht-negative Ziele — bei negativen Werten Fallback auf RMSE mit Begründung statt stillem Unsinn.

---

## 8. Modelle & Ensemble (`train_tabular_model`, `blend_models`)

- **Modelltypen:** LightGBM, XGBoost, CatBoost + **trivialer Baseline** (Mittelwert/Mehrheitsklasse) als Ehrlichkeits-Check (schlägt das „schlaue" Modell den Baseline nicht → Bug/Leakage-Verdacht).
- **CV:** k-fold (Default k=5), task-aware (stratifiziert bei Klassifikation). Schreibt **OOF-Vorhersagen** (`oof.parquet`), **fold-table** (`folds.parquet`: Fold, #Train, #Valid, Metrik → Streuung sichtbar) und **Modell-Artefakte** (pro Fold + finales Voll-Daten-Modell, `model/`).
- **Blend:** nicht-negative Gewichte pro Ziel, optimiert auf den OOF-Vorhersagen der Einzelmodelle (mehrere dekorrelierte GBDTs zu mischen ist der billigste große Score-Hebel).
- `model_type` ist ein **steckbarer Parameter** (NN/AutoML später als weiterer Typ).

---

## 9. Endzustand: Modell-Bundle + `predict` + Report

Ergebnis eines Laufs (Wahl C — beides):

- **Prädiktor-Bundle:** finales Modell(e) + **komplette serialisierte Pipeline** (`pipeline.json`/`pipeline/`): geordnete Liste der zustandslosen Transform-Codes **+** die gefitteten Parameter der strukturierten Encodings. `predict(bundle, neue_rohdaten)` spielt die Pipeline nach und liefert Vorhersagen — verarbeitet beliebige neue Rohdaten des Nutzers.
- **`run_report.md`:** bestes Modell, CV-Metriken + Fold-Streuung, Baseline-Vergleich, Feature-Importances, was der Agent probiert hat, ggf. `holdout_metric` (siehe §10).

---

## 10. Eingabe-Vertrag

- **Trainingsdatei** (Features + Zielspalte) — erforderlich.
- **Task-Spec** (§7) — optional.
- **Test-Set** — optional, in zwei Spielarten:
  - **mit Labels** → finaler ehrlicher Generalisierungs-Check *zusätzlich* zur CV; als `holdout_metric` in den Report. Fängt zu optimistische CV.
  - **ohne Labels** (Kaggle-artig) → das Tool erzeugt `test_predictions.parquet` (Batch-Scoring).
- Modellauswahl erfolgt **ausschließlich** per interner k-fold-CV. Das Test-Set ist Kontrolle/Output, **nie** Selektionskriterium (sonst leakt es).

---

## 11. Observability (local-first)

- OTel-Spans werden **am Tool-Rand automatisch** emittiert (Instrumentierung im Server-Wrapper → Agent braucht keinen Logging-Code). Das ist das Äquivalent zu „W&B Weave traces emitted automatically", nur backend-neutral.
- **Default: lokal** (Konsole/Datei/lokaler OTLP-Collector). Kein Login, kein SaaS, offline-fähig.
- **Opt-in:** Setzt der Nutzer `LOGFIRE_TOKEN`, gehen dieselben Spans *zusätzlich* an sein Logfire (ein Codepfad, kein Lock-in, weil OTel/OTLP).

---

## 12. Server-Instructions (Führung per Prompt)

Der MCP-Server liefert eine **empfohlene Workflow-Anleitung** mit (MCP-Server-`instructions`), damit ein generischer Agent „einfach funktioniert":

> inspect_data → transform_features (zustandslose Domänen-Features) → fit_encoding (zustandsbehaftetes) → train_tabular_model (mehrere Typen) → blend_models → iterieren → finalize_run.

Enthält den ausdrücklichen Hinweis, dass Modelltraining **nur** über `train_tabular_model` läuft (die *Führung*; die *Grenze* erzwingt ohnehin die Sandbox).

---

## 13. Verteilung / Packaging

- Python-Package, installierbar via `pip`/`uv`.
- Konsolen-Entry-Point startet den MCP-Server (stdio).
- README mit Registrierungs-Snippet für gängige MCP-Clients (Claude Code / Codex / Claude Desktop).
- Harte Deps minimal: `mcp`-SDK, `pandas`, `numpy`, `scipy`, `lightgbm`, `xgboost`, `catboost`, `scikit-learn`, `pyarrow`, OTel/`logfire`. Deno/Docker **nicht** hart erforderlich (nur für optionale Sandbox-Tiers).

---

## 14. Teststrategie (Leitplanken für den Implementierungsplan)

- **Leakage-Regressionstest:** Konstruierter Datensatz mit einer seltenen Kategorie; Ziel-Encoding über `fit_encoding` muss ehrliche CV liefern, während ein absichtlich „naiver" Fit deutlich zu gut aussieht → beweist, dass der pro-Fold-Fit greift.
- **Train/Serve-Konsistenz:** `predict` auf denselben Zeilen wie im Training muss (bis auf numerische Toleranz) dieselben transformierten Features erzeugen wie zur Trainingszeit.
- **Sandbox-Disziplin:** `transform_features` mit `import lightgbm` muss mit lehrreichem Fehler abbrechen; ein Endlos-Loop muss durch `timeout` beendet werden.
- **Task-Auto-Detect:** Regression/binary/multiclass-Fixtures → korrekte Erkennung + korrekter RMSLE-Guard bei negativen Zielen.
- **Determinismus:** fixe Seeds → reproduzierbare OOF/Metriken.

---

## 15. Entscheidungs-Log (Ergebnisse des Grillings)

| # | Entscheidung | Wahl | Kern-Begründung |
|---|---|---|---|
| 1 | Ziel | Wiederverwendbares Tool für andere, nicht Kaggle-spezifisch | Nutzwert statt Benchmark-Reproduktion |
| 2 | Form | MCP-Server (Tools, kein eigener Loop) | Wert liegt in den Tools; Loop können Coding-Agenten schon |
| 3 | Traces | Local-first, Logfire opt-in | Kein SaaS-Zwang; OTel = kein Lock-in |
| 4 | Zustandsmodell | Stateless / dateibasiert | Artefakte *sind* der Zustand; robust gegen MCP-Prozessmodell |
| 5 | Escape-Hatch | Ja, aber `df→df`, gesandboxt, kann nicht trainieren | Flexibilität behalten, Disziplin strukturell erzwingen |
| 6 | Sandbox-Default | Subprozess + Import-Whitelist | `pip install`-tauglich, volle pandas-Speed; WASM/Docker als Seams |
| 7 | Endzustand | Modell-Bundle + `predict` + Report | Nutzer will *sein* Problem lösen, nicht nur einen Score |
| 8 | Zustands-Besitz | Split: stateless Escape-Hatch + strukturierte Encoding-Tools | Leakage/Skew per Konstruktion unmöglich |
| 9 | Modelle | GBDT-Trio + Blend + Baseline | Arbeitspferd für Tabular; Blend = billiger Score-Hebel |
| 10 | Task/Metrik | Auto-Detect mit Override (Task-Spec) | Reibungsarm, aber Nutzer behält Kontrolle |

---

## 16. Offene Punkte / Zukunft (nach v1)

- WASM- (Pydantic `mcp-run-python`) und Docker-Sandbox-Tiers als Opt-in.
- Weitere `model_type`s (Tabular-NN, TabPFN, AutoML).
- **Scaler / Power-Transforms / Imputation als strukturierte Encoder** — erst zusammen mit einem nicht-Baum-`model_type` (linear/NN/kNN/SVM), für den sie überhaupt wirken. Bei GBDTs sind monotone Ein-Feature-Transforms No-Ops, deshalb nicht in v1.
- Logfire-Cloud-Auswertung (nur mit explizitem Consent).
- Optionaler MLE-bench-Adapter, falls doch Benchmark-Reproduktion gewünscht.
- Multi-File-Eingabe / Joins.
- **Autonomer Loop-Runner (autoresearch-Stil):** Eine dünne CLI über denselben Tools (Frage 2, Variante B), die den Loop selbst fährt statt ihn dem Nutzer-Agenten zu überlassen — nach Karpathys `autoresearch`-Muster: *Modify → Verify → Keep/Discard → Repeat* über viele Zyklen, mit Verlaufs-Ledger und git-artigem „commit bei Verbesserung, sonst rollback". Der **Task-Spec fungiert dann als GOAL.md / Fitness-Funktion** (die Selektionsmetrik ist die eine Zahl, auf die der Loop optimiert). Voraussetzung, die unser v1-Kern bereits liefert: eine *ehrliche* Fitness-Funktion (leakage-sichere OOF-Metrik) — je autonomer der Loop, desto kritischer, dass die optimierte Zahl vertrauenswürdig ist.

---

## 17. Nachtrag (2026-07-22): Anforderungen aus dem Phase-1-Review — für Phase 2 (v1)

Beim Durcharbeiten der Core-Lib (`notebooks/phase1_core_walkthrough.ipynb`) identifiziert; beide gehören in den Phase-2-Plan, nicht in „nach v1":

1. **`inspect_data`: Profil um Verteilungsinformation erweitern.** Das aktuelle `profile_dataframe` liefert nur Struktur (dtype, Missing, Unique) — für gute Modellierungsentscheidungen des Agenten fehlt Verteilung:
   - Numerische Spalten: `mean`, `std`, `min`, `max`, Quantile, **Skewness**. Direkt entscheidungsrelevant: stark rechtsschiefes nicht-negatives Target → `rmsle` (die `default_metric`-Heuristik prüft bisher nur Nicht-Negativität); schiefe Features → `log1p`-Kandidat für `transform_features`.
   - Kategoriale Spalten: Top-k-Kategorien mit Häufigkeiten → informiert die Smoothing-Wahl beim Target-Encoding (viele seltene Kategorien → mehr Smoothing).
   - Umsetzung: Erweiterung von `profile_dataframe` in der Core-Lib (`inspect.py`); bleibt billig und JSON-serialisierbar.

2. **Neues Tool `group_stats` (Exploration, getrennt vom Profil):** Group-by-Aggregationen (`df, by, agg`) als eigenes MCP-Tool, damit der Agent Hypothesen prüfen kann („unterscheidet sich die Churn-Rate nach City?"). **Wichtig:** In der Tool-Description explizit als *Explorations*-Tool markieren, nicht als Feature-Quelle — wer Target-Statistiken pro Kategorie auf den vollen Daten ansieht und daraus manuell Features baut, holt sich das Leakage-Problem durch die Hintertür zurück, das `fit_encoding` strukturell verhindert.

### 17.1 Revision (2026-07-24): beide Punkte → *ein* read-only-Python-Tool

Statt (1) das Profil um Verteilungs-Statistiken (Skewness, Quantile) zu erweitern und (2) ein eigenes, eng typisiertes `group_stats`-Tool zu bauen, gibt es **ein allgemeines read-only-Python-Tool** neben `transform_features`. Begründung:

- **`profile_dataframe` bleibt ein billiger struktureller Fingerabdruck** (dtype, Missing, Unique) — die always-on-erste-Ansicht. Verteilungsformen (Skewness, Quantile) und Group-by-Statistiken sind kein fixes Set im Profil, sondern rechnet der Agent *bei Bedarf* selbst — genau die, die eine Entscheidung braucht. Kein „welche Stats backen wir ein?"-Kompromiss.
- **Ein Tool statt vieler enger Tools.** `group_stats` wäre nur ein Spezialfall von „lass den Agenten read-only-pandas laufen". Das allgemeine Tool deckt Skewness, Quantile, Group-bys und beliebige Hypothesen-Checks ab.
- **Symmetrie zu `transform_features` (§5.1/§6):** gleiche Sandbox (Subprozess + Import-Whitelist), gleicher „Agent schreibt Python"-Vertrag. Unterschied ist der *Verbleib des Outputs*:
  - `transform_features`: `df → df`, wird als Pipeline-Schritt **persistiert** und speist das Modell.
  - read-only (`explore(df) → summary`): Ergebnis wird dem Agenten **gezeigt und verworfen**. Kein Persistenz-Pfad → kann strukturell nicht zum Feature werden.
- **Der Leakage-Backdoor aus Punkt 2 bleibt geschlossen** — nicht mehr per Tool-Description-Konvention, sondern *strukturell*: Da das read-only-Ergebnis nirgends persistiert wird und zusätzlich größenbegrenzt ist, lässt sich eine per-Kategorie-Target-Statistik nicht als handgebaute (leakage-behaftete) Spalte zurückschmuggeln. `fit_encoding` (OOF, per Fold) bleibt der einzige sanktionierte Weg von Kategorie→Target-Signal zum Feature.

Umsetzung Phase 2 (Stand 2026-07-24): `sandbox.py` (Default-Tier aus §6) + `explore.py` (`explore_dataframe`) implementiert und getestet; Lehr-Notebook `notebooks/phase2_sandbox_and_exploration.ipynb`. `profile_dataframe` bleibt struktur-only. `transform_features` (gleiche Sandbox, `df→df`, persistiert) ist der nächste Chunk.

---

## 18. Option (2026-07-23): Per-Vorhersage-Erklärbarkeit via Post-hoc-SHAP — „influencing factors"

**Status: Option, keine harte v1-Anforderung.** Motivation aus der Praxis (Demand-Forecasting): Kunden reklamieren einzelne „komische" Vorhersagen, und man braucht eine Root-Cause-Antwort — *welches Feature hat diese Vorhersage getrieben?* Aggregierte Modellgüte hilft dabei nicht; das ist per-Zeile-Attribution zur Serving-Zeit.

**Werkzeug: gruppiertes TreeSHAP.** Passt genau, weil alle `model_type`s Baum-Ensembles sind:
- TreeSHAP ist exakt und billig für Bäume; jeder Booster liefert es nativ (LightGBM `pred_contrib=True`, XGBoost `pred_contribs=True`, CatBoost `type="ShapValues"`) → **keine `shap`-Dependency** nötig.
- Additiv in Target-Einheiten: `base + Σ Beiträge = ŷ`. Bei Regression (Forecasting) direkt kommunizierbar.
- **Komponiert linear über den Blend:** Da `ŷ = Σ wᵢ·ŷᵢ` und alle Modelle denselben Feature-Raum teilen, gilt `SHAP(blend) = Σ wᵢ·SHAP(modelᵢ)`. Die emittierten Beiträge **rekonziliieren exakt** zur ausgelieferten Vorhersage (Unit-testbar). `_Baseline` trägt null bei Features + Mittelwert als Base bei — komponiert sauber.

**Spezifikation (falls gebaut):**
- **Attributions-Einheit = rohes Eingabe-Feature**, nicht die daraus generierte Modellspalte. Aus Encoder-Metadaten werden abgeleitete Spalten aufs Roh-Feature zurückgefaltet (one-hot `product_is_X…` oder einzelnes `product_te` → ein Beitrag „product"). Das ist die *einzige* domänen-agnostische Gruppierung; Multi-Feature-„Faktoren" (promo = mehrere Roh-Features) brauchen Domänenwissen und sind **nicht** Teil davon.
- Ausgabe: **signierter Beitrags-Vektor, nach Betrag sortiert** — nicht auf „das eine Feature" reduzieren (bei „Vorhersage zu hoch" will man die Top-*positiven* Treiber).
- Präsentation im **Target-Raum**: additive Einheiten bei rohem Target, **multiplikativ** (`exp`) bei log-transformiertem Target. Erfordert, dass Link/Transform deklariert und in den Explainer getragen wird.
- **Opt-in: paarweise SHAP-Interaktionswerte** für den Drill-down. Statt einen `promo × product`-Effekt still aufzuteilen (unehrlich), erscheint er als eigene benannte Interaktions-Zelle — rekonstruiert die „ehrliche gemeinsame Faktor"-Eigenschaft blockadditiver Modelle. Kostet O(features²)/Zeile → nur im „diese Vorhersage ist komisch"-Pfad, nicht bei jedem Serve.
- **Label an der Ausgabe: „Modell-Begründung, nicht Kausalität".** SHAP erklärt treu, *was das Modell gerechnet hat* — bei korrelierten Features (promo/holiday) kann das vom wahren Treiber abweichen. In genau den High-Stakes-Kundengesprächen, für die man es baut, sonst ein Footgun.
- **Optional: monotone Constraints** (alle drei Booster unterstützen sie), pro Feature im Task-Spec deklarierbar, per Default aus. Greift die Reklamationsursache an der *Quelle* an: ein aus Rauschen gelernter nicht-monotoner „Wackler" erzeugt die komische Vorhersage *und* eine SHAP-Story, die der Domänen-Intuition widerspricht. Braucht Richtung-pro-Feature = Domänenwissen → deklarierbar, nicht automatisch.

**Abhängigkeit vom Kern:** Nur vertrauenswürdig auf leakage-sauberem Encoding (siehe `docs/target_encoding_leakage_insight.md`). SHAP ist *treu* gegenüber der Leakage — ein durch Second-Order-Optimismus über-vertrautes `_te`-Feature bekommt entsprechend zu viel Attribution, die autoritativ aussieht. Encoding-Disziplin ist Vorbedingung für vertrauenswürdige Erklärungen.

**Warum Post-hoc-SHAP statt Glass-Box-Modell (EBM/GAM):** Ziel ist „nützlich für Unternehmen", nicht Kaggle — aber Post-hoc-SHAP entkoppelt „stärkstes Modell wählen" von „erklären". Auf der *Genauigkeits*-Achse wird nichts geopfert (volles GBM bleibt); der einzige Preis ist Erklärungs-*Exaktheit unter Interaktionen*, den die opt-in-Interaktionswerte teilweise zurückholen. Ein Glass-Box-Additivmodell (EBM/GA2M mit automatischer Interaktionserkennung) bliebe die Alternative, falls exakte native Attribution später wichtiger wird als GBM-Power.

---

## 19. Harness & resumierbares Gedächtnis (2026-07-25): Langläufe über Kontext-Grenzen hinweg

**Status: Design-Note für die *Harness*, nicht für den Server.** Der MCP-Server bleibt zustandslos und zielagnostisch (§4, §12) — er *fährt den Loop nicht*. Diese Note fixiert, wie eine dünne Harness den Loop über viele Zyklen und **über Kontext-Kompaktierungen hinweg** laufen lässt („über Nacht"-Betrieb), damit die dafür nötigen Server-Artefakte **jetzt** richtig geschnitten werden. Motivation: der NOMAD-Report (`wandb.ai/wandb_fc/tabular-nomad`) zeigt Nacht-Läufe, verschweigt aber genau diese Mechanik; Kontext-Management liegt strukturell auf der Client-/Harness-Seite.

**Was Langläufe überhaupt ermöglicht (zwei Eigenschaften, nur die erste ist Server-Sache):**
1. **Langsames Kontext-Wachstum** — jedes Tool gibt *kompakte Zusammenfassung + Pfade* zurück, nie den DataFrame (§4). Das ist der eigentliche Enabler: der Agent kann Dutzende Zyklen fahren, bevor das Fenster voll ist. Schon vorhanden.
2. **Kontext-Kompaktierung über Stunden** — wenn das Fenster doch voll wird, muss *etwas* den Transcript zusammenfassen/verwerfen und frisch starten. Das kann der zustandslose Server nicht; es ist Harness-Aufgabe. Der Server *ermöglicht* es nur, indem der Zustand auf Platte liegt.

### 19.1 Zwei Artefakte, bewusst getrennt — nicht `finalize_run` überladen

| | **`run_journal`** (resumierbares Gedächtnis) | **`finalize_run`** (terminales Deliverable) |
|---|---|---|
| Wann | *jeden* Zyklus aktualisiert, während des Laufs | **einmal**, am Ende |
| Für wen | den *nächsten* Agenten (maschinenlesbar) | den Menschen (`report.md` + `predict`-Bundle) |
| Form | kompaktes, strukturiertes Leaderboard | bestes Modell, voller Report, deploybares Bundle |
| Lebenszyklus | append-lastig, wächst | write-once, final |

**Anti-Pattern:** `finalize_run` pro Zyklus neu schreiben. Das koppelt „mein Scratch-Gedächtnis" an „das Ding, das ich ausliefere" und führt eine Bundle-erzeugende Operation 10× aus. Stattdessen: **`run_journal.json` während, `finalize_run` genau einmal am Schluss** (liest den `best`-Pointer des Journals und packt daraus das Bundle).

### 19.2 Wer schreibt was — Fakten vom Werkzeug, Deutung vom Agenten

Nicht *der Agent* pflegt das Log (fragil — Disziplin, die er vergisst; genau der Footgun aus dem NOMAD-Report). Aufteilung nach *wer die Information hat*:

- **Fakten → das Tool appendet deterministisch, als Seiteneffekt.** `train_tabular_model` schreibt eine Leaderboard-Zeile beim tatsächlichen Trainieren: `{features, encoder_config, model_type, cv_score, fold_spread, vs_baseline, artifact_path}`. Der Experiment-Record ist ein *Nebenprodukt des Experiments*, keine Agenten-Pflicht → kann nicht übersprungen werden, kostet null Agenten-Disziplin.
- **Deutung → der Agent appendet eine dünne Notiz.** Nur der Agent weiß *warum* er etwas probiert und was als Nächstes: pro Zyklus `{hypothesis, why, outcome_interpretation, next_ideas, dead_ends}`. Klein, Prosa, billig.

→ Journal = auto-gewachsenes Leaderboard (vertrauenswürdig, vollständig) + schlanke agenten-geschriebene Plan-/Hypothesen-Schicht. Keine der beiden hängt von der Perfektion der anderen ab.

### 19.3 Resume-Handoff: was der frische Agent liest

Nicht „die finalize-Logs", nicht alle Artefakte — nur:

```
task_spec  +  run_journal.json  +  server-instructions
```

Das Leaderboard sagt, **was schon probiert und gescored wurde** (keine Wiederholung), `best` die zu schlagende Latte, `next_ideas`/`dead_ends` die Richtung. Artefakte bleiben auf Platte als Detail, in das er *bei Bedarf* drilldownt — der Resume-Prompt bleibt klein genug, um ihn beliebig oft zu wiederholen.

### 19.4 Schema-Skizze `run_journal.json`

```jsonc
{
  "task": { "target": "churn", "task_type": "binary", "metric": "auc" },
  "baseline": { "cv_score": 0.50 },
  "best": { "experiment_id": "e7", "cv_score": 0.83, "bundle_hint": "runs/e7/" },
  "leaderboard": [            // 1 Zeile/Training, vom Tool deterministisch appended
    { "id": "e7", "cycle": 4, "features": ["log_income","city_te"],
      "encoder_config": {"city":"target"}, "model_type": "lgbm",
      "cv_score": 0.83, "fold_spread": 0.011, "vs_baseline": "+0.33",
      "artifact_path": "runs/e7/" }
  ],
  "log": [                    // 1 Eintrag/Zyklus, vom Agenten geschrieben
    { "cycle": 4, "hypothesis": "city treibt churn -> target-encode",
      "why": "explore zeigte churn-rate-Spread über Städte",
      "outcome_interpretation": "bestes Modell bisher; fold-spread niedrig",
      "next_ideas": ["income/age-ratio", "xgb zum Blend"],
      "dead_ends": ["age allein: kein Lift"] }
  ]
}
```

### 19.5 Trigger & Stop — Harness-Sache, nicht Agenten-Sache

- **Kompaktieren nach *Budget*, nicht nach fixem N.** Zyklen variieren stark im Kontext-Kosten (ein Feature vs. ein CatBoost-Training). Auslöser: Fenster überschreitet ~70%. Ein Zyklus-Cap ist ein akzeptabler *v1*-Ersatz, aber das echte Signal ist Token-Budget.
- **Die Harness besitzt die Stop-Entscheidung und liest dafür das Leaderboard.** Regel über den auto-geloggten Scores: „kein Fortschritt über die letzten K Zyklen ODER Wall-Clock-Budget erschöpft → stop + `finalize_run`". Der Agent muss das nicht besitzen; das verhindert auch das ewige Weiterlaufen.

### 19.6 Loop-Skizze

```
loop (harness):
  spawn agent  mit  [ task_spec + run_journal + instructions ]
  agent fährt Zyklen: inspect → hypothesize → transform_features/fit_encoding
                      → train_tabular_model (appendet Leaderboard-Zeile)
                      → agent appendet Hypothesen-/next-ideas-Notiz
  bis  Kontext ~70%  ODER  Zyklus-Cap
  kompaktiere: Transcript verwerfen, run_journal behalten (es IST der Zustand)
  wiederhole mit frischem Agenten
stop wenn: kein Fortschritt über K Zyklen  ODER  Zeitbudget
  → finalize_run genau einmal (bestes Modell → report + bundle)
```

**Kern-Inversion ggü. dem naiven Entwurf:** der dauerhafte Zustand ist **kein Report, den der Agent pflegt**, sondern ein **Journal, das die Tools pflegen**, mit einer dünnen Deutungs-Schicht darüber. Das macht die „Kontext leeren und resumen"-Grenze beliebig oft sicher überquerbar. `run_journal` als Tool-geschriebenes Artefakt gehört damit in den Server-Scope (Seiteneffekt von `train_tabular_model` + ein schlankes `journal_append` für die Agenten-Notiz); die Loop-/Kompaktier-/Stop-Logik bleibt außerhalb, in der Harness.
