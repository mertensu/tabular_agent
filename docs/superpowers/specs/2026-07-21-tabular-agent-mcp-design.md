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
