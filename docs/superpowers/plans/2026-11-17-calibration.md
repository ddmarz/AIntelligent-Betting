# Calibrazione delle probabilità — Piano di implementazione

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Misurare quanto sono calibrate le probabilità over/under (corner, tiri in porta, gol) e 1X2 del modello, a partire dai CSV per partita di un run walk-forward.

**Architecture:** (1) I CSV per partita del run aggiungono le colonne che servono: dispersioni NB, quote grezze e gol reali. (2) Il nuovo modulo `dl_model/calibration.py`, senza torch, trasforma le distribuzioni salvate in probabilità di over per linea, calcola tabelle di affidabilità, ECE e analisi dei casi convinti, disegna i grafici e offre il comando `python -m dl_model.calibration`.

**Tech Stack:** Python, pandas, numpy, scipy (`nbinom`, `poisson`), matplotlib (backend `Agg`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-deep-learning-multitask-design.md`, sezione "Calibrazione".

## Global Constraints

- **Nessun commit** (richiesta dell'utente): al posto dei commit c'è un controllo `git status`.
- Interprete: `~/anaconda3/envs/Python3144/python.exe`. Test: `~/anaconda3/envs/Python3144/python.exe -m pytest -q`, dalla root del repo.
- `dl_model/calibration.py` **non importa torch**, né direttamente né tramite altri moduli di `dl_model`. In questo ambiente numpy/MKL e torch caricano due copie di OpenMP e il processo può terminare.
- Distribuzione NB: `scipy.stats.nbinom(n=disp, p=disp/(disp+mean))`, la stessa parametrizzazione di `dl_model/losses.negative_binomial_nll` (verificato).
- Linee: corner totali `[6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]`; corner per squadra `[2.5, 3.5, 4.5, 5.5, 6.5]`; tiri in porta totali `[6.5, 7.5, 8.5, 9.5, 10.5, 11.5]`; tiri in porta per squadra `[2.5, 3.5, 4.5, 5.5]`; gol totali `[1.5, 2.5, 3.5]`.
- 10 fasce di uguale ampiezza su [0, 1]; intervallo di Wilson al 95% (z = 1.96); soglia di "convinto" `0.7` (parametro).
- Riferimento di mercato solo per 1X2 e gol over/under 2,5, con il margine tolto in proporzione.
- Output in `checkpoints/<nome run>_calibration/`: `summary.csv`, `bins.csv`, un PNG per famiglia.

## Review Focus

1. **Quote O/U mancanti codificate come 0.** In `matches_clean.csv`, `Over25`/`Under25` valgono `0.0` quando `has_ou=False`. Il riferimento di mercato deve escludere le righe con quota ≤ 1, invece di produrre probabilità infinite o NaN. Il test è nel Task 3.
2. **Probabilità esattamente 1.0 o 0.0.** Un over quasi certo (es. corner totali over 6,5 con medie alte) deve finire nell'ultima fascia, 90–100%, e non fuori dalla tabella. Il test è nel Task 2.
3. **Fasce vuote o con pochissime partite.** Una fascia senza partite deve comparire con `n=0` e valori NaN, senza far crashare né ECE né grafici. Una fascia con 3 partite deve avere un intervallo di Wilson largo. Il test è nel Task 2.
4. **Troncamento della convoluzione.** Con medie alte, il supporto 0..`max_count` deve contenere praticamente tutta la massa: con media 15 e dispersione 5, la somma della pmf deve essere ≥ 0.9999. Il test è nel Task 2.
5. **Run vecchio senza le nuove colonne.** Lanciare la calibrazione su un run fatto prima del Task 1 deve fallire con un messaggio che dice di rilanciare il training, non con un `KeyError` generico. Il test è nel Task 3.

---

### Task 1: Colonne aggiuntive nei CSV per partita

**Files:**
- Modify: `dl_model/evaluation.py` (`GOAL_RESULT_COLUMNS`, `COUNT_RESULT_COLUMNS`, record di `evaluate_goals` ed `evaluate_counts`)
- Test: `tests/test_dl_evaluation.py`

**Interfaces:**
- Consumes: nulla.
- Produces: `GOAL_RESULT_COLUMNS` aggiunge, dopo `FTResult`, le colonne `FTHome`, `FTAway`, `OddHome`, `OddDraw`, `OddAway`, `Over25`, `Under25` (valori copiati dalla riga della partita). `COUNT_RESULT_COLUMNS` aggiunge, dopo le `pred_*`, `disp_home_shots_on_target`, `disp_away_shots_on_target`, `disp_home_corners`, `disp_away_corners` (dagli output `disp_*` del modello).

- [ ] **Step 1: Aggiornare i test.** Nel fixture di `tests/test_dl_evaluation.py` aggiungere `"Over25": [1.9, 2.1]` e `"Under25": [1.95, 1.75]`. Aggiungere le nuove colonne ai set attesi di `test_evaluate_goals_returns_expected_columns` e `test_evaluate_counts_returns_expected_columns`, più:
  ```python
  # in test_evaluate_goals_keeps_match_identity_and_consistent_probabilities
  assert list(result["FTHome"]) == [2, 0]
  assert list(result["OddHome"]) == [1.8, 2.2]
  assert list(result["Over25"]) == [1.9, 2.1]
  # in test_evaluate_counts_returns_expected_columns
  assert (result[["disp_home_shots_on_target", "disp_away_shots_on_target",
                  "disp_home_corners", "disp_away_corners"]] > 0).all().all()
  ```
- [ ] **Step 2: Eseguire i test e verificare che falliscano.** Run: `~/anaconda3/envs/Python3144/python.exe -m pytest tests/test_dl_evaluation.py -q`. Expected: 3 FAIL sui set di colonne / `KeyError`.
- [ ] **Step 3: Implementare** le colonne del Produces in `evaluate_goals`/`evaluate_counts` e nelle due costanti.
- [ ] **Step 4: Eseguire tutta la suite.** Run: `~/anaconda3/envs/Python3144/python.exe -m pytest -q`. Expected: tutti passati.
- [ ] **Step 5: Nessun commit.** `git status` mostra solo `dl_model/evaluation.py` e `tests/test_dl_evaluation.py` fra i file toccati in questo task.

---

### Task 2: Probabilità e metriche di calibrazione (funzioni pure)

**Files:**
- Create: `dl_model/calibration.py`
- Create: `tests/test_calibration.py`

**Interfaces:**
- Consumes: nulla (solo numpy, pandas, scipy).
- Produces, tutto in `dl_model/calibration.py`:
  - `MAX_COUNT = 60`
  - `nb_pmf(mean: np.ndarray, disp: np.ndarray, max_count: int = MAX_COUNT) -> np.ndarray`, forma `(n, max_count + 1)`
  - `prob_over_single(mean, disp, line: float) -> np.ndarray`, cioè P(X > line)
  - `prob_over_total(mean_home, disp_home, mean_away, disp_away, line: float, max_count: int = MAX_COUNT) -> np.ndarray`, cioè P(X_casa + X_trasf > line) per convoluzione delle due pmf
  - `prob_over_goals(lambda_home, lambda_away, line: float) -> np.ndarray`, Poisson con λ = λ_casa + λ_trasf
  - `market_prob_over(over_odds, under_odds) -> np.ndarray`, cioè `(1/over) / (1/over + 1/under)`
  - `wilson_interval(successes, n, z: float = 1.96) -> tuple[float, float]`, che restituisce `(nan, nan)` se `n == 0`
  - `reliability_table(pred: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> pd.DataFrame`, con colonne `bin_low, bin_high, n, mean_pred, obs_freq, ci_low, ci_high`. Sempre `n_bins` righe, anche se vuote. La fascia è `min(floor(pred * n_bins), n_bins - 1)`, quindi `pred = 1.0` cade nell'ultima.
  - `expected_calibration_error(table: pd.DataFrame) -> float`, cioè Σ (n_b/N)·|mean_pred_b − obs_freq_b| sulle fasce con n > 0
  - `confidence_summary(pred, outcome, threshold: float = 0.7) -> dict`, con chiavi `share_confident`, `over_n`, `over_mean_pred`, `over_obs`, `over_ci_low`, `over_ci_high`, `under_n`, `under_mean_pred`, `under_obs`, `under_ci_low`, `under_ci_high`. "Over convinto" = `pred >= threshold`; "under convinto" = `pred <= 1 - threshold`, riportato dal lato dell'under: `under_mean_pred` = media di `1 - pred`, `under_obs` = frequenza di `outcome == 0`.

- [ ] **Step 1: Scrivere i test in `tests/test_calibration.py`**
  ```python
  def test_prob_over_single_matches_scipy_nbinom():
      p = prob_over_single(np.array([5.0]), np.array([3.0]), 4.5)
      assert p[0] == pytest.approx(nbinom.sf(4, 3.0, 3.0 / 8.0))

  def test_prob_over_total_tends_to_poisson_sum_with_huge_dispersion():
      p = prob_over_total(np.array([5.0]), np.array([1e6]), np.array([4.0]), np.array([1e6]), 9.5)
      assert p[0] == pytest.approx(poisson.sf(9, 9.0), abs=1e-3)

  def test_prob_over_total_decreases_with_line():
      args = (np.array([5.5]), np.array([6.0]), np.array([4.5]), np.array([6.0]))
      probs = [prob_over_total(*args, line)[0] for line in (6.5, 8.5, 10.5, 12.5)]
      assert probs == sorted(probs, reverse=True)

  def test_nb_pmf_support_holds_almost_all_mass():
      assert nb_pmf(np.array([15.0]), np.array([5.0])).sum() >= 0.9999

  def test_prob_over_goals_is_poisson_of_summed_lambdas():
      assert prob_over_goals(np.array([1.5]), np.array([1.0]), 2.5)[0] == pytest.approx(poisson.sf(2, 2.5))

  def test_market_prob_over_removes_margin():
      assert market_prob_over(np.array([1.9]), np.array([1.9]))[0] == pytest.approx(0.5)

  def test_wilson_interval_known_values_and_empty():
      assert wilson_interval(3, 4) == pytest.approx((0.3006, 0.9544), abs=1e-4)
      assert all(np.isnan(v) for v in wilson_interval(0, 0))

  def test_reliability_table_perfect_calibration_and_edges():
      pred = np.array([0.75, 0.75, 0.75, 0.75, 1.0])
      outcome = np.array([1, 1, 1, 0, 1])
      table = reliability_table(pred, outcome)
      assert len(table) == 10
      row = table[table["bin_low"] == pytest.approx(0.7)].iloc[0]
      assert (row["n"], row["mean_pred"], row["obs_freq"]) == (4, pytest.approx(0.75), pytest.approx(0.75))
      assert table.iloc[-1]["n"] == 1                                 # pred = 1.0 nell'ultima fascia
      assert table.iloc[0]["n"] == 0 and np.isnan(table.iloc[0]["obs_freq"])
      assert expected_calibration_error(table) == pytest.approx(0.0)

  def test_reliability_table_small_bin_has_wide_interval():
      table = reliability_table(np.array([0.55, 0.55, 0.55]), np.array([1, 0, 1]))
      row = table[table["n"] == 3].iloc[0]
      assert row["ci_high"] - row["ci_low"] > 0.5

  def test_confidence_summary_reports_both_sides():
      s = confidence_summary(np.array([0.8, 0.8, 0.2, 0.5]), np.array([1, 0, 0, 1]))
      assert s["share_confident"] == pytest.approx(0.75)
      assert (s["over_n"], s["over_mean_pred"], s["over_obs"]) == (2, pytest.approx(0.8), pytest.approx(0.5))
      assert (s["under_n"], s["under_mean_pred"], s["under_obs"]) == (1, pytest.approx(0.8), pytest.approx(1.0))

  def test_calibration_module_does_not_import_torch():
      code = "import sys, dl_model.calibration; assert 'torch' not in sys.modules"
      subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)
  ```
  (`REPO_ROOT = Path(__file__).resolve().parent.parent`.)
- [ ] **Step 2: Eseguire i test e verificare che falliscano.** Run: `~/anaconda3/envs/Python3144/python.exe -m pytest tests/test_calibration.py -q`. Expected: errore di import `dl_model.calibration`.
- [ ] **Step 3: Implementare le funzioni del Produces.** `prob_over_*` usa `sf(floor(line))`. `prob_over_total` convolve riga per riga le due pmf di `nb_pmf` e somma la massa sopra `floor(line)`. Si può fare in forma vettoriale, per esempio con `np.einsum` sul prodotto esterno e una maschera `i + j > line`.
- [ ] **Step 4: Eseguire i test e verificare che passino.** Run: come allo Step 2. Expected: 11 passed.
- [ ] **Step 5: Nessun commit.** `git status` mostra i due file nuovi.

---

### Task 3: Report dai CSV di un run, grafici e comando

**Files:**
- Modify: `dl_model/calibration.py`
- Modify: `tests/test_calibration.py`

**Interfaces:**
- Consumes: le funzioni del Task 2; le colonne dei CSV del Task 1.
- Produces, in `dl_model/calibration.py`:
  - costanti `CORNERS_TOTAL_LINES`, `CORNERS_TEAM_LINES`, `SOT_TOTAL_LINES`, `SOT_TEAM_LINES`, `GOALS_LINES` (valori nei Global Constraints), `CONFIDENCE_THRESHOLD = 0.7`, `CHECKPOINT_DIR` = `<root>/checkpoints`;
  - `build_events(goals: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame`: formato lungo con colonne `family, market, line, source, pred, outcome`. Le famiglie sono `corners_total`, `corners_team`, `sot_total`, `sot_team`, `goals_ou`, `1x2`. `market` vale `"over <linea>"` per gli over/under e `"home"`/`"draw"`/`"away"` per l'1X2. `source` vale `"model"` o `"market"`. Le righe per squadra impilano casa e trasferta. Il mercato compare solo per l'1X2 (`market_p_*`) e per `goals_ou` 2,5 (`market_prob_over(Over25, Under25)`, solo dove entrambe le quote sono > 1). Per l'1X2 `line` è NaN;
  - `calibration_report(events: pd.DataFrame, n_bins: int = 10, threshold: float = CONFIDENCE_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]`, che restituisce `(summary, bins)`. Il raggruppamento è per `family, market, line, source`. `summary` ha una riga per gruppo, con le colonne di gruppo, `n`, `ece` e le chiavi di `confidence_summary`. `bins` ha le colonne di gruppo più quelle di `reliability_table`;
  - `plot_reliability(bins: pd.DataFrame, out_dir: Path) -> list[Path]`: un PNG `reliability_<family>.png` per famiglia. Ogni grafico ha la diagonale, una curva per `market`, il mercato tratteggiato e la dimensione dei punti ∝ `n`. Le fasce con `n == 0` vengono saltate. Si usa `matplotlib.use("Agg")`;
  - `find_run(checkpoint_dir: Path, run_name: str | None = None) -> str`: restituisce il nome del run, ossia il prefisso `multitask_<data-ora>`. Senza nome sceglie il più recente fra i `*_goals.csv` in ordine alfabetico, che coincide con quello cronologico. Se non ne trova, solleva `FileNotFoundError` con un messaggio chiaro;
  - `load_run(checkpoint_dir: Path, run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]`: legge i due CSV. Se mancano colonne del Task 1, solleva `ValueError` con un messaggio che contiene `"rilancia il training"`;
  - `main(argv: list[str] | None = None) -> None`, con argparse: `run_name` opzionale, `--checkpoint-dir` e `--threshold`. Scrive `summary.csv`, `bins.csv` e i PNG in `<checkpoint-dir>/<run>_calibration/`, e stampa `summary` (`market, line, source, n, ece, share_confident, over_obs, under_obs`). Sotto `if __name__ == "__main__": main()`.

- [ ] **Step 1: Scrivere i test**, con un helper `_run_frames()` che costruisce i due DataFrame di 3 partite con tutte le colonne di `GOAL_RESULT_COLUMNS`/`COUNT_RESULT_COLUMNS`. Le colonne si possono scrivere in chiaro nel test, perché importare `dl_model.evaluation` porterebbe dentro torch. La terza partita ha `Over25 = Under25 = 0.0`.
  ```python
  def test_build_events_families_and_counts():
      events = build_events(*_run_frames())
      model = events[events["source"] == "model"]
      assert set(model["family"]) == {"corners_total", "corners_team", "sot_total", "sot_team", "goals_ou", "1x2"}
      assert len(model[model["family"] == "corners_total"]) == 3 * 7
      assert len(model[model["family"] == "corners_team"]) == 3 * 2 * 5
      assert len(model[model["family"] == "1x2"]) == 3 * 3

  def test_build_events_corner_total_matches_probability_function():
      goals, counts = _run_frames()
      events = build_events(goals, counts)
      row = events[(events["family"] == "corners_total") & (events["line"] == 9.5)].iloc[0]
      expected = prob_over_total(counts["pred_home_corners"].values[:1], counts["disp_home_corners"].values[:1],
                                 counts["pred_away_corners"].values[:1], counts["disp_away_corners"].values[:1], 9.5)[0]
      assert row["pred"] == pytest.approx(expected)
      assert row["outcome"] == int(counts["HomeCorners"].iloc[0] + counts["AwayCorners"].iloc[0] > 9.5)

  def test_build_events_market_skips_missing_ou_odds():
      events = build_events(*_run_frames())
      market_ou = events[(events["source"] == "market") & (events["family"] == "goals_ou")]
      assert len(market_ou) == 2                           # la terza partita ha quote 0
      assert set(market_ou["line"]) == {2.5}
      assert np.isfinite(market_ou["pred"]).all()

  def test_calibration_report_one_summary_row_per_group():
      summary, bins = calibration_report(build_events(*_run_frames()))
      assert len(summary) == len(summary[["family", "market", "line", "source"]].drop_duplicates())
      assert {"n", "ece", "share_confident", "over_obs", "under_obs"} <= set(summary.columns)
      assert len(bins) == 10 * len(summary)

  def test_load_run_rejects_old_run_without_new_columns(tmp_path):
      goals, counts = _run_frames()
      goals.drop(columns=["Over25"]).to_csv(tmp_path / "multitask_old_goals.csv", index=False)
      counts.to_csv(tmp_path / "multitask_old_counts.csv", index=False)
      with pytest.raises(ValueError, match="rilancia il training"):
          load_run(tmp_path, "multitask_old")

  def test_find_run_picks_latest(tmp_path):
      for stamp in ("20261117-100000-000000", "20261118-090000-000000"):
          (tmp_path / f"multitask_{stamp}_goals.csv").write_text("x")
      assert find_run(tmp_path) == "multitask_20261118-090000-000000"

  def test_main_writes_summary_bins_and_plots(tmp_path):
      goals, counts = _run_frames()
      goals.to_csv(tmp_path / "multitask_20261117-100000-000000_goals.csv", index=False)
      counts.to_csv(tmp_path / "multitask_20261117-100000-000000_counts.csv", index=False)
      main(["--checkpoint-dir", str(tmp_path)])
      out = tmp_path / "multitask_20261117-100000-000000_calibration"
      assert (out / "summary.csv").exists() and (out / "bins.csv").exists()
      assert len(list(out.glob("reliability_*.png"))) == 6
  ```
- [ ] **Step 2: Eseguire i test e verificare che falliscano.** Run: `~/anaconda3/envs/Python3144/python.exe -m pytest tests/test_calibration.py -q`. Expected: `ImportError` su `build_events` e simili.
- [ ] **Step 3: Implementare** le funzioni del Produces.
- [ ] **Step 4: Eseguire tutta la suite.** Run: `~/anaconda3/envs/Python3144/python.exe -m pytest -q`. Expected: tutti passati.
- [ ] **Step 5: Verificare a mano il comando** su CSV finti: Run: `~/anaconda3/envs/Python3144/python.exe -m dl_model.calibration --checkpoint-dir <tmp con i CSV del test>`. Expected: la tabella riassuntiva stampata e i file scritti. **Non** lanciare il training, perché lo fa l'utente.
- [ ] **Step 6: Nessun commit.** `git status` mostra solo i file di questo piano.
