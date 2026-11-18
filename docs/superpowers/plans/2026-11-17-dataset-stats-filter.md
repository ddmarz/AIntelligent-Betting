# Drop righe senza stat, nuovo `has_stats`, tiri in porta — Piano di implementazione

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scartare dal dataset le partite senza nessuna statistica, ridefinire `has_stats` come "tiri in porta e corner tutti presenti" e far predire al modello DL i tiri in porta al posto dei tiri totali.

**Architecture:** Due interventi indipendenti. (1) `dataset/dataset.py`: nuovo drop di righe nella pipeline e nuova regola per `has_stats` dentro `add_availability_flags`. (2) `dl_model/`: il task "shots" diventa "shots_on_target", con target `HomeTarget`/`AwayTarget` e rinomina coerente di chiavi, testa e colonne di report.

**Tech Stack:** Python, pandas, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-07-12-dataset-feature-engineering-design.md` (drop e `has_stats`, Task 1) e `docs/superpowers/specs/2026-09-06-deep-learning-multitask-design.md` (tiri in porta, Task 2). Le spec sono già aggiornate: il piano non le modifica.

## Global Constraints

- **Nessun commit.** L'utente ha chiesto di non fare commit, né del design né delle modifiche. Al posto dei commit c'è un passo di verifica.
- I test DL richiedono un ambiente Python con `torch`. Né `~/anaconda3` (base) né `~/anaconda3/envs/Python3144` lo hanno: chiedere all'utente quale interprete usare prima del Task 2. I test di `dataset/dataset.py` girano anche con `~/anaconda3/python.exe -m pytest`.
- Colonne predette (definizione di `has_stats`): `HomeTarget`, `AwayTarget`, `HomeCorners`, `AwayCorners`.
- Drop: si scarta una riga se **tutte** le colonne di `STATS_COLUMNS` presenti nel DataFrame sono mancanti.
- Gli altri flag di `FLAG_GROUPS` mantengono la semantica "almeno una colonna presente". Il riempimento dei mancanti (0 per le numeriche, "Unknown" per le altre) resta invariato.
- Naming: `shots_on_target_home`/`shots_on_target_away` (target), `mean_home_shots_on_target`, `mean_away_shots_on_target`, `disp_home_shots_on_target`, `disp_away_shots_on_target` (output), `shots_on_target_head` (modello), `shots_on_target_nll` (valutazione). Nessun residuo di `shots` (tiri totali) in `dl_model/`.
- Struttura del repo: la pipeline è in `dataset/dataset.py` (importata come `from dataset.dataset import ...`, con i test lanciati dalla root del repo tramite `python -m pytest`) e i CSV sono in `dataset/files/`. I percorsi (`FILES_DIR` in `dataset/dataset.py`, `CLEAN_MATCHES_PATH` in `dl_model/data.py`) sono già relativi ai file sorgente: il piano non li tocca.
- Valori attesi sul dataset reale: 227.525 righe dopo i filtri su target e quote, 173.033 dopo il nuovo drop, di cui 112.488 con `has_stats=True`.

## Review Focus

1. **Zero tiri in porta è un valore valido.** Una partita con `HomeTarget=0` e le altre tre colonne presenti deve avere `has_stats=True`: il flag guarda la presenza (`notna`), non la veridicità del valore. Il test è nel Task 1.
2. **Riga con solo il primo tempo.** Una riga con `HTHome`/`HTAway`/`HTResult` presenti e tiri/corner mancanti va tenuta, con `has_stats=False` e tiri in porta/corner riempiti a 0. È il caso degli zeri finti, cioè il bug che motiva la modifica. Il test è nel Task 1.
3. **Statistiche parziali.** Una riga con tiri in porta presenti ma un corner mancante deve avere `has_stats=False`, non True. Il test è nel Task 1.
4. **I tiri totali non decidono più nulla.** Una riga con `HomeShots`/`AwayShots` mancanti ma tiri in porta e corner presenti deve avere `has_stats=True`. Il dataset DL deve leggere `HomeTarget` anche quando `HomeShots` ha un valore diverso. I test sono nel Task 1 e nel Task 2.
5. **Un `matches_clean.csv` vecchio usa ancora la vecchia regola.** Il CSV generato prima di questa modifica ha il vecchio `has_stats` e il modello lo userebbe senza errori. Non è testabile nel codice: va rigenerato (fuori scope, ma da ricordare all'utente alla consegna).

---

### Task 1: Drop delle righe senza statistiche e nuovo `has_stats` in `dataset/dataset.py`

**Files:**
- Modify: `dataset/dataset.py`: costanti (`STATS_COLUMNS`, `FLAG_GROUPS`), `drop_missing_odds`/`drop_missing_target`/`add_availability_flags`, `build_clean_dataset`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: nulla.
- Produces: `PREDICTED_STATS_COLUMNS: list[str]`, `drop_missing_stats(matches: pd.DataFrame) -> pd.DataFrame`; `add_availability_flags(matches)` con la stessa firma e una nuova regola per `has_stats`.

- [ ] **Step 1: Scrivere i test che falliscono in `tests/test_dataset.py`**

Import: `from dataset.dataset import PREDICTED_STATS_COLUMNS, STATS_COLUMNS, add_availability_flags, drop_missing_stats`.

Fixture `_stats_matches()`: un DataFrame con tutte le colonne di `STATS_COLUMNS` più `Division`, `HomeTeam`, `AwayTeam`, e queste righe (NaN dove non indicato):

| riga | contenuto |
|---|---|
| `none` | tutte le stat NaN |
| `ht_only` | solo `HTHome=1`, `HTAway=0`, `HTResult="H"` |
| `full` | tutte le 15 stat valorizzate, con `HomeTarget=0` |
| `partial` | `HomeTarget`, `AwayTarget`, `HomeCorners` valorizzati, `AwayCorners` NaN |
| `no_total_shots` | `HomeTarget`, `AwayTarget`, `HomeCorners`, `AwayCorners` valorizzati, `HomeShots`/`AwayShots` NaN |

```python
def test_predicted_stats_columns():
    assert PREDICTED_STATS_COLUMNS == ["HomeTarget", "AwayTarget", "HomeCorners", "AwayCorners"]

def test_drop_missing_stats_removes_only_rows_without_any_stat():
    result = drop_missing_stats(_stats_matches())
    assert list(result["HomeTeam"]) == ["ht_only", "full", "partial", "no_total_shots"]

def test_drop_missing_stats_uses_only_present_columns():
    matches = _stats_matches().drop(columns=["HTHome", "HTAway", "HTResult"])
    result = drop_missing_stats(matches)
    assert list(result["HomeTeam"]) == ["full", "partial", "no_total_shots"]

def test_has_stats_requires_all_predicted_columns():
    result = add_availability_flags(drop_missing_stats(_stats_matches())).set_index("HomeTeam")
    assert result.loc["ht_only", "has_stats"] == False
    assert result.loc["full", "has_stats"] == True          # HomeTarget=0 è un valore valido
    assert result.loc["partial", "has_stats"] == False
    assert result.loc["no_total_shots", "has_stats"] == True

def test_missing_stats_are_still_filled():
    result = add_availability_flags(drop_missing_stats(_stats_matches())).set_index("HomeTeam")
    assert result.loc["ht_only", "HomeTarget"] == 0
    assert result.loc["partial", "AwayCorners"] == 0
    assert result[STATS_COLUMNS].notna().all().all()
```

(`HomeTeam` fa da etichetta di riga. Usare `== False`/`== True` e non `is`, perché i valori sono `numpy.bool_`.)

- [ ] **Step 2: Eseguire i test e verificare che falliscano**

Run: `~/anaconda3/python.exe -m pytest tests/test_dataset.py -v`
Expected: FAIL con `ImportError` su `PREDICTED_STATS_COLUMNS` / `drop_missing_stats`.

- [ ] **Step 3: Implementare in `dataset/dataset.py`**

- `PREDICTED_STATS_COLUMNS = ["HomeTarget", "AwayTarget", "HomeCorners", "AwayCorners"]`, accanto a `STATS_COLUMNS`, con un commento che spiega che sono le stat che il modello predice e che definiscono `has_stats`.
- `drop_missing_stats(matches)`, accanto a `drop_missing_odds`/`drop_missing_target`: scarta le righe in cui tutte le colonne di `STATS_COLUMNS` presenti nel DataFrame sono NaN.
- `add_availability_flags`: `has_stats` = `matches[PREDICTED_STATS_COLUMNS].notna().all(axis=1)`, calcolato **prima** del `fillna`. La voce `"has_stats": STATS_COLUMNS` resta in `FLAG_GROUPS`, perché serve per il riempimento delle 15 colonne e per `select_output_columns`. Solo il valore del flag `has_stats` usa la nuova regola, gli altri flag restano invariati.
- `build_clean_dataset`: chiamare `drop_missing_stats` subito dopo `drop_missing_odds`.

- [ ] **Step 4: Eseguire i test e verificare che passino**

Run: `~/anaconda3/python.exe -m pytest tests/test_dataset.py -v`
Expected: 5 passed.

- [ ] **Step 5: Verificare i conteggi sul dataset reale (senza scrivere `matches_clean.csv`)**

Run: `~/anaconda3/python.exe -c "from dataset.dataset import build_clean_dataset; d = build_clean_dataset(); print(len(d), int(d['has_stats'].sum()))"`
Expected: `173033 112488`. Ci vuole qualche minuto. Il file `dataset/files/matches_clean.csv` non va sovrascritto: rigenerarlo è fuori scope.

- [ ] **Step 6: Nessun commit (richiesta dell'utente).** Controllare con `git status` che le modifiche siano solo in `dataset/dataset.py` e `tests/test_dataset.py`.

---

### Task 2: Tiri in porta al posto dei tiri in `dl_model/`

Tutto in un solo task: le chiavi sono condivise fra `data.py`, `model.py`, `train.py` ed `evaluation.py`, quindi una rinomina parziale lascerebbe la suite rotta.

**Files:**
- Modify: `dl_model/data.py:64-65`, `dl_model/model.py:66,75,81-84`, `dl_model/train.py:43-47,55,113,126-127`, `dl_model/evaluation.py:89-104`
- Modify (test): `tests/test_dl_data.py:15-35,58-75`, `tests/test_dl_model.py:16-20`, `tests/test_dl_evaluation.py:43-62,84-93`, `tests/test_dl_train.py:28-37,75`

**Interfaces:**
- Consumes: la colonna `has_stats` di `matches_clean.csv` con la semantica del Task 1. Il codice DL la legge come prima.
- Produces: chiavi target `shots_on_target_home`/`_away`; chiavi output `mean_home_shots_on_target`, `mean_away_shots_on_target`, `disp_home_shots_on_target`, `disp_away_shots_on_target`; attributo `MultiTaskMatchModel.shots_on_target_head` (`nn.Linear(fusion_dim, 4)`); colonne di `evaluate_counts` e di `count_results` di `run_walk_forward` = `{"shots_on_target_nll", "corners_nll"}`.

- [ ] **Step 1: Aggiornare i test**

- Fixture di `test_dl_data.py` e `test_dl_evaluation.py`: aggiungere `HomeTarget`/`AwayTarget` (`[5, 3, 2]`/`[4, 4, 1]` in `test_dl_data.py`, `[5, 2]`/`[4, 1]` in `test_dl_evaluation.py`). Tenere `HomeShots`/`AwayShots` con valori diversi, così si dimostra che il dataset legge la colonna giusta.
- `test_match_dataset_returns_expected_tensor_shapes_and_values`: aggiungere
  ```python
  assert targets["shots_on_target_home"].item() == 5.0
  assert targets["shots_on_target_away"].item() == 4.0
  assert "shots_home" not in targets
  ```
- `test_dl_model.py`: in `expected_keys`, sostituire le 4 chiavi `*_shots` con `mean_home_shots_on_target`, `mean_away_shots_on_target`, `disp_home_shots_on_target`, `disp_away_shots_on_target`.
- `test_dl_evaluation.py`: `assert set(result.columns) == {"shots_on_target_nll", "corners_nll"}` e `np.isfinite(result["shots_on_target_nll"])`.
- `test_dl_train.py`: in `_fake_matches`, aggiungere `"HomeTarget": int(np.random.poisson(5))`, `"AwayTarget": int(np.random.poisson(4))`. In `test_run_walk_forward_returns_expected_report_shapes`, cambiare le colonne attese in `{"shots_on_target_nll", "corners_nll"}`.

- [ ] **Step 2: Eseguire i test e verificare che falliscano**

Run: `<python con torch> -m pytest tests/ -v`
Expected: FAIL su `KeyError`/assert delle chiavi `shots_on_target_*` e delle colonne `shots_on_target_nll`. I test di `test_dataset.py` e `test_dl_losses.py` passano.

- [ ] **Step 3: Implementare la rinomina**

- `data.py`: `shots_on_target_home`/`_away` letti da `HomeTarget`/`AwayTarget`.
- `model.py`: `shots_on_target_head` e le 4 chiavi di output del Produces, nello stesso ordine di prima (media casa, media trasferta, dispersione casa, dispersione trasferta).
- `train.py`: in `compute_total_loss`, la loss usa le nuove chiavi (variabili `shots_on_target_loss*`) ed è mascherata con `has_stats` come prima. Colonne del DataFrame vuoto: `["shots_on_target_nll", "corners_nll"]`. Report: `"Shots-on-target/corners task (has_stats rows, Negative-Binomial NLL):"` e `"  shots on target NLL: ..."`.
- `evaluation.py`: `evaluate_counts` produce `shots_on_target_nll`.

- [ ] **Step 4: Eseguire i test e verificare che passino**

Run: `<python con torch> -m pytest tests/ -v`
Expected: tutti passati.

- [ ] **Step 5: Controllare che non restino riferimenti ai tiri totali**

Run: `grep -rn -i "shots" dl_model/ | grep -v -i "shots_on_target\|shots on target\|Shots-on-target"`
Expected: nessun output.

- [ ] **Step 6: Nessun commit (richiesta dell'utente).** Controllare con `git status` che le modifiche siano solo nei file elencati.
