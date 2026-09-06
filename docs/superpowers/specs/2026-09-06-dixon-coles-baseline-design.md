# Fase 2 (parte 1) — Modello Dixon-Coles per il risultato

## Contesto

Il progetto League Prediction si sviluppa in tre fasi indipendenti:

1. Pulizia dataset e feature engineering (Fase 1, completata — vedi `docs/superpowers/specs/2026-07-12-dataset-feature-engineering-design.md`)
2. Modelli predittivi: deep learning + Dixon-Coles (e possibile confronto con XGBoost)
3. Tool di value betting

La Fase 2 include più modelli sufficientemente diversi tra loro (Dixon-Coles, deep learning, eventualmente XGBoost) da giustificare una spec e un piano di implementazione separati per ciascuno. Questa spec copre **solo** il primo sotto-progetto: un modello Dixon-Coles classico per il risultato delle partite (gol), usato come baseline statistica interpretabile prima di costruire il deep learning.

Modelli per altre statistiche di partita (corner, falli, cartellini) sono un sotto-progetto successivo, non coperto da questa spec.

## Obiettivo

Costruire, per un piccolo insieme di campionati target, un modello Dixon-Coles (Poisson bivariato con correzione per i risultati bassi e decadimento temporale) che stimi la distribuzione di probabilità dei punteggi di una partita, da cui derivare le probabilità di esito 1X2. Il modello serve da baseline indipendente dalle feature di Fase 1: se il deep learning (che userà quelle feature) non supera questa baseline, è un segnale di problemi nel deep learning, non un'indicazione che le feature siano inutili.

## Scope dati

- Sorgente: `matches_clean.csv` (output di Fase 1).
- Campionati target: Big 5 europei — `Division ∈ {I1, E0, SP1, D1, F1}` (Serie A, Premier League, La Liga, Bundesliga, Ligue 1). Gli altri 33 campionati restano nel dataset per un uso futuro (deep learning) ma non sono usati da questo modello.
- Colonne usate: `Division`, `MatchDate`, `HomeTeam`, `AwayTeam`, `FTHome`, `FTAway`, `OddHome`, `OddDraw`, `OddAway` (le ultime tre solo per il benchmark di valutazione, non per il fit).
- Nessun'altra feature di Fase 1 (Elo, forma, rolling goals, cluster `C_*`) entra nel modello: è una scelta deliberata per mantenerlo come baseline minimale basata solo sulla storia dei gol. Un'estensione con covariate (Dixon-Coles "GLM-style") è una possibile iterazione futura, fuori scope qui.

## Formulazione del modello

Per una partita tra squadra di casa `i` e ospite `j`, disputata al campionato `L`:

- `λ_home = exp(attack_i + defense_j + home_advantage_L)`
- `λ_away = exp(attack_j + defense_i)`
- `Gol_home ~ Poisson(λ_home)`, `Gol_away ~ Poisson(λ_away)`, condizionatamente indipendenti dato `(λ_home, λ_away)`.

**Correzione per risultati bassi.** La probabilità congiunta è corretta da un fattore `τ`, applicato solo ai quattro risultati con meno di 2 gol per squadra:

```
τ(0,0) = 1 − λ_home·λ_away·ρ_L
τ(0,1) = 1 + λ_home·ρ_L
τ(1,0) = 1 + λ_away·ρ_L
τ(1,1) = 1 − ρ_L
τ(x,y) = 1   altrimenti

P(gol_home=x, gol_away=y) = τ(x,y) · Poisson(x; λ_home) · Poisson(y; λ_away)
```

**Parametri stimati per campionato `L` (via massima verosimiglianza pesata):**
- `attack_i`, `defense_i` per ogni squadra `i` che ha giocato in `L` nella finestra di fit.
- `home_advantage_L` (uno scalare).
- `ρ_L` (uno scalare).

**Vincolo di identificabilità:** la media degli `attack_i` all'interno del campionato è fissata a 0 durante l'ottimizzazione (senza questo vincolo, attack e defense sono definiti solo a meno di una costante additiva comune).

**Decadimento temporale.** Ogni partita di allenamento con data `d`, valutata "as of" una data di riferimento `t`, pesa `exp(−ξ · (t − d)_giorni)` nella somma della log-verosimiglianza pesata:

```
log L(attack, defense, home_advantage, ρ) =
    Σ_partite  exp(−ξ · giorni_trascorsi) · log P(gol_home, gol_away | parametri)
```

`ξ` **non** è stimato per massima verosimiglianza insieme agli altri parametri (la MLE su ξ degenera verso valori estremi). È invece scelto per validazione (vedi sotto) e condiviso tra tutti i campionati target, non stimato separatamente per ciascuno.

## Strategia di fit

- Un fit produce un `DixonColesModel` per un singolo campionato `L`, "as of" una data `t`: usa tutte le partite di `L` con `MatchDate < t`, pesate per decadimento temporale.
- Ottimizzazione numerica di `attack, defense, home_advantage, ρ` via `scipy.optimize.minimize` (log-verosimiglianza negativa pesata), con il vincolo di identificabilità sopra.
- Non c'è rifit "per stagione": il decadimento temporale rende obsoleto il concetto di confine di stagione per il fit (la Season resta rilevante solo per organizzare la valutazione, vedi sotto).

## Selezione dell'iperparametro ξ

- Candidati: una piccola griglia, es. `{0, 0.0005, 0.001, 0.002, 0.005}` (unità: al giorno).
- Per ciascun candidato, si esegue l'intera procedura di valutazione walk-forward (sotto) su tutti i campionati target e si calcola la log-loss media aggregata.
- Si sceglie il valore di ξ con la log-loss media più bassa. Un singolo ξ è condiviso da tutti i campionati target (nessuna ottimizzazione per-campionato in questa iterazione).

## Valutazione (walk-forward)

- Per ciascun campionato target, le prime ~3 stagioni di storico servono solo da burn-in per il fit iniziale (nessuna partita in questa finestra viene valutata).
- Dalla quarta stagione in poi, procedura walk-forward stagione per stagione: per ogni stagione `S`, fit del modello usando tutto lo storico con `MatchDate` antecedente alla prima partita di `S` (pesato per decadimento), poi predizione di tutte le partite di `S`.
- **Metrica primaria:** log-loss (cross-entropy) delle probabilità 1X2 predette contro l'esito reale, aggregata su tutte le partite valutate.
- **Benchmark di confronto (diagnostico, non un obiettivo da battere):** log-loss delle probabilità implicite di mercato, derivate da `OddHome/OddDraw/OddAway` normalizzate per rimuovere il margine bookmaker (`p_i = (1/quota_i) / Σ(1/quota_k)`). Serve a capire quanto segnale aggiuntivo (o mancanza) ha il modello rispetto al mercato, non come criterio di accettazione.

## Struttura del codice

Nuovo modulo `dixon_coles.py` (stesso livello di `dataset.py`):

- `fit_dixon_coles(matches, as_of_date, xi) -> DixonColesModel`: fitta un modello per un singolo campionato, usando solo le partite con `MatchDate < as_of_date`. `DixonColesModel` contiene `attack: dict[str, float]`, `defense: dict[str, float]`, `home_advantage: float`, `rho: float`, `teams: list[str]`.
- `predict_score_matrix(model, home_team, away_team, max_goals=10) -> np.ndarray`: matrice di probabilità `P(gol_home=x, gol_away=y)` per `x, y` da 0 a `max_goals`.
- `match_outcome_probs(score_matrix) -> tuple[float, float, float]`: probabilità `(P(1), P(X), P(2))` derivate sommando la matrice sopra/sotto/sulla diagonale.

Script/notebook di valutazione separato (es. `evaluate_dixon_coles.py`) che orchestra: caricamento dati, griglia ξ, walk-forward su tutti i campionati target, calcolo log-loss modello vs benchmark di mercato, stampa dei risultati.

## Output

- Nessuna persistenza su disco del modello fitted in questa iterazione (il salvataggio per uso in produzione, in vista della Fase 3, è rimandato a quando si integrerà un modello scelto nel tool di value betting).
- Il deliverable di questa iterazione è: il modulo `dixon_coles.py` riusabile, e un report (script o notebook) con la log-loss del modello per campionato/stagione confrontata al benchmark di mercato, più il valore di ξ selezionato.

## Non in scope per questa iterazione

- Modelli per corner, falli, cartellini (sotto-progetto successivo, con la stessa logica Poisson ma modelli dedicati — vedi nota sotto).
- Deep learning, XGBoost.
- Estensione di Dixon-Coles con covariate (Elo, forma, cluster) — variante GLM possibile in futuro.
- ξ stimato per-campionato invece che condiviso.
- Persistenza/serializzazione del modello per produzione.
- Campionati oltre ai Big 5.

## Nota sui prossimi sotto-progetti di Fase 2

Dopo la validazione di questo modello (baseline sul risultato), i prossimi passi discussi sono:

1. Modelli statistici (Poisson, senza la correzione ρ specifica al risultato calcio) per corner, falli, cartellini — stesso impianto di fit/valutazione, sotto-progetto e spec a parte.
2. Modello di deep learning che usa le feature complete di Fase 1 (Elo, forma, rolling goals, cluster) su tutti i 38 campionati, da confrontare contro questa baseline.
