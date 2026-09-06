# Fase 2 — Modello Deep Learning multi-task (gol, tiri, corner)

## Contesto

Questa spec copre un sotto-progetto di Fase 2. L'obiettivo è un modello di deep learning che sfrutti le feature complete di Fase 1 (Elo, forma, rolling goals, cluster) su tutti i 38 campionati.

Ispirazione: Gao, Cheng & Gao, *"Predicting sport event outcomes using deep learning"* (PeerJ Computer Science, 2025) — un'architettura ibrida 1D CNN + Transformer per la classificazione 1X2. Questa spec adatta quell'architettura a un problema diverso: non classificazione a 3 classi, ma **regressione di conteggi multi-task** (gol, tiri, corner).

## Obiettivo

Un singolo modello che, data una partita, predica simultaneamente:
- Gol attesi (casa/trasferta), da cui derivare 1X2, Over/Under, punteggio esatto.
- Tiri attesi (casa/trasferta).
- Corner attesi (casa/trasferta).

## Scope dati

- Task gol: tutti i 38 campionati del dataset (`matches_clean.csv`), sfruttando la stessa logica della spec di Fase 1 (pattern generali appresi anche da campionati non target).
- Task tiri/corner: stesso dataset, ma la loss di questi due task è calcolata **solo** sulle righe con `has_stats=True` (dove i dati erano realmente disponibili); le altre righe partecipano comunque al task gol.
- Nessuna feature di mercato (quote) in input: il modello deve predire dai dati di partita/storico, non riflettere il mercato — necessario per un confronto onesto con le quote in fase di valutazione, e per essere potenzialmente utile alla Fase 3 (value betting: un modello che copia il mercato non trova mai valore).
- Valutazione comparativa con il mercato ristretta ai Big 5 (`I1, E0, SP1, D1, F1`, `dl_model.data.TARGET_LEAGUES`).

## Feature di input

**Categoriche (ramo Transformer, via embedding):**
- `Division`, `HomeTeam`, `AwayTeam`.

**Continue (ramo CNN 1D, normalizzate):**
- `EloDiff`
- `HomeAvgGoalsFor5`, `HomeAvgGoalsAgainst5`, `AwayAvgGoalsFor5`, `AwayAvgGoalsAgainst5`
- `Form3Home`, `Form5Home`, `Form3Away`, `Form5Away`
- `C_LTH`, `C_LTA`, `C_VHD`, `C_VAD`, `C_HTB`, `C_PHB`
- Flag di disponibilità come indicatori 0/1: `has_elo`, `has_rolling_stats`, `has_form`, `has_cluster_data`

## Architettura

Adattamento diretto dell'architettura del paper:

- **Ramo CNN 1D**: il vettore di feature continue (19 valori) trattato come sequenza 1D a un canale. Due layer convoluzionali (kernel piccolo, es. 3), ciascuno seguito da Leaky ReLU e Batch Normalization, poi max pooling. Connessione residua: l'input passa anche per un fully-connected che ne adatta la dimensione, sommato all'output del ramo convoluzionale.
- **Ramo Transformer**: le tre feature categoriche mappate a embedding (dimensioni tipiche: 16 per squadra, 8 per campionato — appresi durante il training, non fissi), trattate come sequenza di 3 token, elaborate da un blocco di self-attention multi-head (es. 4 head) con feed-forward e layer normalization.
- **Fusione**: gli output dei due rami si concatenano, passano per un fully-connected condiviso.
- **Teste di output** (invece della softmax a 3 classi del paper):
  - Testa gol: 2 valori `z_home_goals`, `z_away_goals` → `λ = exp(z)`.
  - Testa tiri: 2 medie (`z_home_shots`, `z_away_shots` → `λ = exp(z)`) + 2 parametri di dispersione.
  - Testa corner: stessa struttura della testa tiri, valori separati.

## Distribuzioni e loss

- **Gol**: Poisson. `loss = λ − y·log(λ)` per ciascun lato (casa/trasferta), sommate. λ è l'output della rete (`exp(z)` dell'ultimo layer della testa gol).
- **Tiri e corner**: Binomiale Negativa (gestisce la sovradispersione tipica di questi conteggi, a differenza della Poisson pura). Parametrizzazione media-dispersione: la rete produce sia la media attesa che un parametro di dispersione per ciascuna metrica/lato.
- **Loss totale**: somma delle tre loss (pesi uguali in questa iterazione — se una task dominasse le altre durante il training, ribilanciare i pesi è un raffinamento futuro, non affrontato qui).
- Le righe senza `has_stats` non contribuiscono alle loss di tiri/corner (mascherate), ma contribuiscono comunque alla loss dei gol.

## Strategia di training: walk-forward con fine-tuning

Ispirata direttamente al paper (che riallena con una finestra scorrevole):

- Le stagioni (colonna `Season`, calcolata su tutti i 38 campionati insieme) sono ordinate cronologicamente. Le prime N stagioni (burn-in, da definire in fase di implementazione in base a quante partite/squadre servono per stabilizzare gli embedding) servono per un **training iniziale completo** (molte epoche).
- Da lì in avanti, un passo walk-forward per stagione: poche epoche di **fine-tuning** sui dati della stagione appena resa disponibile (non un retraining completo da zero — troppo costoso per una rete neurale su questo volume di dati), poi valutazione sulla stagione successiva.
- Questo è più economico del retraining completo ripetuto, e riflette come il modello verrebbe effettivamente aggiornato in un uso reale (in vista della Fase 3).

## Valutazione

- **Gol**: dalla coppia `(λ_home, λ_away)` si deriva una matrice di probabilità sui punteggi (Poisson prodotto, senza alcuna correzione per i risultati bassi — la rete può già imparare eventuali correlazioni dai dati). Da questa matrice si derivano le probabilità 1X2, confrontate via log-loss contro le probabilità implicite di mercato (quote de-vigged), sui Big 5, nella stessa finestra di valutazione walk-forward.
- **Tiri e corner**: nessun benchmark di mercato disponibile. Si riporta la negative log-likelihood Binomiale Negativa media su hold-out, come misura di quanto bene il modello approssima le distribuzioni osservate.

## Struttura del codice

Package `dl_model/`, completamente autonomo (nessuna dipendenza da altri modelli):
- `dl_model/data.py`: preparazione dei dati — costruzione vocabolari per le feature categoriche (squadra, campionato), normalizzazione delle feature continue, `Dataset` PyTorch che restituisce feature categoriche/continue e i sei target (gol, tiri, corner per lato) più la maschera `has_stats`; `TARGET_LEAGUES` per lo scope di valutazione.
- `dl_model/model.py`: il modulo PyTorch con i due rami (CNN 1D, Transformer), la fusione, e le tre teste di output.
- `dl_model/losses.py`: `poisson_nll(lambda_pred, target) -> Tensor` e `negative_binomial_nll(mean_pred, dispersion_pred, target) -> Tensor`.
- `dl_model/evaluation.py`: `poisson_score_matrix`, `match_outcome_probs`, `implied_probabilities`, `log_loss` (utility di valutazione generiche, non specifiche a nessun modello) più `evaluate_goals`/`evaluate_counts`.
- `dl_model/train.py`: orchestrazione — training iniziale su burn-in, loop di fine-tuning walk-forward, valutazione (log-loss gol vs mercato sui Big 5, NB-NLL tiri/corner), report finale.

## Non in scope per questa iterazione

- Falli e cartellini (task aggiuntivi, non coperti qui).
- Ricerca approfondita di iperparametri/architettura (dimensioni embedding, numero di layer, learning rate) — si parte da valori ragionevoli di default, senza una fase di tuning sistematico.
- Bilanciamento automatico dei pesi tra le tre loss.
- Persistenza/serializzazione del modello per produzione (rimandata all'integrazione nella Fase 3). Il checkpoint usato per il fine-tuning walk-forward è interno alla procedura di valutazione, non un artefatto da riusare altrove.
- Ottimizzazione per GPU/hardware specifico.
- Confronto con altri modelli (baseline statistiche, gradient boosting): non più nello scope di questo progetto.
