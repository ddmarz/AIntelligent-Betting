# Fase 2 — Modello Deep Learning multi-task (gol, tiri in porta, corner)

## Contesto

Questa spec copre un sotto-progetto di Fase 2. L'obiettivo è un modello di deep learning che sfrutti le feature complete di Fase 1 (Elo, forma, rolling goals, cluster) su tutti i campionati del dataset pulito (22 su 38: i 16 senza nessuna statistica vengono scartati in Fase 1).

Ispirazione: Gao, Cheng & Gao, *"Predicting sport event outcomes using deep learning"* (PeerJ Computer Science, 2025) — un'architettura ibrida 1D CNN + Transformer per la classificazione 1X2. Questa spec adatta quell'architettura a un problema diverso: non classificazione a 3 classi, ma **regressione di conteggi multi-task** (gol, tiri in porta, corner).

## Obiettivo

Un singolo modello che, data una partita, predica simultaneamente:
- Gol attesi (casa/trasferta), da cui derivare 1X2, Over/Under, punteggio esatto.
- Tiri in porta attesi (casa/trasferta), colonne `HomeTarget`/`AwayTarget`. Sostituiscono i tiri totali (`HomeShots`/`AwayShots`) delle versioni precedenti di questa spec.
- Corner attesi (casa/trasferta).

## Scope dati

- Task gol: tutti i campionati del dataset (`matches_clean.csv`), sfruttando la stessa logica della spec di Fase 1 (pattern generali appresi anche da campionati non target).
- Task tiri in porta/corner: stesso dataset, ma la loss di questi due task è calcolata **solo** sulle righe con `has_stats=True`. Secondo la spec di Fase 1, `has_stats` è True solo se `HomeTarget`, `AwayTarget`, `HomeCorners` e `AwayCorners` sono tutti disponibili (112.488 righe su 173.033). Le altre righe partecipano comunque al task gol.
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
  - Testa tiri in porta (`shots_on_target_head`): 2 medie (`z_home_shots_on_target`, `z_away_shots_on_target` → `λ = exp(z)`) + 2 parametri di dispersione. Output: `mean_home_shots_on_target`, `mean_away_shots_on_target`, `disp_home_shots_on_target`, `disp_away_shots_on_target`.
  - Testa corner: stessa struttura della testa tiri in porta, valori separati.

## Distribuzioni e loss

- **Gol**: Poisson. `loss = λ − y·log(λ)` per ciascun lato (casa/trasferta), sommate. λ è l'output della rete (`exp(z)` dell'ultimo layer della testa gol).
- **Tiri in porta e corner**: Binomiale Negativa (gestisce la sovradispersione tipica di questi conteggi, a differenza della Poisson pura). Sulle righe con `has_stats=True` i tiri in porta hanno media 5,1 e varianza 7,6 in casa, 4,1 e 5,9 in trasferta: la sovradispersione c'è. Parametrizzazione media-dispersione: la rete produce sia la media attesa che un parametro di dispersione per ciascuna metrica/lato.
- **Loss totale**: somma delle tre loss (pesi uguali in questa iterazione — se una task dominasse le altre durante il training, ribilanciare i pesi è un raffinamento futuro, non affrontato qui).
- Le righe senza `has_stats` non contribuiscono alle loss di tiri in porta/corner (mascherate), ma contribuiscono comunque alla loss dei gol.

## Strategia di training: walk-forward con fine-tuning

Ispirata direttamente al paper (che riallena con una finestra scorrevole):

- Le stagioni (colonna `Season`, calcolata su tutti i campionati del dataset insieme) sono ordinate cronologicamente. Le prime N stagioni di burn-in (`BURN_IN_SEASONS = 5`) servono per un **training iniziale completo** (`INITIAL_EPOCHS = 20`).
- Da lì in avanti, un passo walk-forward per stagione: poche epoche di **fine-tuning** sui dati della stagione appena resa disponibile (non un retraining completo da zero — troppo costoso per una rete neurale su questo volume di dati), poi valutazione sulla stagione successiva.
- Questo è più economico del retraining completo ripetuto, e riflette come il modello verrebbe effettivamente aggiornato in un uso reale (in vista della Fase 3).
- **Normalizzazione senza informazioni dal futuro:** medie e deviazioni standard delle feature continue sono calcolate **solo sulle stagioni di burn-in** e restano fisse per tutto il walk-forward. Così nessuna statistica delle stagioni di valutazione entra nel modello. I vocabolari di squadre e campionati sono invece costruiti su tutte le partite, perché contengono solo i nomi e non informazione sui risultati.
- **Riproducibilità:** ogni run imposta il seed `SEED = 1926` per `random`, `numpy` e `torch` (CUDA compresa), con cuDNN in modalità deterministica. Due run con gli stessi dati e parametri danno gli stessi risultati.
- **Dispositivo:** il modello gira su GPU (`cuda`) se disponibile, altrimenti su CPU. Il training e la valutazione spostano ogni batch sul dispositivo del modello.

## Checkpoint

Alla fine del walk-forward il modello finale, che ha visto tutte le stagioni, viene salvato in `checkpoints/multitask_<data-ora>.pt`, nella root del repo. La cartella è esclusa da git e ogni run crea un file nuovo senza sovrascrivere i precedenti. Il file contiene tutto il necessario per riusare il modello:
- i pesi (`model_state_dict`), salvati su CPU così si caricano anche senza GPU;
- i vocabolari (`division_vocab`, `team_vocab`);
- medie e deviazioni standard di normalizzazione (`means`, `stds`, come dizionari colonna → valore);
- seed, iperparametri (`burn_in_seasons`, `initial_epochs`, `fine_tune_epochs`, `learning_rate`, `batch_size`) e ultima stagione vista (`last_season`).

Si carica con `torch.load(path, weights_only=True)`.

Accanto al checkpoint, con lo stesso nome, vengono salvati i risultati della valutazione walk-forward partita per partita, per le analisi successive (per campionato, per stagione, calibrazione):
- `multitask_<data-ora>_goals.csv` (partite dei Big 5): campionato, stagione, data, squadre, risultato, `lambda_home`/`lambda_away`, probabilità 1X2 del modello (`model_p_*`) e del mercato senza margine (`market_p_*`), log-loss di entrambi.
- `multitask_<data-ora>_counts.csv` (partite con `has_stats=True`): campionato, stagione, data, squadre, tiri in porta e corner reali e predetti (`pred_*`, la media della Binomiale Negativa), NLL di tiri in porta e corner.

## Valutazione

- **Gol**: dalla coppia `(λ_home, λ_away)` si deriva una matrice di probabilità sui punteggi (Poisson prodotto, senza alcuna correzione per i risultati bassi — la rete può già imparare eventuali correlazioni dai dati). Da questa matrice si derivano le probabilità 1X2, confrontate via log-loss contro le probabilità implicite di mercato (quote de-vigged), sui Big 5, nella stessa finestra di valutazione walk-forward.
- **Tiri in porta e corner**: nessun benchmark di mercato disponibile. Si riporta la negative log-likelihood Binomiale Negativa media su hold-out, come misura di quanto bene il modello approssima le distribuzioni osservate.

## Calibrazione

### Perché

L'obiettivo è potersi fidare delle probabilità del modello sulle linee over/under: se dice "70% che i corner totali siano più di 6,5", sull'insieme delle partite in cui dice 70% l'over deve uscire circa 7 volte su 10. Non interessa l'accuratezza della singola predizione, interessa che le percentuali siano affidabili, soprattutto quando il modello è molto convinto. La calibrazione è necessaria per cercare un edge, ma non basta: un modello calibrato che dice sempre 55% è affidabile ma inutile, quindi si misura anche quanto spesso il modello è convinto.

### Mercati e linee

Tutte le probabilità vengono dalle distribuzioni che il modello già produce:
- **Corner** (Binomiale Negativa per squadra, `scipy.stats.nbinom(n=disp, p=disp/(disp+mean))`, la stessa parametrizzazione della loss):
  - totali partita: over 6,5 / 7,5 / 8,5 / 9,5 / 10,5 / 11,5 / 12,5;
  - per squadra, con casa e trasferta insieme: over 2,5 / 3,5 / 4,5 / 5,5 / 6,5.
- **Tiri in porta** (stessa distribuzione):
  - totali partita: over 6,5 / 7,5 / 8,5 / 9,5 / 10,5 / 11,5;
  - per squadra: over 2,5 / 3,5 / 4,5 / 5,5.
- **Gol under/over** (Poisson, dalle λ): totali partita over 1,5 / 2,5 / 3,5. La somma di due Poisson indipendenti è una Poisson con λ = λ_casa + λ_trasferta.
- **1X2**: vittoria casa, pareggio e vittoria trasferta, ciascuno trattato come evento sì/no.

Per i totali di corner e tiri in porta la distribuzione è la somma, per convoluzione, delle distribuzioni di casa e trasferta, che il modello tratta come indipendenti. Nei dati però i corner di casa e trasferta hanno correlazione −0,20: se il modello sui totali risulta poco convinto, la causa probabile è questa.

L'under di una linea è il complemento dell'over: la calibrazione è simmetrica, quindi basta l'over.

**Riferimento di mercato:** per 1X2 e gol over/under 2,5 si calcolano le stesse misure anche sulle probabilità di mercato (quote Bet365, margine tolto in proporzione), per vedere se il modello è calibrato quanto il mercato.

### Misure, per ogni mercato e linea

- **Tabella di affidabilità:** probabilità predetta in 10 fasce di uguale ampiezza (0–10%, …, 90–100%). Per ogni fascia: numero di partite, probabilità media predetta, frequenza reale e intervallo di confidenza al 95% della frequenza reale (Wilson). Con poche partite l'intervallo è largo e lo si vede.
- **Errore di calibrazione medio (ECE):** media, pesata per numero di partite, della distanza fra probabilità predetta e frequenza reale nelle fasce. È un numero solo per confrontare linee e mercati.
- **Quando il modello è convinto:** sulle partite con P(over) ≥ 70% e su quelle con P(over) ≤ 30% si riportano numero di partite, probabilità media predetta e frequenza reale, con l'intervallo di Wilson. Risponde direttamente a "quando è convinto, ci si può fidare?". La soglia del 70% è un parametro.
- **Quanto spesso è convinto:** quota di partite con P(over) ≥ 70% o ≤ 30%.
- **Frequenza di base (`base_rate`) e Brier skill score (`brier_skill`):** quanto spesso l'evento si verifica in media, e quanto il modello fa meglio di chi predice sempre quella media (0 = non aggiunge nulla, 1 = perfetto). Servono perché su linee sbilanciate (es. corner totali over 6,5, che esce circa l'86% delle volte, o il pareggio nell'1X2) anche un modello che ripete la media risulta "convinto" e calibrato: in quei casi la calibrazione e i casi convinti vanno letti accanto a `base_rate` e `brier_skill`.

Si usano solo le predizioni walk-forward, fatte prima di vedere la stagione, cioè quelle salvate nei CSV per partita del run. Il report misura quindi la calibrazione che si sarebbe avuta davvero.

### Dati necessari dal run

- `_counts.csv` aggiunge le dispersioni: `disp_home_shots_on_target`, `disp_away_shots_on_target`, `disp_home_corners`, `disp_away_corners`.
- `_goals.csv` aggiunge i gol reali (`FTHome`, `FTAway`), necessari per sapere se l'over gol è uscito, e le quote grezze: `OddHome`, `OddDraw`, `OddAway`, `Over25`, `Under25`. Le quote servono per il riferimento di mercato sull'over/under 2,5 e serviranno per il backtest del ROI. Quando le quote over/under mancano valgono `0` nel dataset: quelle righe sono escluse dal riferimento di mercato.

Un run fatto prima di questa modifica non ha queste colonne e va ripetuto.

### Output

Comando `python -m dl_model.calibration [<nome run>]`. Senza argomento usa il run più recente in `checkpoints/`. Legge i due CSV del run e scrive in `checkpoints/<nome run>_calibration/`:
- `summary.csv`, una riga per mercato e linea: mercato, linea, ambito (totale, squadra, 1X2), n, ECE, quota di partite convinte, e per i casi convinti over e under n, probabilità media e frequenza reale con intervallo;
- `bins.csv`, tutte le fasce di tutte le linee;
- un grafico di affidabilità (PNG) per famiglia: corner totali, corner per squadra, tiri in porta totali, tiri in porta per squadra, gol over/under, 1X2. Ogni grafico mostra la diagonale della calibrazione perfetta, una curva per linea e punti di dimensione proporzionale al numero di partite. Dove c'è, la curva del mercato è tratteggiata.

Stampa anche a terminale la tabella riassuntiva. Il modulo non importa torch: usa solo pandas, numpy, scipy e matplotlib, così da evitare il conflitto fra le due copie di OpenMP (numpy/MKL e torch) visto in questo ambiente.

### Approcci considerati

- **Modulo `dl_model/calibration.py` con test e comando a riga di comando (scelto):** è riproducibile su ogni run, testato, e gli output restano accanto al checkpoint.
- **Analisi nel notebook `prove.ipynb`:** è più veloce da scrivere, ma va rifatta a mano a ogni run e non è testata.
- **Calibrazione dentro il report di `train.py`:** lega l'analisi al training, che costa decine di minuti, e non permette di rianalizzare un run già fatto.

### Fuori scope

- Ricalibrare il modello (temperature scaling, regressione isotonica): prima si misura, poi si decide.
- La calibrazione per singolo campionato o stagione: si potrà aggiungere come filtro.
- Il backtest del ROI: è un passo successivo, che userà le quote grezze aggiunte qui.

## Struttura del codice

Package `dl_model/`, completamente autonomo (nessuna dipendenza da altri modelli):
- `dl_model/data.py`: preparazione dei dati — costruzione vocabolari per le feature categoriche (squadra, campionato), normalizzazione delle feature continue, `Dataset` PyTorch che restituisce feature categoriche/continue e i sei target (gol, tiri in porta, corner per lato: `goals_*`, `shots_on_target_*`, `corners_*`) più la maschera `has_stats`; `TARGET_LEAGUES` per lo scope di valutazione; `move_batch_to_device` per spostare un batch sul dispositivo del modello.
- `dl_model/model.py`: il modulo PyTorch con i due rami (CNN 1D, Transformer), la fusione, e le tre teste di output.
- `dl_model/losses.py`: `poisson_nll(lambda_pred, target) -> Tensor` e `negative_binomial_nll(mean_pred, dispersion_pred, target) -> Tensor`.
- `dl_model/evaluation.py`: `poisson_score_matrix`, `match_outcome_probs`, `implied_probabilities`, `log_loss` (utility di valutazione generiche, non specifiche a nessun modello) più `evaluate_goals`/`evaluate_counts`.
- `dl_model/calibration.py`: probabilità over/under e 1X2 dalle distribuzioni salvate nei CSV di un run, tabelle di affidabilità, ECE, analisi dei casi convinti, grafici; comando `python -m dl_model.calibration`.
- `dl_model/train.py`: orchestrazione — `set_seed`, normalizzazione sul burn-in (`build_model_and_vocabs(matches, normalization_matches=...)`), `save_checkpoint`, training iniziale su burn-in, loop di fine-tuning walk-forward, valutazione (log-loss gol vs mercato sui Big 5, NB-NLL tiri in porta/corner, colonne `shots_on_target_nll` e `corners_nll`), report finale.

## Non in scope per questa iterazione

- Falli e cartellini (task aggiuntivi, non coperti qui).
- Ricerca approfondita di iperparametri/architettura (dimensioni embedding, numero di layer, learning rate) — si parte da valori ragionevoli di default, senza una fase di tuning sistematico.
- Bilanciamento automatico dei pesi tra le tre loss.
- Caricamento del checkpoint per l'inferenza in produzione (rimandato all'integrazione nella Fase 3): qui il modello viene soltanto salvato.
- Ottimizzazione per GPU oltre allo spostamento sul dispositivo (mixed precision, caricamento dei dati vettorializzato invece che riga per riga).
- Confronto con altri modelli (baseline statistiche, gradient boosting): non più nello scope di questo progetto.
