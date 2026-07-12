# Fase 1 — Pulizia Dataset & Feature Engineering

## Contesto

Il progetto League Prediction si sviluppa in tre fasi indipendenti:

1. **Pulizia dataset e feature engineering** (questa spec)
2. Modelli predittivi: deep learning + Dixon-Coles (e possibile confronto con XGBoost)
3. Tool di value betting: dato un set di quote fornite dall'utente, suggerisce le giocate di valore secondo le predizioni dei modelli

Questa spec copre solo la Fase 1.

## Dati sorgente

- `Matches.csv`: 230.557 partite, 48 colonne, 38 campionati/divisioni, dal 2000-07-28 al 2025-06-01. Fonte: [Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025).
- `EloRatings.csv`: storico Elo per squadra/paese, rilevato due volte al mese, copre lo stesso intervallo temporale.

## Obiettivo

Produrre un unico CSV pulito, pronto per addestrare sia il modello Dixon-Coles che il modello di deep learning nella Fase 2, senza necessità di ulteriore pulizia a valle.

## Recupero Elo mancanti

`HomeElo`/`AwayElo` mancano nel 38.6% delle righe di `Matches.csv`. Per ogni riga con Elo mancante, si esegue un merge con `EloRatings.csv`: si prende l'ultima rilevazione Elo disponibile per quella squadra (match su `club` + `country`) con data antecedente o uguale a `MatchDate`.

## Righe scartate

- Righe senza `OddHome`/`OddDraw`/`OddAway` (1.3% del totale). Senza quota storica quella riga non è utilizzabile né per il training né per il futuro backtest della strategia di value betting (le quote storiche fungono da probabilità implicita di mercato, un segnale predittivo di per sé, oltre che da riferimento per simulare la strategia sul passato).

## Colonne scartate

- `MatchTime`: 57% mancante, basso valore predittivo atteso.
- `MaxHome`, `MaxDraw`, `MaxAway`, `MaxOver25`, `MaxUnder25`: quote massime tra ~17 bookmaker, ridondanti rispetto alle quote Bet365 (`OddHome/Draw/Away`, `Over25/Under25`) già presenti.

## Colonne mantenute con flag di disponibilità

Per le colonne seguenti, il valore mancante viene imputato a 0 (valore neutro) e viene aggiunta una colonna booleana che indica se il dato era originariamente disponibile. Questo perché l'assenza non è casuale (dipende da campionato/anno) e va distinta esplicitamente da uno zero reale:

| Colonne | Flag |
|---|---|
| `HomeShots`, `AwayShots`, `HomeTarget`, `AwayTarget`, `HomeFouls`, `AwayFouls`, `HomeCorners`, `AwayCorners`, `HomeYellow`, `AwayYellow`, `HomeRed`, `AwayRed`, `HTHome`, `HTAway`, `HTResult` | `has_stats` |
| `HandiSize`, `HandiHome`, `HandiAway` | `has_handicap` |
| `Over25`, `Under25` | `has_ou` |
| `C_LTH`, `C_LTA`, `C_VHD`, `C_VAD`, `C_HTB`, `C_PHB` | `has_cluster_data` |

Nota su `C_*`: rappresentano probabilità di appartenenza a cluster di pattern di partita (Low Tempo Home/Away-oriented, Visibly Home/Away Dominated, High Tempo Balanced, Highly Physical Balanced — vedi README del dataset sorgente). Sono strutturalmente assenti per 15 campionati su 38 (non missing casuale), da qui la necessità del flag.

## Feature derivate (v1)

- `EloDiff = HomeElo - AwayElo`: sintetizza la forza relativa tra le due squadre.
- Media mobile gol fatti/subiti nelle ultime 5 partite per squadra: calcolata cronologicamente sullo storico completo di ciascuna squadra (indipendentemente da casa/trasferta), usando `FTHome`/`FTAway` delle partite precedenti alla partita corrente. Per le prime partite di una squadra in cui la finestra di 5 non è ancora piena, si usa una finestra parziale (minimo 1 partita) e si aggiunge un flag `has_rolling_stats` per segnalare quando la finestra non è completa.

Altre feature lagged/derivate (es. rolling su tiri/corner, scontri diretti, giorni di riposo tra partite) sono rimandate a un'iterazione successiva, dopo aver visto il codice e i risultati preliminari di questa v1, e comunque prima di iniziare la Fase 2 (modelli).

## Split train/test

- Viene aggiunta una colonna `Season` derivata da `MatchDate` (es. `"2023-2024"`, con confine di stagione a luglio: mesi ≥ luglio appartengono alla stagione che inizia in quell'anno).
- `Season` è usata **solo** come chiave per lo split train/test nella Fase 2; non viene inclusa tra le feature del modello.
- `MatchDate` grezza è mantenuta nell'output come metadato (utile per riferimento/join) ma esclusa dalle feature effettive del modello, per evitare data leakage.

## Copertura campionati

Tutti i 38 campionati vengono mantenuti nel dataset pulito, anche quelli che non saranno il target diretto delle predizioni: l'idea è che i modelli possano imparare relazioni generali utili (es. dinamiche Elo, pattern di forma) anche da campionati non target.

## Output

Un unico file CSV, ordinato per `MatchDate`, con:

- Metadati: `Division`, `MatchDate`, `Season`, `HomeTeam`, `AwayTeam`
- Target: `FTHome`, `FTAway`, `FTResult`
- Feature esistenti pulite (Elo, quote, form, handicap, over/under, statistiche partita, cluster)
- Flag di disponibilità: `has_stats`, `has_handicap`, `has_ou`, `has_cluster_data`, `has_rolling_stats`
- Feature derivate v1: `EloDiff`, medie mobili gol fatti/subiti (ultime 5 partite)

## Non in scope per questa fase

- Feature lagged aggiuntive oltre a quelle elencate in v1 (rimandate a iterazione successiva).
- Qualsiasi logica di modeling (Dixon-Coles, deep learning, XGBoost).
- Logica di value betting/quote fornite dall'utente.
