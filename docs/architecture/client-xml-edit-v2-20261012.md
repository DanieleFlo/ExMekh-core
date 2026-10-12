# Client XML edit v2 — 12 ottobre 2026

ExMekh-Lib usa `edit-file(edits)` e negozia soltanto v2. La wheel versionata
e installata nel frontend contiene il client aggiornato. Non serve cambiare
configurazione per attivarlo e non esiste un fallback al contratto a marcatori.

## Responsabilità

| Componente | Responsabilità |
| --- | --- |
| Server | Grammatica/masking, frontiere e chiusure XML, sampling/MTP, exchange, checkpoint e KV |
| `EditDocument` | Parser indipendente del documento completo, entità/CDATA, corpi esatti e chiusura per elemento |
| `InteractiveContract` | Versione, scope, ID, ordine degli edit, path congelati e validazione wire senza coercizione |
| `FileInteractiveCalls` | Evidenze visibili, autorizzazione per call ID/indice, snapshot iniziale degli alias, rifiuto e recupero |
| Selezioni client | Unicità/boundary dei letterali, righe inclusive, limiti numerici e conflitti |
| Servizio filesystem client | Workroot, encoding/BOM, permessi, identità/hash, controlli dopo hook, lock e commit |

Il documento dopo gli hook deve contenere gli stessi path nello stesso ordine
degli scambi autorizzati. Tutti i risultati vengono pianificati e tutti i target
verificati prima della prima scrittura. Ogni file canonico è pubblicato una sola
volta. Rifiutare il secondo elemento elimina anche il primo della stessa call;
le altre chiamate già complete restano eseguibili. Steering, cancellazione e
cambi di scope impediscono il dispatch o la scrittura.

I vecchi marcatori sono testo ordinario. I comandi XML si distinguono dal testo
escapato o CDATA e non sono rimossi dai file fisici. Il parser preserva CR/LF e
spazi, usa la chiusura finale di ogni edit e accetta i precedenti bare close come
testo. Il client valida la semantica degli intervalli che il server verifica
solo sintatticamente.

## Verifiche

La suite completa ExMekh-Lib sul client aggiornato passa: **2.588 PASS,
53 SKIP live opt-in**, copertura **100%**, Ruff pulito e Pyright senza errori/avvisi. Il frontend,
con la wheel XML installata, passa **938 test**, inclusi Playwright, con
**40 SKIP live** e copertura **100%**. La verifica finale della wheel e le
prove live sono riportate nel [report JSON](../performance/client-xml-edit-v2-20261012.json).

Prima prova target-only: **7 PASS, 7 FAIL**, 14 casi, nessuno skip. I fallimenti
riguardano intento/routing del modello: riscrittura al posto di delete, assenza
del routing alla skill, newline mancanti e contenuto letterale diverso. Alias,
attacco di 30 secondi, rifiuto del secondo elemento in entrambi i modi e
lettura testuale passano. Gli assert sui byte e le prime evidenze restano
conservati nel [primo tentativo](../performance/client-xml-edit-v2-20261012-a1-wire.json).
Le istruzioni dei test sono state chiarite e il sampling reso
deterministico; non si riparano gli argomenti restituiti.

Il [secondo tentativo](../performance/client-xml-edit-v2-20261012-a2-wire.json) ha
registrato **11 PASS, 3 FAIL**: delete, file vuoto e il caso dei vecchi marcatori.
Quest'ultimo caso live è stato rimosso su indicazione dell'utente; i marcatori
sono abbandonati anche nei prompt. Resta una regressione deterministica che
verifica l'assenza della loro precedente semantica.

La [prova finale target-only](../performance/client-xml-edit-v2-20261012-a3-wire.json)
ha eseguito tutti i **13 casi attuali**: **11 PASS, 2 FAIL, nessuno skip**.
Passano modifica, creazione, inserimento, intervalli, contenuto XML letterale,
più file, alias, attacco di 30 secondi, rifiuto del secondo elemento in entrambi
i modi e lettura testuale. Delete fallisce dopo 20 turni: il modello genera
un tag delete protetto in CDATA come testo e poi ripete il tentativo. Il caso
file vuoto genera riscritture non vuote. Questi XML validi vengono applicati
letteralmente dal client; i fallimenti semantici restano nel report.

La [verifica MTP](../performance/client-xml-edit-v2-20261012-mtp-wire.json) passa
**4 casi su 4**: creazione, alias e rifiuto del secondo elemento in entrambe
le modalità streaming. La telemetria registra **15 cicli e 5 token speculativi
accettati**. I server di test sono stati fermati, un modello alla volta.
Il client è implementato e installato per default; i due fallimenti di intento
nel run target-only restano aperti e non qualificano delete/svuotamento del modello.

## Limiti

Il masking garantisce vincoli del documento, non l'intento dell'utente:
un contenuto generato valido può essere semanticamente sbagliato. In particolare,
CDATA protegge un tag come testo: un `<delete/>` dentro CDATA non deve cancellare
un file. I controlli su newline esatti sono casi di fedeltà ai byte richiesti,
non una policy generale che limita il contenuto liberamente generato. Questa
migrazione non promette rollback transazionale tra file distinti. Un errore
operativo dopo un primo commit restituisce le voci `files` già applicate.
Scritture esterne non cooperative possono ancora correre nell'intervallo
finale fra controllo e rename/unlink. Il commit nativo Windows resta
unsupported; questa qualifica usa Linux/WSL, Qwen27 CUDA e contesto 22593.

Queste prove non riqualificano tutti i backend, modelli o percorsi vision.
Lo [storico server](clean-code-review-20261012.md) e le precedenti anomalie
mantengono evidenze e fallimenti originali.

Una possibile revisione futura distingue operazioni XML esterne, per esempio
`<delete path="..."/>` e `<write path="...">contenuto</write>`, nello stesso tool.
Separare il comando dal corpo potrebbe rendere la scelta del modello più semplice;
questa migrazione mantiene il contratto concordato e non introduce altri vincoli.
