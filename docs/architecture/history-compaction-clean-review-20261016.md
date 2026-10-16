# Revisione della compattazione e dei confini del frontend — 16 ottobre 2026

La revisione riguarda la compattazione esplicita di ExMekh-Lib e i percorsi
collegati di history, cache, file, output, lifecycle delle skill e frontend.
Sono stati riletti anche i confini dei provider e cercati accessi privati
fra classi nei sorgenti Python delle due repository.

## Responsabilità e interfacce

| Componente | Responsabilità |
| --- | --- |
| `HistoryCompactor` | Coordina snapshot, candidati, valutazione e ricostruzione; non applica il risultato al nodo. |
| `CompactionScorer` | Costruisce il catalogo e la richiesta sullo schema della history specifica, quindi valida tutti i voti. |
| `CompactionPresenter` | Produce i riassunti e la rappresentazione XML dei blocchi sostitutivi, conservando allegati e riferimenti. |
| `NodeRuntime` / `TurnControl` | Coordinano applicazione, esclusività, richieste accodate e confini sicuri della run. |
| `DeepSkillExecutionTracker` | Protegge e riallinea le proprie ancore senza cambiare identità degli scope. |
| `ToolOutputView` | Presenta metadati e stato degli output con un contratto comune per risultati visibili e discendenti. |
| `XmlHeader` / `XmlHeaderContent` | Analizzano una volta il framing; espongono campi decodificati, corpo esatto e riferimenti attraverso API pubbliche. |
| `OutputFileSession` / `NodeCache` / `OutputFileWorkspace` | Possiedono rispettivamente lettura/autorizzazione, scope del nodo e traduzione dei path. |
| `ChatMessage` / `ChatViewComponent` | Il modello confronta i dettagli di esecuzione; la vista decide se aggiornare i controlli. |

Il frontend non interpreta più gli elementi XML degli header della libreria.
Il lettore distingue metadati assenti da una lista esplicitamente vuota.
Le ricevute di annullamento senza header mantengono il loro parser separato,
perché appartengono a un protocollo diverso.

## Difetti riprodotti e corretti

- Un ID riutilizzato poteva introdurre nella compattazione i discendenti del
  risultato più recente, anche quando nella history era visibile uno snapshot
  precedente. I metadati della cache vengono usati soltanto se il contenuto
  corrisponde al risultato visibile.
- Una lista `used-tools` esplicitamente vuota veniva trattata come metadati
  mancanti e sostituita con i riferimenti in cache.
- Gli output discendenti conservavano `response_type` anziché il campo pubblico
  `response-type`. La presentazione passa ora dal renderer del record.
- Il frontend interpretava campi base64 come testo letterale e rimuoveva le
  righe vuote iniziali del corpo. La lettura condivisa elimina soltanto il
  separatore di framing e rifiuta metadati codificati non validi.
- La workspace autorizzava i file tramite una richiesta del limite di lettura,
  poi li apriva direttamente. Ora il proprietario verifica identità e versione
  prima, durante e dopo la lettura; la workspace usa il metodo della cache.
  Metadati file incompatibili con il manifest fanno fallire la compattazione
  senza scartare silenziosamente il riferimento o applicare una nuova history.
- Dopo una riconnessione, ogni aggiornamento del reasoning ricreava anche i
  controlli delle skill. Il test browser falliva perché il pulsante spariva
  continuamente dal DOM. Gli aggiornamenti che cambiano soltanto testo e
  reasoning della radice preservano ora gli elementi dei pannelli di esecuzione;
  cambiamenti ai risultati o ai token delle skill continuano ad aggiornarli.
- I pannelli reasoning, a differenza di tool e skill, venivano richiusi a ogni
  ridisegno. Gruppo e singoli blocchi usano ora lo stesso `DisclosureState`,
  conservando le scelte per identità anche quando vengono ricreati.

Sono presenti regressioni deterministiche per questi casi, per modifiche ai
file durante la lettura e per compattazione richiesta durante un ritorno nativo.
Il test browser originale passa senza aumentare timeout, forzare clic o ridurre
le sue asserzioni.

## Principi controllati e limiti

La revisione ha verificato responsabilità, API pubbliche, dipendenze,
nomenclatura, separazione fra dati e presentazione, assenza di duplicazioni,
gestione degli errori e preservazione del comportamento. La scansione degli
accessi privati rimasti trova copie della stessa classe, chiamate fra classi
base/derivate e API standard come `_replace`; non introduce accessi del
frontend al runtime interno della libreria.

Non sono stati aggiunti limiti ai riassunti o reasoning alle chiamate di
compattazione. La selezione del nodo resta obbligatoria in
`DeepAgent.compact_history(node)`.

Questo controllo non certifica automaticamente ogni modulo della repository.
Le classi di orchestrazione UI e il server C restano estesi: dividerli senza un
confine funzionale dimostrato produrrebbe una riscrittura più ampia di questa
revisione. I kernel e i backend GPU non sono stati modificati. I test del
workflow non misurano il riuso della KV cache e non certificano il prompt
effettivo ricostruito dal server dopo un ritorno nativo.

## Verifiche

Gli esiti, i conteggi e la provenienza della wheel sono registrati nel
[riepilogo della qualifica](../performance/history-compaction-clean-review-20261016.json).
