# Revisione delle responsabilità dei draft OLD — 12 ottobre 2026

La revisione conserva il default XML v2 e il verificatore CUDA Qwen27 da 32/64
token. Corregge la selezione del checkpoint nelle proposte di snapshot e
l'attribuzione delle metriche, e centralizza regole già condivise dai tool.
Le misure della precedente implementazione restano nel
[report dei draft OLD](../performance/old-text-snapshot-chunks-20261012.md);
non sono misure del nuovo binario.

## Responsabilità e interfacce

| Componente | Responsabilità |
| --- | --- |
| `ds4_edit_document.h` | Dialetto XML, checkpoint, selezioni, snapshot e continuazioni XML; usa le API del matcher senza leggere gli indici interni. |
| `ds4_interactive_edit.h` | Adattamento tra XML e framing nativo; richiede il restart attraverso l'API del documento e non modifica lo stato durante le proposte. |
| Protocollo interattivo | Lifecycle, pause, autorizzazioni ricevute, KV e ricevute; invoca i callback del comportamento con stato opaco. |
| Scheduler server | Tokenizzazione delle proposte, limiti, policy adattiva e pubblicazione dei soli token accettati; usa l'API pubblica della sessione. |
| Motore | Logits target, verifica, rollback/replay, recurrent/attention KV, catch-up MTP e RNG. |
| Client | Filesystem/workroot, inspect/read, hash, alias, intervalli, conflitti, hook e commit dei file; resta proprietario di preflight e scritture. |

La policy dei batch ha operazioni distinte per entrare in una fase, leggere
la capacità e osservare l'esito. La capacità è una query senza mutazioni.
Le dimensioni nominali restano 32 e 64; limiti di contesto, budget e snapshot
possono troncarle. Le code sotto 32 token usano il percorso ordinario/MTP.
Una proposta senza accettazioni aggiuntive viene sospesa fino alla selezione
successiva; un'accettazione parziale bassa introduce una pausa di 32 token.

## Correzioni e uniformità

- Una proposta su un prefisso precedente al checkpoint accettato confondeva
  la selezione completa con un delta. Ora riceve il restart effettivo della
  scansione. Il test riproduceva il difetto prima della correzione e verifica
  che la proposta precedente non modifichi il checkpoint corrente.
- Le righe e i fallback dei draft da snapshot venivano conteggiati come MTP.
  Il sampler conserva metriche comuni del masking, ma attribuisce i contatori
  di verifica al percorso che li ha prodotti. Il log OLD espone anche righe
  campionate e fallback.
- DSML, GLM, Qwen e risultati dei tool applicavano la stessa regola di escaping
  con funzioni duplicate e un nome specifico di GLM usato anche da Qwen.
  `append_tool_body_text` è ora l'unico writer del framing nativo. XML ed escape
  nativi restano due livelli distinti.
- CLI e ramo batching del server interpretavano `DS4_MTP_SPEC_DISABLE=0`
  diversamente dal resto del server. Tutti usano il parser booleano condiviso.
- Entrambi i verificatori trattano un callback che restituisce un token fuori
  dal vocabolario come una verifica declinata: ripristinano RNG e conservano
  soltanto il primo token target, senza restituire un token invalido al caller.

I file principali del motore e del server restano grandi unità C. Questa
revisione migliora i confini coinvolti e i comportamenti verificabili; non
certifica che ogni parte della repository soddisfi ogni principio di clean code.
La suddivisione completa di queste unità richiede un intervento distinto con
qualifica dei backend. Non sono state modificate le classi client già separate
tra documento, selezione, approvazione e filesystem.

## Verifiche e limiti

Le evidenze finali, gli hash e gli esiti delle prove sono nel
[riepilogo JSON](../performance/snapshot-draft-clean-review-20261012.json).
I log e gli scambi live sono conservati localmente in
`performance-results/snapshot-clean-review-20261012/`.
Il test rosso iniziale è conservato in `/tmp/ds4-snapshot-review-red.log`.

Il masking governa il protocollo e la compatibilità OLD con lo snapshot.
Non garantisce che il modello scriva il contenuto NEW richiesto dall'utente.
Il test HTML che richiede 14 paragrafi aveva già prodotto un solo paragrafo
anche sulla baseline; l'esito viene conservato come errore di intento.
La qualifica live riguarda CUDA/Qwen27. Metal, SSD streaming e distribuito
non sono qualificati da queste prove; il nuovo verificatore mantiene capacità
zero sui percorsi non supportati.
