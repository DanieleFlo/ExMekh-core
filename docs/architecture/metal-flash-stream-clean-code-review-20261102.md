# Riesame Clean Code dello streaming Flash, 2 novembre 2026

Base `1544bb91eafaefefe32862ab408c412f0beed354` più le modifiche Flash
non committate. Riesaminati cache Metal, coda C condivisa, letture CUDA/PLE,
trasporto numerico, probe e fixture, piano e artefatti prestazionali.

## Responsabilità individuate prima del refactoring

- La cache Metal deve possedere binding, layout compatto, pin, budget e fence
  GPU. Non deve implementare anche descriptor, task I/O, contatori e callback.
- La coda `ds4_read_queue.h` possiede la concorrenza. I consumer Metal, CUDA,
  PLE e `ds4_transfer` usano già i suoi metodi, senza leggere i suoi campi.
  `ds4_transfer` possiede staging e callback DMA: non è un lettore di file e
  non è il posto per aggiungere le letture dirette dei pesi Metal.
- Il lettore host deve possedere descriptor duplicato, coda lazy, letture
  complete, errori e contatori. Il caller deve ricevere un risultato senza
  conoscere task, worker o mutex. I worker non devono stampare diagnostica.
- Il vecchio helper riporta `errno` anche su EOF (`pread == 0`), dove può
  essere un valore rimasto da operazioni precedenti. La diagnosi deve essere
  stabile: EOF prematuro è EIO, errori concorrenti pubblicati nell'ordine
  richiesto dopo aver atteso tutti i job del batch.
- Il caso di errore nel main della fixture Metal mescola setup, troncamento,
  sentinelle, cache e cleanup. Estrarlo in una funzione dedicata chiarisce il
  contratto senza ridurre gli assert.

Il refactoring conserva quattro worker, profondità 16, chunk I/O di 8 MiB,
layout, ordine GPU e frontiere numeriche. Non è un nuovo tuning del kernel.
Rilette le fonti fissate della campagna: Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`
`metal/src/engine/stream.{cpp,h}` per pubblicazione e lifetime; Transformers
`ca8c871ede62b521a75a2ddfc5e1968ac9c8e097` `modeling_qwen4_exp.py` e ggml
`dde879bc4f954b809892fd0d660515f42536394a` `ggml-metal-ops.cpp` per i confini
che restano invariati. Manifest completo nel report prestazionale Flash.
Non è emersa una nuova differenza numerica o una regressione misurata.

## Modifiche e verifica

Separato `ds4_file_reader.h`: un proprietario C privato, con API di apertura,
lettura esatta/indicizzata, conteggio e chiusura. La cache Metal non accede a
descriptor, task o campi del reader. La coda condivisa resta invariata.
Un risultato d'errore esplicito sostituisce le stampe dai worker; solo il
caller aggiorna contatori e diagnostica, conservando `errno` coerente.
L'owner viene acquisito prima di sostituire quello vivo e chiuso prima del
rilascio delle destinazioni. La fixture dei guasti ha ora un proprio metodo.

Eseguiti test host del lettore con input identici, indici duplicati/non ordinati,
limiti, EOF parziale e counter, proprietà del descriptor, fallback dopo
fallimento del pool e riapertura. Poi fixture Metal resident/streamed,
lifecycle, gate Flash reale indipendente dal benchmark e controllo breve
di performance contro il binario pre-refactoring conservato.

Il primo confronto a quattro token (warm-up e due coppie) indica steady decode
1,07/1,08 → 1,01/1,03 token/s. È un segnale da indagare, non una regressione
causale già dimostrata. Prima di modificare il trasporto sono stati riletti
`Strata stream.cpp:read_part/io_main`, la coda C, l'helper originale e il nuovo
lettore: stesso ordine di richieste, quattro worker, profondità 16, chunk
8 MiB e destinazioni; nuovi solo ownership, validazione anticipata e risultato.
Transformers/ggml restano invariati e i logits/stati coincidono. Si amplia il
confronto a cinque coppie con otto token per distinguere effetto riproducibile
e variabilità, senza cambiare kernel o dispatch sulla base del solo segnale.

| Verifica finale | Esito |
| --- | --- |
| Build CLI/server/bench/eval/suite e fixture pertinenti | PASS; soltanto warning già presenti nel core incluso |
| Lettore host, coda condivisa, AddressSanitizer/UndefinedBehaviorSanitizer | PASS, inclusi fallimento del primo worker e fallimento dopo due avvii |
| Qualificatore, launcher e fondamenta: 49 casi host | PASS; il solo bind loopback negato dal sandbox è stato ripetuto con successo fuori sandbox |
| Primitive Flash streamed/resident, grafo e lifecycle | PASS |
| Primitive/grafo Qwen27 e lifecycle ordinario/streamed, senza GGUF completo | PASS |
| Flash reale target-only e MTP | PASS: 18 vettori full-vocabulary e i due stati prefill bit-exact col codice precedente; MTP delta massimo zero |

Il primo build ha trovato una variabile rimasta nel main durante l'estrazione
della fixture: corretta rendendola locale. Un giro di gate era stato avviato
prima di conoscere l'esito della build; è escluso dai risultati, anche dove
stampava PASS. Tutti i dieci gate conteggiati sopra sono eseguiti nuovamente
dopo la build riuscita. Il report conserva entrambi i tentativi e i rispettivi
hash, senza far passare vecchi binari per una verifica del nuovo codice.

Il primo benchmark viene rifiutato all'ammissione: 1,89 GiB richiesti e
1,11 GiB disponibili dopo riserva. Un successivo controllo rileva 6,37 GiB
host disponibili, permettendo il profilo originale 8192/64/cache 1 GiB.
Nessun guard o parametro del profilo è stato aggirato per ottenere la misura.

La conferma usa un warm-up scartato e cinque coppie alternate, 64 token input
e otto teacher-forced di decode. Mediane prima/dopo: prefill 17,68 → 17,48
token/s (−1,1%), decode 0,97 → 0,99 (+2,1%), steady 1,02 → 1,02 (0%).
Il calo iniziale non si riproduce in modo stabile: CV del prefill candidate
circa 8%, decode 5,9–14,3% includendo lo steady nelle due versioni. Verdetto prestazionale
**NEED_MORE_DATA**: non si promuove un nuovo speedup e non si certifica
una stretta non-regressione con questi campioni. Si conservano tutti i
campioni, compreso l'ultimo più lento, e le misure storiche della campagna
Flash restano riferite al codice allora misurato.

Il 2 novembre 2026 l'utente conferma il miglioramento nell'uso, riferisce
attività concorrenti sul PC durante queste misure e autorizza l'attivazione.
Il percorso resta abilitato per impostazione predefinita nello streaming Flash
Metal; nessun flag o ulteriore intervento su kernel/dispatch è necessario.
La build di produzione risulta aggiornata e i test host lettore/coda passano.
L'accettazione operativa è distinta dal verdetto statistico sopra, che conserva
tutti i campioni originali.

Risultati, comandi, hash e campioni nel
[report verificabile](../performance/metal-inference-20261102/flash/clean-code-review.json).
Artefatti completi in `build/metal-perf-flash-20261102/clean-review/`.
La serie di commit separa ottimizzazione originaria, refactoring/regressioni
e piano/evidenze prestazionali.

## Limiti

Il riesame può verificare e migliorare questi confini, non certificare ogni
principio Clean Code per l'intero repository. Core C/server e runtime Metal
conservano responsabilità legacy e stato globale; una loro migrazione richiede
un progetto distinto con gate per modello/backend. Non si converte il codice
in C++ o in classi Objective-C per attività host che appartengono al C.
