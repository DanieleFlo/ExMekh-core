# Revisione clean code T19–T22 e integrazione multimodale

Revisione del 22–23 ottobre 2026 delle modifiche native, dei launcher, del
packaging e dei confini fra componenti, provider, history e frontend. Comprende
i chiamanti esterni al diff iniziale; non certifica ogni file della repository
o tutti i principi clean code come proprietà assoluta.

## Responsabilità e comportamenti corretti

| Problema | Correzione |
| --- | --- |
| Il bridge numerico leggeva campi interni della cache e possedeva il suo snapshot | Budget e statistiche appartengono a `ds4_weight_cache`; il bridge gestisce soltanto lo scope del runtime e usa l'API pubblica |
| Encoder e graph decidevano direttamente la policy di preload | La cache decide se preriscaldare il prefisso richiesto in base alla propria capienza; i percorsi bounded acquisiscono i gruppi su richiesta |
| I tre encoder Gemma duplicavano conteggio e rilascio dei gruppi | Usano i metodi della cache; la pipeline legge il budget pubblico dei componenti senza invocare stime DiT con geometrie fittizie |
| Il budget pinned veniva restituito prima di liberare i buffer | La risorsa possiede il distruttore degli slot, eseguito prima di accreditare i byte; il test osserva il ledger durante la liberazione |
| Il costruttore dello staging ereditava un evento compute estraneo | Una prenotazione figlia prende in prestito il budget; la durata delle copie è controllata da `wait`, anche in recupero dopo un'allocazione fallita |
| La chiusura CUDA ignorava un errore di drain dello staging | `prepare_close` è una fase fallibile dell'interfaccia interna del backend: un errore conserva contesto, callback, storage e contabilizzazione per il retry |
| Un errore nel packager lasciava una directory finale incompleta | Preparazione, copia sorgenti, documentazione e manifest hanno funzioni distinte; staging temporaneo e rinomina pubblicano soltanto un pacchetto completo |

I test della cache leggono lo snapshot pubblico. Il budget mantiene il proprio
layout; il backend pesi a cinque campi e il piano Gemma v1 conservano il loro
contratto. Gli adattatori continuano a possedere le equazioni dei modelli, la
cache possiede la residenza, il registro possiede la contabilizzazione e il
backend possiede stream/eventi CUDA. Il server non interpreta i tensori.

Nel frontend sono verificati anche i due ordini di composizione dei wrapper
provider: restituiscono lo stesso client nativo e una cancellazione impedisce
una nuova acquisizione. Le anteprime conservano la compatibilità con i payload
precedenti senza decodificare i pixel nel collector. Non sono introdotte nuove
gerarchie o sostituzioni della UI.

## Verifiche

Risultati e hash della nuova esecuzione sono registrati nel
[manifest della revisione](native-suite-refactor-review-20261022.json).
Log, diagnostiche negative e artefatti locali sono sotto
`performance-results/native-suite-refactor-review-20261022/`.

I test negativi usano lo staging del pacchetto precedente e il core numerico
del commit di partenza: falliscono sulle nuove asserzioni di lifetime. I moduli
corretti sono verificati anche con AddressSanitizer e UndefinedBehaviorSanitizer.
Il recupero dopo una copia fallita, il drain ripetibile e cento alternanze
verificano contenuti e contabilità, oltre ai codici di ritorno.

La suite completa frontend include i browser offline; l'accettazione live
riusa `make test-suite-agent-live` con immagini, quattro modalità embedding,
ranking della libreria, ritorno al LLM e browser reale. Ogni processo modello
viene eseguito in sequenza sulla RTX 3090. Il server verificato usa Qwen27,
MTP, proiettore visivo e contesto 26000.

| Gate | Esito |
| --- | --- |
| Frontend completo, inclusi browser offline | 1133 PASS, 45 live esclusi, copertura linee/branch 100% |
| Libreria e consumer nativi live; browser reale dell’agente | 11 PASS; 1 PASS, 156,84 secondi |
| Cinque PNG deterministici rispetto a prima della revisione | SHA256 identici, incluso paesaggio 1024×768 |
| Contratti, API, processor, launcher e gate release | 63 PASS |
| Bundle e cache nativa | 12 PASS |
| Numerica CPU / CUDA | 22 PASS + 3 riservati a CUDA / 25 PASS |
| Risorse, executor, runtime, stato sessioni e server legacy | PASS |
| Lifecycle CUDA MTP/visione, tre ripristini entro capienza; executor tre modalità | PASS, continuazioni bit-exact |
| Lifetime negativo e sanitizer | Le due regressioni falliscono sul codice precedente; ASan/UBSan PASS sul codice corretto |
| Pacchetto reale aggiornato | 287 file verificati, `development` |

Il primo passaggio frontend aveva tutti i test verdi ma copertura 99,93%:
le nuove regressioni sui wrapper nativi e sugli allegati precedenti chiudono
quelle lacune senza abbassare il gate di copertura.

I report T19–T22 e dell'accettazione precedente conservano hash e scope storici.
Questa revisione non attribuisce al nuovo binario le misure di stress o qualità
raccolte su binari precedenti. Restano aperti gli studi di ottimizzazione e i
gate qualità T11 già documentati. Metal, ROCm, SSD streaming di altri modelli
e inferenza distribuita non sono qualificati da questa esecuzione locale.
