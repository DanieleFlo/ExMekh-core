# Riesame Clean Code delle fondamenta Metal

29 ottobre 2026. Riesame delle modifiche non ancora committate sul riferimento
`8e8ac45a7b463f7d4d72d859234f794b815019ef`, compresi i collegamenti con la
suite numerica, i launcher e le risorse prompt del frontend. Non cambia le
formule dei kernel o le soglie numeriche del [piano Metal](../roadmaps/metal-apple-silicon-support-20261028.md).

## Responsabilità e interfacce

| Componente | Responsabilità dopo il riesame |
| --- | --- |
| `ds4.c` | Stima di modello/sessioni e orchestrazione; chiama il backend per aggiornare disponibilità e ownership |
| `ds4_metal_device.h` | Lettura macOS/MTLDevice; conserva i valori misurati e la pressione anche quando l'ammissione è negata |
| `ds4_metal_memory.h` | Policy pura, riserva e applicazione della stessa contabilità host/device ai due owner Metal; nessun import SDK |
| Backend GPU legacy | Ownership di buffer, ritiri dopo fence e riconciliazione delle proprie risorse; il motore non ricostruisce i ledger |
| `DS4MetalNumeric` | Ownership di indirizzi, buffer, command queue e operazioni numeriche; usa il campionamento e la policy condivisi |
| `CommandRunner` | Ambiente dei processi, timeout, terminazione del gruppo e raccolta delle prove di ciascun comando |
| `QualificationReport` | Persistenza atomica, risultati dei gate/validazioni e verdetto finale attraverso metodi pubblici |
| Fixture Qwen35 | Ownership esplicita di mmap, descrittori e pesi sintetici; verifiche separate per chunk, replay, guasti e cleanup |
| Oracle FP16 Qwen | Un solo decoder scalare di storage, riusato dai probe Metal/CUDA e verificato su tutti i 65536 pattern |

Il test lifecycle continua a costruire descrittori interni sintetici perché
verifica stato e ownership senza un GGUF. Per il path dei checkpoint e il
rilascio dei pesi usa ora `ds4_session_checkpoint_path()` e
`ds4_engine_weights_release()`, senza accedere al layout privato o ricostruirne
manualmente il comportamento.

## Correzioni di coerenza

- Il campione della RAM disponibile è distinto dalla capacità ammessa: una
  pressione elevata azzera l'ammissione, conservando il dato diagnostico.
  Un errore di campionamento invalida la disponibilità precedente. Cap espliciti,
  riserve e risorse vive sopravvivono alla riconciliazione.
- Backend LLM e suite applicano la stessa policy, evitando due implementazioni
  che potrebbero divergere su pressione, riserva o memoria già allocata.
- Il qualificatore distingue comando mancante, errore e timeout, conserva il
  fallimento nel report e termina anche i discendenti del processo. Validazioni
  di modello o simboli fallite impediscono un verdetto PASS.
- L'inventario valida conteggi, shape, blocchi per riga, offset e nomi duplicati.
  Distingue convoluzione e proiezione in base all'operazione; una matrice F32
  non viene etichettata automaticamente come normalizzazione.
- Il replay dichiarato bit-exact confronta i byte, oltre alla finitezza dei
  valori. Il guasto di allocazione viene provocato dopo la creazione degli
  alias scratch, verificando il percorso di cleanup descritto dal test.
- Il decoder FP16 degli oracle distingueva soltanto zero e valori normali,
  azzerando i subnormali. La versione condivisa conserva quei valori e il
  segno dello zero; un riferimento aritmetico verifica tutti i pattern finiti
  e la classificazione di infiniti/NaN. I kernel e le tolleranze non cambiano.
- I test dei launcher hanno un ambiente isolato condiviso e nomi coerenti con
  entrambi i modelli. I profili espliciti non dipendono dal nome del binario
  per leggere le variabili stampate.
- La guida frontend distingue la preparazione residente 4096 dalla priorità
  streamed 8192. Il prompt tool usato per il conteggio coincide con quello del
  server e ha un test di contratto per prefisso e suffisso.

## Verifica e limiti

Il [registro del riesame](../performance/metal-apple-silicon-20261029/foundation-cleanup/README.md)
conserva comandi, esiti, log e hash dei sorgenti successivi al refactoring.
I report precedenti descrivono i loro snapshot originali e non vengono riscritti
per far coincidere retroattivamente gli hash con i nuovi file.

Il riesame verifica responsabilità, leggibilità, ownership e contratti delle
aree modificate e dei componenti collegati. Non è una certificazione universale
di tutti i principi Clean Code sull'intero repository. Non qualifica inferenza
reale Qwen27/Flash, MTP/vision, contesto effettivo 8k o altri acceleratori;
T03 e i gate applicativi della roadmap mantengono i limiti già dichiarati.
