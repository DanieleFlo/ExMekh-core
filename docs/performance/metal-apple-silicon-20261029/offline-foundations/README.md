# Task Metal eseguibili prima del download Flash

29 ottobre 2026, baseline `8e8ac45a7b463f7d4d72d859234f794b815019ef`,
Mac17,9/M5 Pro, 24 GiB, macOS 26.6.2, clang 21.0.0, SDK 26.5.
[verification.json](verification.json) fissa comandi, hash sorgenti/output,
ambiente, tempi, campioni RSS e memoria prima/dopo. Nessun GGUF è stato
aperto e nessun kernel numerico o percorso di produzione è stato modificato.

## Task verificati

| Area del piano | Evidenza aggiunta | Esito |
| --- | --- | --- |
| T00–T01: build Metal | `make -j2 all ds4-server-suite`, link del nuovo test | PASS; gli eseguibili ordinari erano aggiornati |
| T01: controllo CPU | Copia separata dei sorgenti C/header/include e `make -j2 cpu`, quattro eseguibili arm64 | PASS; nessun oggetto Metal riusato e nessuna inferenza CPU |
| T01: launcher | Dieci test esistenti, più dry-run di launcher comune e dei due wrapper agent | PASS; scelta automatica Metal, 512/64, target-only, suite esplicita; profili CUDA conservati |
| T02: risorse | Runtime, registro risorse, executor e factory suite: quattro gruppi host | PASS; ammissione, alias, fence, cancellazione, fault e ownership |
| T02/T07: lifecycle Metal | Shape complete Qwen27 e Flash, buffer reali, contesto 16 e prefill 2 | PASS; stato sintetico e nessun forward del modello |

La build CPU vive sotto
`build/metal-qualification/offline-foundations-20261029/cpu/`: non sostituisce
gli eseguibili Metal della directory principale. `otool -L` del suo `ds4`
mostra soltanto libSystem, senza Metal o runtime CUDA. Restano due warning
preesistenti di parametri inutilizzati nelle funzioni Flash incluse dal test.

## Lifecycle e recovery

`tests/test_qwen_metal_lifecycle.c` sostituisce il solo test Flash precedente.
Le due famiglie condividono la sequenza di verifica e conservano inizializzazione
e stato distinti. Il target storico `make test-flash-metal-lifecycle` resta
disponibile. Il target combinato esegue le famiglie in sequenza anche con `-j`.

- KV, convoluzione e stato ricorrente hanno valori distinti per layer/campo;
  Flash include storia PLE e frontiere di posizione. Sono salvati anche token,
  logits e metadata opachi del chiamante. Il caso Qwen27 è target-only.
- L'hash dell'intero payload deve coincidere dopo restore nello stesso oggetto,
  in una nuova sessione e dopo park/rilascio pesi/reload/resume.
- Un budget che consente soltanto 4096 byte aggiuntivi forza un fallimento
  durante la costruzione del candidato. Registro e sessione viva devono
  restare identici, senza risorse o riserve residue.
- Checkpoint corrotto, troncato prima della fine dell'header e mancante
  restituiscono rispettivamente INVALID, INVALID e RESTORE. Payload, metadata
  del chiamante e contatori restano intatti. Dopo i guasti, park/resume passa.
- Anche la sessione vuota attraversa park/resume. A fine prova tutti i ledger,
  root, view e reservation tornano alla baseline, comprese le pagine residenti
  del mapping di fixture e i byte checkpoint.

Ogni processo mantiene il cap di **1 GiB sullo storage device registrato** e
la riserva host condivisa di **2 GiB**. Driver, compilazione shader e altre
allocazioni host non sono storage device registrato: RSS e cap sono misure
distinte. Il mapping dei pesi è una fixture temporanea di soli 4096 byte;
la prova di reload non legge i pesi di un modello reale.

| Caso, esecuzione finale | Tempo con apertura | Massimo RSS campionato |
| --- | ---: | ---: |
| Qwen27 target-only | 7,12 s | 0,79 GiB |
| Flash | 4,58 s | 0,96 GiB |

Il working set raccomandato è 19.069.665.280 byte, mentre la disponibilità
host misurata prima dei due casi è rispettivamente 6.211.141.632 e
6.416.891.904 byte. `currentAllocatedSize` iniziale vale 1.769.472 byte,
`maxBufferLength` 14.302.248.960 byte: il report conserva queste grandezze
separatamente. Swap invariato prima/dopo entrambi i casi finali; campioni di
`memory_pressure -Q` 47%/47% e 46%/45%. Sono campioni ai confini della prova,
non un monitoraggio continuo o un benchmark. Il JSON conserva anche il primo
run, senza sostituirne misure o output. `make docs-check`: dieci test PASS.

```sh
make -j2 all ds4-server-suite tests/test_qwen_metal_lifecycle
make test-runtime test-resources test-executor test-suite-accelerator
python3 -m unittest tests.test_flash_launcher -v
make test-qwen-metal-lifecycle
```

## Limiti della qualifica

Il test usa le API di lifecycle di produzione su storage Metal reale, ma
stato e descrittori sono sintetici. Non dimostra logits corretti, equivalenza
dopo un forward, MTP/vision Qwen27, restore dopo restart o alternanza reale
con image/embedding. F01–F12 mantengono i gate reali aperti; F12 guadagna
copertura isolata di recovery. Il download Flash dell'utente può proseguire
indipendentemente.

Il gate BF16 interno Gemma resta FAIL secondo le evidenze precedenti. HTTP,
frontend ExMekh, oracle image indipendente, CUDA/ROCm/distribuito e altri
chip Apple non sono stati provati in questa sessione. Nessuna capability
pubblica viene promossa.
