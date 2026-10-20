# Planner, executor e bundle nativi T05–T06

Sorgenti di partenza: `65618064d22919ad2436ded995857baa9b7d4fd8`, worktree
inizialmente pulito. Hash dei sorgenti, comandi, risultati e limiti della prima
implementazione, misurata prima del commit, sono registrati in
`verification.json`; log e output voluminosi sono sotto
`performance-results/native-suite-t05-t06-20261020/`.
Il successivo riesame di responsabilità e comportamento è documentato nella
[revisione architetturale](../../architecture/native-suite-clean-code-review-20261020.md)
e in `review-verification.json`, con log sotto
`performance-results/native-suite-review-t05-t06-20261020/`.

## Pianificazione e transazione

`ds4_executor.c/.h` riceve descrittori pubblici di gruppi, visite e vittime;
non legge strutture private dei modelli. Le varianti descrivono l'intera
operazione, comprese visite ripetute, gruppi trattenuti e working set. Il costo
include ogni reload necessario. Le vittime distinguono cache ricostruibili,
sessioni e pesi; espongono byte recuperabili, checkpoint, temporanei di restore
e callback idempotenti. I costi devono usare una stessa unità: l'adapter engine
fornisce stime in byte trasferiti, sostituibili con costi misurati.

Il planner sceglie prima una variante senza eviction. Altrimenti confronta le
combinazioni ammissibili e applica l'isteresi richiesta. Considera budget
device/host/pinned/checkpoint, disponibilità globale e prenotazioni, incluso
il ripristino nell'ordine inverso usato dall'executor. Un limite ridotto sotto
le allocazioni vive non diventa erroneamente capacità utilizzabile sommando
soltanto i byte liberabili. L'API ammette fino a 64 gruppi e 16 vittime;
oltre questi limiti restituisce un errore esplicito.

Una transazione runtime trattiene il lock seriale durante tutte le visite ai
modelli. L'executor valida, ammette il checkpoint prima di alterare lo stato,
esegue drain/quiesce, salva tutte le sessioni interessate, rilascia le vittime,
prenota il picco, carica/esegue/scarica i gruppi e ripristina. Soltanto dopo
un restore riuscito libera la prenotazione e invoca `commit`. Le allocazioni
dei backend consumano la prenotazione scoped; il free restituisce capacità
riutilizzabile. Le reservation nidificate dei checkpoint prendono in prestito
la stessa capacità senza doppio addebito.

Il callback `cancelled` riceve cancel, timeout o disconnect dal chiamante.
Il recupero continua a drenare e ripristinare anche dopo la cancellazione.
`recover` ripete soltanto cleanup e restore, mai compute o richieste già
parzialmente avanzate. Gli errori recuperabili conservano checkpoint,
reservation e bitmap consultabili con `ds4_execution_recovery_query`; `close`
restituisce `BUSY` finché rimane lavoro di recupero. Un errore `DEVICE_LOST`
espone `UNAVAILABLE`. Dopo riparazione esplicita del backend si può riprovare
il cleanup, ma il risultato originale resta fallito, anche quando il device
è stato perso durante il restore successivo a compute riuscito.

`ds4_engine_reclaim_create` produce descrittori per le sessioni residenti e i
pesi Qwen27 CUDA, riusando T04. L'adapter possiede stime e checkpoint; l'executor
vede callback. Il checkpoint già scritto viene trasferito alla sessione senza
serializzarlo due volte. Engine, sessioni, descrittori e contesti devono restare
vivi fino alla chiusura dell'operazione. Qualifica: Qwen27 CUDA single GPU. Il
wrapper vision legacy mantiene il percorso T04; il routing comune HTTP è T17.

## Bundle e gruppi di pesi

I manifest in `config/suite-bundles/` fissano revisione e SHA-256 di pesi,
configurazione, tokenizer, processor e scheduler. Qwen contiene text encoder,
DiT e VAE; Gemma contiene testo, vision, audio e le due proiezioni. I selector
assegnano ogni tensore esattamente a un componente e, dove presente, al layer.
I componenti Gemma condividono una sola mmap del file pesi. Il compiler verifica
compatibilità fra revisione della recipe, componenti e asset.

`tools/suite_bundle.py build` verifica dimensioni e SHA-256 effettivi e produce
`bundle.json` canonico e `bundle.ds4b`. Il formato binario little endian ha
header di 64 byte: magic `DS4BND\1\0`, ABI a offset 8, lunghezza payload a 16,
SHA-256 del payload a 24 e otto byte riservati a zero a 56. Seguono stringhe con
lunghezza uint32, file, componenti e record tensoriali. L'indice è limitato a
64 MiB. Il loader valida lunghezze, contatori, layout e overflow; rifiuta host
big endian, revisioni non fissate, mismatch file/componente e indici corrotti.

`ds4_bundle_open` verifica ogni artefatto con un buffer di 64 KiB e apre mmap
read-only dei byte originali. Prenota il limite conservativo dei metadata
prima delle allocazioni, poi contabilizza i byte persistenti effettivi. MMAP,
host e device hanno ledger separati. I file devono restare immutabili:
dimensioni e mtime/ctime sono ricontrollati dopo l'hash e prima/dopo ogni
upload. All'apertura non avvengono dequantizzazione, conversioni globali o una
seconda copia dei pesi.

Shape e stride seguono l'ordine GGUF, con dimensione zero contigua. Il loader
copre rank 0–8, scalari BF16, shape VAE a cinque dimensioni, endian,
allineamento, blocchi e offset effettivi. Scale norm e calibrazioni restano
nell'inventario; configurazioni e parametri per layer restano negli asset
verificati. Prima dell'inferenza l'adapter deve usare
`ds4_bundle_validate_component` con l'inventario atteso esaustivo: tipi, shape,
gruppi e nomi devono coincidere. L'interpretazione numerica è T07–T16.

Il tipo GGML viene letto dal singolo tensore: Q4_K_M non impone il tipo a tutta
la directory. Le geometrie Q8_1 (36 byte) e IQ1_S (50 byte) sono confrontate con
`cuda/mmq/ggml-common.h`; la tabella legacy di `ds4.c` resta invariata. Formati
AWQ/BnB/ConvRot non diventano implicitamente GGML né ricevono kernel dal loader.

`ds4_weight_group` impacchetta un solo gruppo nello storage del backend tramite
allocate/write/quiesce/release pubblici. Il backend contabilizza lo storage;
il loader contabilizza i metadata. Il gruppo diventa residente soltanto dopo
copie complete e fence. Un upload parziale richiede release/drain prima del
retry; mmap e metadata sopravvivono al rilascio dei pesi. Un bundle con gruppi
vivi rifiuta la chiusura.

La conversione safetensors→GGUF è offline, a blocchi, deterministica e conserva
i bit BF16/FP32. Gli scalari sono rappresentati fisicamente con una dimensione
GGUF di valore 1 e registrati in `ds4.scalar_tensors`, così l'inventario logico
mantiene rank zero. Non è una quantizzazione.

## Strumenti offline

Rigenerare i profili e compilare gli snapshot locali T00:

```sh
python3 tools/prepare_suite_bundles.py \
  --root performance-results/native-suite-oracles-20261020/models \
  --output config/suite-bundles \
  --compile-output performance-results/native-suite-t05-t06-20261020/bundles
```

Download esplicito di un componente e degli asset della recipe:

```sh
./download_model.sh suite-bundle \
  --manifest config/suite-bundles/qwen-image-2.1-turbo-bf16.json \
  --root models/suite --component vae
python3 tools/suite_bundle.py build \
  --manifest config/suite-bundles/embeddinggemma-2-bf16.json --root models/suite \
  --output models/suite/embeddinggemma-2-bf16
python3 tools/suite_bundle.py discover \
  --manifest config/suite-bundles/embeddinggemma-2-bf16.json --root models/suite
```

`download` verifica Range/Content-Range, riprende `.partial`, controlla SHA-256
prima del rename e non modifica `ds4flash.gguf` o il modello selezionato. Il
download selettivo non completa il bundle: `build` richiede tutti i file.
`discover` distingue file verificati, host/device e disponibilità d'inferenza;
restituisce `ready=false` finché manca un adapter qualificato. Python non viene
avviato dal percorso di inferenza.

## Verifiche e limiti

| Gate | Esito |
| --- | --- |
| Budget: no eviction, cache, sessioni, soli pesi, combinazione, streaming, isteresi | PASS |
| Fault prima/durante/dopo compute, cancel, drain, release, restore, device perso | PASS; nessun replay o falso successo |
| Qwen27 reale: nessuna eviction, park/resume, weights-only | PASS; continuazione bit-exact nei tre casi |
| Inventario Qwen: 16 artefatti, 3 componenti, 1.285 tensori | PASS contro bundle compilato e T00; 32.451.607.098 byte |
| Inventario Gemma: 12 artefatti, 5 componenti, 1.376 tensori | PASS contro bundle compilato e T00; 480 scalari; 1.525.786.076 byte |
| Conversione ripetuta, tipi misti, binding, corruzione, budget host, lifetime/upload parziale, download/resume | PASS |
| Executor e loader con ASan/UBSan e leak detection | PASS |
| Runtime/resources/checkpoint, contratti, product text, server, session state, read queue/SSD | PASS |
| Qwen testo/vision, MTP off/on, contro T00 | PASS; logits e payload bit-exact |
| Flash Next launcher e streamed backing, runtime Qwen→Flash→Qwen | PASS; dump Flash bit-exact |
| CUDA SSD cache | PASS |
| Metal, ROCm, distributed; nuovi kernel/inferenza Qwen Image e Gemma | NOT_VERIFIED; fuori dalla qualifica locale T05/T06 |
| GGUF Unsloth Q4_K_M candidato | NOT_VERIFIED; manifest fissato, pesi assenti; qualifica numerica T20 |

Il picco RSS del processo C che verifica hash e apre mmap è riportato in
`verification.json`. Misura apertura e dump, non un forward o il picco VRAM di
inferenza. Non viene dichiarato un guadagno prestazionale; questi test non sono
un benchmark statistico di latenza. I warning di build riguardano
parametri/funzioni legacy; i nuovi moduli compilano senza warning.

Un primo tentativo di preparazione includeva il chat template standalone Gemma,
assente dallo snapshot locale e non usato dalla recipe embedding. È stato
escluso esplicitamente; il template del processor Qwen resta incluso. Il log
riuscito è `bundles-verified.log`; i tentativi precedenti sono conservati.
