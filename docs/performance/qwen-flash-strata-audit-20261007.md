# Riesame Strata: Flash prefill e 27B decode, 7 ottobre 2026

**Aggiornamento 8 ottobre:** su richiesta dell'utente, il 4 BIS è ritirato.
KV offload, streaming dei pesi e speculazione estesa del 27B sono rimossi
dai percorsi attivi e raccolti nell'[archivio documentato](../experiments/qwen27-4bis-20261008/README.md).
Le sezioni 27B qui sotto restano cronologia delle prove; lo sviluppo riprende
sui soli punti Flash. Il 27B ordinario resta un gate di regressione.

## Fonte fissata

Aggiornamento dell'8 ottobre: la [nuova verifica del prefill](qwen-flash-prefill-strata-20261008.md)
fissa Strata `6674a0065fb96bacde33e3eb10f91a1df86f95f2` e descrive gli
incrementi v64–v66 successivi al ritiro del 27B. Le sezioni sottostanti
conservano la cronologia; il vecchio confronto fra gruppi GU e down è
superato dal ring di esperti completi già implementato nella v57.

Nuova copia locale: `performance-results/flash-strata-20261006/strata-source-20261007`.
Commit `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a`, del 7 ottobre alle
13:22:57 +0200. Il precedente riferimento era
`1735d6471df29b42c26170efaac1f1446a58640f`.
La licenza è MIT, copyright 2026 Niko1221 e contributori. Questa ispezione
non importa kernel o modifica GGUF, quantizzazione dei pesi o PLE BF16.

| File esaminato | SHA256 |
| --- | --- |
| `LICENSE` | `949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0` |
| `src/prefill/prefill.cpp` | `cfe076cc4b34cb03018c30a455bc0f5c583f8ee47fd4c33afc4bcd6624f42be3` |
| `src/prefill/moe_fused_iq.cu` | `2d17ca116a4013d79ccf02794659554a163d4f7002a9ce88700165ad878cafef` |
| `include/strata/prefill/moe_fused_iq.hpp` | `84d0c99ace277a53deb5d091b2a0a5137fdc390f834815b43e8a5a910f42b4c8` |
| `src/core/verify.cpp` | `afd975cc5a8771219700d70c32cd47fc0d59b4c2d83f9bc62b4d535771da3f96` |
| `src/core/expert_source.cpp` | `2f0e14294ba463fd4fbcfb58829865c5f33712bcc72c6856ebbe2ff595b6e208` |
| `src/core/expert_cache.cpp` | `91eb2373d1893ff2e6fe88887a121f525549ad0d66184dee190e1ad8d07946d2` |
| `src/kernels/cuda/native_mmvq.cu` | `43caa800023d7e223629ab8c6a28bd10c50b56c2dd7232b2b5fa4c00144b2e98` |

I link ai sorgenti devono usare questo commit, anziché una testa mobile:
[prefill](https://github.com/Niko1221/Strata/blob/e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a/src/prefill/prefill.cpp),
[verifica](https://github.com/Niko1221/Strata/blob/e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a/src/core/verify.cpp).
Il termine `native_dense` di Strata indica le proiezioni dense di Flash:
non è un backend già pronto per il modello Qwen 27B di DS4.

## Stato rispetto al piano iniziale

| Priorità | Implementazione disponibile | Qualificazione e lavoro residuo |
| --- | --- | --- |
| 1: MoE CPU/GPU | Piano completo gate/up/down; attivazioni FP32 CPU; formati IQ2_XXS e Q2_K con padding; pool persistente; slot originali e combine GPU | Decode e MTP passano anche i gate v53: checkpoint/restore, residenza parziale, RAM bounded/zero e 68 righe teacher-forced complete. Il prefill CPU misto non supera il confronto teacher-forced: KL 0,20535, massimo errore 2,417. Rimane escluso dalla selezione automatica; la qualità non viene allentata. |
| 2: tier PCIe | Quota CPU/PCIe, staging pinned bounded condiviso, override e tre famiglie del tuner: decode, verifica MTP, prefill; modello in RAM ordinaria | Griglie v57 complete: 20 combinazioni per decode e 20 per MTP, cinque campioni ciascuna; seguono cinque A/B alternati completi. Scope MTP e workspace per fase corretti. Prefill automatico breve qualificato; dettaglio nella sezione dell'8 ottobre. |
| 3: KV Flash | Storico FP16 RAM, QSA originale, DMA esplicito WSL, mappe logiche/slot, pin/eventi, rollback e checkpoint; staging e prefetch per layer nel prefill | Contesti effettivi 64K e 128K qualificati v32 con logits completi bit-identici. Sono diagnostiche singole, non cinque coppie lunghe. |
| 4: profilo esperti | Conteggio una volta per router; fingerprint, caricamento/salvataggio; ranking recente, LRU e profilo persistente; quote di esperti completi nel prefill stabile | Oracle 4096 epoche e confronto delle vittime CUDA passano. La qualificazione v53 confronta cinque policy su trace ordinario e cambio codice/prosa/tool, con cinque campioni e budget uguale: logits bit-identici, conteggi singoli e fallback validati. I tempi delle policy restano diagnostici, senza promozione automatica. |
| 4 BIS: 27B storico | KV FP32 RAM e attention per tile, con NextN5 | Numerica e capacità 64K/128K verificate. L'8 ottobre l'utente sostituisce il target: massimo 64K, KV/ricorrenza GPU e streaming dei pesi; le ulteriori prove KV offload 128K vengono abbandonate. |

La v32 Flash misura a 64K 226,08 token/s di prefill e 16,64 di decode
contro 157,91 e 15,10 del riferimento; a 128K 199,71 e 14,85 contro
155,74 e 9,88. I campioni lunghi conservano i logits bit per bit, ma usano
un riferimento precedente e non soddisfano da soli la promozione A/B.
Il confronto breve v30 a cinque coppie misura +13,54% decode e −1,54%
prefill. Non si sommano percentuali di prove e binari diversi.

## Flash: ottimizzazioni estratte e omissioni concrete

`prefill.cpp` avvia per chunk grandi uno stream in ordine fisso di tutti
gli esperti non residenti, prima che il router ne richieda il consumo.
Il ring contiene blob completi GU/down; eventi `copied`/`used` proteggono
ogni slot. La soglia iniziale è 1024 righe. Per chunk piccoli Strata
mantiene la selezione dei soli esperti richiesti e può usare il pool CPU.
Questo è diverso dai gruppi DS4: DS4 copia solo gli esperti selezionati,
esegue prima tutti i gruppi gate/up e poi tutti i gruppi down. Il trasferimento
del down non viene anticipato insieme alla tripletta e il confine fra le
due proiezioni drena lo stream. È un'omissione utile da misurare.

Strata bilancia inoltre **byte di ring, workspace del chunk e prestito della
cache esperti**, invece di fissare un numero di slot indipendente dal pack.
Ridurre il workspace dei kernel fusi permette chunk più grandi e ammortizza
gli stessi byte esperti su più token. DS4 ha staging doppio, RAM complementare,
PLE anticipato, prefetch KV e quote complete; non ha ancora estratto tutto
il ring prima del router, il prestito o la scelta congiunta chunk/ring.
Un confronto fra chunk diversi deve dichiarare entrambi e non viene usato
come misura del guadagno a chunk invariato.

Il percorso Strata `moe_fused_iq.cu` copre IQ2_XXS per GU, ma il down CUDA
fuso copre Q2_0 e IQ4_NL, **non Q2_K** del GGUF locale. Quantizza inoltre
attivazioni e intermedio SwiGLU in int8 per gruppi di 32. Il port non è
quindi una semplice chiamata mancante: servirebbero un kernel Q2_K con
scale/min originali, padding 640→768 e qualificazione numerica distinta.
DS4 fonde già gate/up e SwiGLU nei propri kernel tensor core FP16; non
introduce la quantizzazione int8 delle attivazioni di Strata.

`mmq_resident_sort.hpp` è opzionale e riguarda il pack non nativo con tutti
gli esperti residenti. Non corrisponde al nostro GGUF nativo con residenza
parziale; non è una lacuna da correggere attivando indiscriminatamente il flag.

## 27B: cosa si può riutilizzare dal decode Strata

La verifica Strata conserva grafi di finestre, consuma residenti e miss in
percorsi distinti e può committare sullo stesso stream senza una fence host
intermedia. Un evento pubblica il commit agli altri consumatori. Il grafo
senza passaggi host richiede la residenza dell'intero working set.
Nel 27B esteso i pesi sono residenti, ma il KV remoto richiede preparazione
RAM e DMA esplicito; non si può eliminare quel traffico catturando soltanto
il forward in un grafo CUDA.

`STRATA_VERIFY_QDEDUP` riusa in particolare la stessa immagine Q8_1 fra
esperto condiviso ed esperti instradati, quando non usa il fork su uno
stream separato (`verify.cpp`, blocco `shared_expert_multi`). È un caso MoE
Flash: il 27B denso non possiede quella coppia di consumatori. Strata
intercala inoltre 2–4 righe nel kernel MMVQ. DS4 esegue già la verifica 2–16 righe
con kernel 2/3/4 righe e riduzioni identiche al decode per tutti i formati
locali. La deduplicazione fra proiezioni che leggono lo stesso input non è
ancora generalizzata: va qualificata sul doppio residuo Q8_1 locale e con
lifetime esplicito, anziché applicare una cache basata soltanto sul puntatore
allo scratch riutilizzato. Le fusioni PLE/HC/QSA e il fork dello shared expert
sono specifici di Flash e non diventano vantaggi MoE sul 27B denso.

La cache KV NextN nel workspace target già esistente applica il principio
di riuso: a 64K il suo layer intero resta disponibile fra proposte e cicli,
senza un secondo budget VRAM. NextN5 v46 passa token e logits completi AR
bit per bit e misura 1,90 token/s complessivi contro 0,67 di AR. Il costo
medio è 18,73 ms di proposta, 1841,09 ms di verifica e 56,76 ms di rollback.
La verifica rilegge circa 4 GiB remoti per ciclo a 64K: il link Gen4 x2
misurato a circa 3 GiB/s resta il costo dominante. La proposta a 5 non
equivale a cinque token accettati; nel campione sono 3,875 token utili per
verifica. Le profondità maggiori non vengono più confrontate su richiesta
dell'utente.

## Preparazione NextN per chunk: controllo v48

Le primitive v47 confrontano KV e carry completi con il percorso seriale
anche a 32K/64K, mRoPE, vision e rollback; tutti i sei gate passano.
A 64K effettivi v48 separa target e preparazione NextN. Con preparazione
seriale: target 167,64 s, preparazione 13,18 s. Per chunk: target 168,10 s,
preparazione 7,95 s. Le due continuazioni NextN5 coincidono bit per bit
prima e dopo 32 token e misurano circa 1,90–1,91 token/s complessivi.

Il secondo controllo AR differisce dal primo AR e da entrambe le
continuazioni NextN5 (massimo 0,00717, KL circa 4,07e−7 alla frontiera).
Il report `qwen27-nextn-v48-phases-64-on/report.json` conserva quindi
`all_pass=false`. La ripetizione del controllo AR è distinta e non cancella
questa evidenza. Il controllo AR ripetuto
`qwen27-nextn-v48-phases-64-on-control2/report.json` coincide bit per bit
con entrambe le continuazioni NextN5. Non sono state cambiate le soglie.
La variante seguente abilita il batching nella sola modalità CUDA estesa
offloaded, mantenendo l'override seriale e il carattere sperimentale della
modalità. Il default NextN rimane 5.

## Incremento successivo al riesame

`ds4_qwen4_ring.cuh` aggiunge un ring sperimentale di esperti completi:
`DS4_CUDA_QWEN4_PREFILL_RING=1`, con policy `stable`, Flash CUDA singola GPU
e prefill oltre otto righe. `PREFILL_RING_MB=1..1024` assegna il budget
totale dei due buffer (default 128 MiB); le dimensioni GGUF delle tre
proiezioni determinano il numero di esperti per gruppo. Metadata pinned e
conteggi GPU sono contabilizzati separatamente. Gli hit restano nelle
proprie allocazioni; ogni proiezione mancante viene trasferita una volta.

Il gruppo esegue i kernel originali GU/SwiGLU e Q2_K down, conservando
padding e slot `(token, expert)`; l'evento di consumo viene pubblicato solo
dopo il down. Il gruppo seguente può trasferirsi durante entrambi i kernel.
Gli errori drenano stream e worker prima di riutilizzare lo storage.
I flag non attivano il percorso sul 27B. È ancora una differenza rispetto
al ring Strata: il piano usa il router corrente e non anticipa gli esperti
di layer successivi prima del routing. Prestito cache e scelta congiunta
chunk/ring rimangono da valutare separatamente.

I sette gate v49 passano: primitive con avvolgimento, duplicati, proiezioni
parzialmente residenti, RAM zero e bounded; checkpoint/restore/MTP Flash;
68 righe teacher-forced complete bit-identiche; isolamento 27B, mRoPE e
cancellazione NextN. La diagnostica `ring_groups` conferma l'uso effettivo.

Cinque coppie alternate sullo stesso binario v49, 2048 token effettivi,
capacità 8192, chunk 1024, due warm-up e 16 token di decode: prefill mediano
386,88 → 386,67 token/s (−0,05%, CV 1,76%/1,06%); decode 32,41 → 31,42
(−3,05%, CV 3,58%/1,48%). I logits completi coincidono in tutte le coppie.
I gruppi trasferiti scendono da 4608 a 3456. Il ring rimane opt-in:
questa prova non misura un vantaggio, né completa anticipo e prestito cache
di Strata. Report `strata-ring-v49-paired/ring-paired.json` e
`strata-ring-v49-gates/qualification.json`; binario bench
`7164c27c77cc951c53dd878045c3b8c0439bb12be4e8612837cf140309cbc99d`.

## 27B: arena di rollback dopo il prefill

Le cinque righe di verifica conservate per i 48 layer ricorrenti occupano
circa 0,740 GiB. Il percorso RAM trasferisce queste righe D2H per ciclo,
poi H2D lo stato scelto; a 64K il rollback misura circa 57 ms. La nuova
variante `DS4_CUDA_QWEN_MTP_DECODE_ROWS=1` tenta una sola allocazione GPU
all'inizio del decode, riutilizza l'arena tra cicli e la restituisce prima
del nuovo prefill. Le snapshot di annullamento complete rimangono separate.
Gli stati e l'aritmetica restano FP32 originali.

Il tentativo drena e restituisce lo scratch temporaneo del prefill sopra
128 MiB, invalida i grafi che lo referenziano e riserva crescita KV promessa,
una eventuale allocazione old/new e 128 MiB per il workspace di verifica.
La verifica fino a 16 righe usa la quantizzazione delle attivazioni già
presente nei kernel del percorso originale e partials
split-K bounded, invece della matrice di score del prefill. L'allocazione
opzionale ha un budget di 1024 MiB (`DS4_CUDA_QWEN_MTP_DECODE_ROWS_MB`,
0..8192, zero forza RAM) e può fallire: mantiene RAM e scratch condiviso e non riprova
ogni token. Il checkpoint conserva soltanto i dati logici; arena e validità
intermedia non ne modificano il formato. La variante resta opt-in durante
la qualificazione.

La v53 ammette anche una parte delle righe quando l'arena intera non entra:
conserva un prefisso in VRAM e trasferisce in RAM soltanto la coda, senza
modificare il kernel ricorrente o il suo ordine aritmetico. Il ripristino
usa D2D per il prefisso e il percorso H2D originale per la coda. Il test
forzato a 512 MiB conserva tre righe su cinque e confronta ogni rollback,
frontiera e continuazione con l'oracle interamente residente. Passano anche
zero MiB, annullamento, riuso dopo restore e restituzione prima del prefill.
I 12 gate finali preliminari sono registrati in
`decode-rows-v53-gates/qualification.json`, tutti con esito positivo.

A 64K effettivi la diagnostica v51 conserva tutte le cinque righe: decode
complessivo 1,916 → 2,246 token/s, verifica media 1824,49 → 1574,58 ms,
rollback 56,07 → 1,26 ms. Token e logits completi prima/dopo 32 token
coincidono bit per bit; il confronto mantiene NextN5. A 128K v51 la memoria
libera dopo restituzione dello scratch non basta per tutte le righe: usa
RAM, passa la stessa verifica numerica e misura circa 0,380 token/s contro
0,348 del controllo v49. Questa variazione non viene attribuita all'arena
inattiva né al prefill, che precede l'attivazione. Entrambe sono diagnostiche
singole (`decode-rows-v51/report.json`), non promozioni a cinque coppie.

La ripetizione v53 a 131072 token effettivi più 32 token, dopo warm-up
di 16 e restore alla stessa frontiera, ammette due righe su cinque
(0,296 GiB). Entrambi i logits completi e tutti i token coincidono bit per
bit con v51: `decode-rows-v53-128-resume1/report.json`, `all_pass=true`.
Decode complessivo 0,39160 token/s, verifica media 4510,07 ms. Il restore
del prefisso costa circa 1–2 ms, mentre la coda conserva circa 55 ms.
Il prefill della ripetizione misura però 81,15 token/s contro 194,34 v51;
anche il run v53 interrotto, sul medesimo binario, era più veloce nel
prefill. Questa regressione resta aperta e non è attribuita all'arena,
attivata soltanto dopo il prefill. La sonda durante la prova registra
Gen4 x2, assenza di swap e circa 0,3 GiB VRAM liberi. Non prova una causa
hardware: occorre una diagnostica separata. Nessuna promozione prestazionale
viene ricavata da questo confronto singolo.

## Correzione della calibrazione MTP

Il riesame ha trovato una lacuna nel tuner locale: una sola famiglia per
T≤8 mescolava decode ordinario e verifica. `DS4QWEN4TUNE2` registra ora tre
scelte distinte: T=1, T=2..8, T>8. Le tre famiglie conservano pool e staging
persistenti anche con worker diversi, evitando teardown fra decode e MTP.
Un profilo V1 valido mantiene decode/prefill ma conserva il precedente
percorso GPU per MTP. Fingerprint incompatibili, righe troncate, NaN,
worker fuori range e dati residui non attivano una preferenza.

`tests/test_qwen4_tuning.cpp` verifica il loader e la selezione reali.
La griglia MTP misura l'intero decode, incluse proposta, verifica, attese e
combine, e richiede chiamate di verifica multi-riga effettive. Mantiene le
impostazioni MTP Flash iniziali. Aggiorna soltanto la scelta MTP del profilo,
conservando le scelte decode/prefill già calibrate con lo stesso fingerprint.
La selezione rimane sperimentale fino ai confronti A/B e ai gate di qualità.

Le prime due griglie v53 completano tutte le 20 combinazioni ciascuna,
con due warm-up e cinque campioni, sul binario congelato
`8536e4814f640ae825d1c988fc1a034dccc33cf7482932dbca82ff0926184ac5`.
La griglia ordinaria sceglie quota PCIe 0,5 e otto worker per il decode,
GPU 100% per il prefill. Il controllo congiunto conserva entrambi i logits
completi bit per bit e passa il limite di regressione del 5%.
`calibration-v53/calibration.json` resta una preferenza sperimentale in
attesa di cinque coppie alternate.

La griglia MTP a 16 token conserva i logits bit per bit in tutte le quote,
ma il riferimento ha CV 22,67% sui cinque campioni (12,08–22,46 token/s).
Di conseguenza nessuna quota MTP è eleggibile: la riga MTP del profilo
conserva il percorso precedente. Questo esito non dimostra una regressione
dei kernel o una quota ottimale. Si ripete con cinque warm-up, 64 token e
cinque campioni, senza cambiare la profondità MTP Flash; segue il confronto
alternato A/B sul profilo definitivo. Il report precedente non viene
cancellato o reinterpretato come promozione.

## Profilo persistente sul candidato finale

`profile-v53/report.json` registra `all_pass=true`: policy recente, LRU,
profilo appreso, stabile e stabile con profilo; due warm-up e cinque
campioni, frontiere 256/512 e trace con cambio codice/prosa/tool a
256/512/768. Tutti i confronti dei logits completi coincidono bit per bit,
così come le 68 righe teacher-forced. Il training iniziale e il successivo
apprendimento contano ogni scelta una volta per layer/token; i 48 layer
target hanno i delta attesi e il layer NextN inutilizzato non riceve conteggi.
Il file sorgente del profilo resta immutato: ogni prova usa una copia.
Troncamento, fingerprint errato e checksum errato vengono ignorati con
continuazioni identiche al riferimento. Questi risultati qualificano il
caricamento/apprendimento e l'isolamento, non una policy universalmente
più rapida: i confronti end-to-end e la variabilità determinano le preferenze.

## Gate vision e correzione del replay dopo rollback

La qualificazione finale v53 conserva un fallimento: `vision-v53/report.json`,
caso API Flash/MTP/thinking. Payload multimodali, checkpoint e frontiere
complete di tutti i quattro casi sono bit-identici; 39 controlli API su 40
passano, ma quel caso arriva al limite di 512 token dentro il ragionamento.
Il report negativo resta conservato e ferma la coda dei benchmark successivi.

Le ripetizioni della stessa fixture isolano il flag dei grafi: MTP ordinario,
scheduler ibrido da solo, policy stabile da sola, KV streaming (con e senza
pubblicazione differita) e QSA register da solo passano. `DECODE_GRAPHS=1`
da solo riproduce il fallimento. I report sono in
`vision-thinking-v53-isolation`, `-step2` e `-step3`.

Il rollback Flash scambia le allocazioni di stato e history GDN con le
snapshot. La chiave del grafo conservava residui e flag di snapshot, ma
non gli indirizzi dei due banchi: poteva riutilizzare un kernel che scriveva
nel banco diventato snapshot. La correzione aggiunge i riferimenti live,
snapshot e snapshot2 e il residuo alternativo alla chiave, mantenendo
riutilizzabili entrambi gli assetti. Gli slot per layer/isola passano da
quattro a otto per le forme 1/2/3 righe e i due banchi. Un test CUDA con
contatori indipendenti alterna i banchi e richiede replay effettivi.
Si ricompilano tutti i binari come v54; gli esiti v53 restano legati ai
propri hash. La primitiva `graph-banks-v54-primitive` passa ed esegue replay
reali. Anche le venti verifiche API di `vision-thinking-v54-fixed` passano:
grafi da soli e tutti i flag con quota PCIe 0,5. Il vecchio binario v53
con `DS4_QWEN4_MTP_SWAP_RESTORE=0` passa gli stessi dieci casi, confermando
il legame con lo scambio dei banchi. Fixture, limiti di 512 token, sampling
e tolleranze restano invariati. Anche i 16 gate di
`graph-banks-v54-gates/qualification.json` passano: MTP/budget, stato,
checkpoint, 68 righe teacher-forced con tutti i flag, backend CPU/GPU,
fallback RAM e isolamento Native 27B. Il tuner incrementa il fingerprint
del scheduler: il profilo v53 non attiva preferenze sulla correzione v54.
Calibrazione, profili, vision completa e coppie A/B vengono ripetuti sul
nuovo artefatto; i TPS precedenti non vengono trasferiti al binario corretto.
In particolare, la griglia MTP v53 usava grafi sia nel riferimento sia
nei candidati: l'identità reciproca dei logits non elimina il difetto
condiviso. Questi tempi restano evidenza storica, senza promozione; la
griglia viene sostituita da una nuova misura v54, senza cancellare i dati.

## Chiusura Flash breve e nuovo target 27B, 8 ottobre

La v57 corregge due costi dell'integrazione: lo scope MTP comprende anche
le proposte T=1; i workspace delle tre famiglie vengono preparati al primo
uso, con gruppi piccoli sotto 1024 righe e restituzione dello staging GPU
quando il prefill termina. La diagnostica v55 attribuiva migliaia di eviction
alle allocazioni eager di famiglie inutilizzate e al prefill stage trattenuto.
Il controllo v57 recupera decode MTP e mantiene logits completi bit-identici.

`final-flash-v57.json` è COMPLETE/all_pass: griglie ordinarie/MTP, cinque
policy con cambio di dominio, primitive/stato/teacher-forced, 40 API/vision,
regressione 27B ordinario e cinque A/B alternati per le scelte automatiche.
I 16 gate di `phase-workspaces-v57-gates` passano. Le nuove modalità sono
ancora opt-in; questo report qualifica i workload misurati.

| V57 auto, 256 token effettivi, cache calda | Riferimento | Candidato | Delta |
| --- | ---: | ---: | ---: |
| Prefill ordinario, token/s | 504,07 | 587,23 | +16,50% |
| Decode ordinario totale, token/s | 33,85 | 36,64 | +8,24% |
| Prefill con MTP iniziale, token/s | 491,36 | 581,16 | +18,28% |
| Decode MTP totale, token/s | 35,27 | 36,18 | +2,58% |

Tutti i logits completi delle coppie sono bit-identici. Il guadagno decode
MTP è inferiore al CV del candidato (5,14%) e non viene promosso; il suo
prefill supera la variabilità. La policy stabile separata resta sperimentale:
prefill breve −14%, anche se il decode guadagna circa il 10,7%.
Report `auto-decode-v57/preferences-paired.json`,
`auto-mtp-v57/preferences-paired.json`, `flash-short-v57/qualified-paired.json`.

La prima coda si fermava sul controllo dei cached token HTTP: Flash non
abilita automaticamente la cache su disco e la sessione live contiene
anche la risposta. Il benchmark ora abilita una cache privata da 512 MiB.
Cinque coppie in `http-mtp-v57-prefix-fixed/report.json` passano testo/usage
e riusano 653 token in ogni prefix hit. TTFT mediano freddo 17,577→13,497 s,
prompt nuovo a esperti caldi 8,193→8,732 s, prefix hit 0,346→0,320 s.
Il costo del prompt nuovo (+6,58%) mantiene questa combinazione HTTP
sperimentale; il vecchio report senza hit resta conservato.

Il nuovo 27B usa `--qwen-weight-streaming on`, massimo 65536 token,
KV FP32 e snapshot ricorrenti GPU. Il primo oracle indipendente passa due
slot riutilizzati, tutti-residenti e metadata invalidi. Il GGUF reale trova
un errore preesistente: la tabella IQ4_NL indicava 256 valori/50 byte invece
dei 32 valori/18 byte effettivi. La copia integrale lo mascherava; un pack
di tensori separati lo respinge per bounds. La v60 corregge soltanto il
conteggio, con un gate host dedicato e primitive originali invariate.
`weights-v60-short-mtp/report.json` passa tutti i logits completi e token:
512 token più 32, NextN5, 4,49 GiB di complemento pesi RAM e nessun KV o
snapshot ricorrente offloaded.

La prima diagnostica a 65503+32 token (`weights-v60-64k-resume1`) è stata
interrotta per pressione VRAM, dopo la segnalazione dell'utente: 24266 MiB
occupati, 61 MiB liberi e 14–19 s per chunk. Il budget esplicito di 11264 MiB
aggirava la riserva automatica. La stima di 22,64 GiB escludeva il picco
CUDA/driver; non si tratta di un risultato prestazionale completato.
Il riferimento KV-offload già concluso rimane conservato: prefill 370,71
token/s, decode totale 1,05, NextN5. Il suffisso differisce dai precedenti
report 64K, quindi non se ne trasferisce l'acceptance.

La correzione successiva limita anche il budget esplicito, conta graph e
snapshot effettivi e prealloca lo scratch massimo. Riserva 2 GiB oltre le
allocazioni previste e rifiuta sessioni/forward sotto 1 GiB libero. Prima
del contesto effettivo 64K serve una prova corta con tutte le celle 64K
allocate e misure della memoria reale. La prova interrotta non è promossa.

`weights-v61-64k-allocation-probe` completa la prova corta: tutte le celle
64K allocate, 4096 token reali più 16 generati e NextN5. La richiesta 11 GiB
viene limitata a 8,738 GiB; il pack effettivo è 8,644 GiB, complemento RAM
5,649 GiB, ring 0,469 GiB, staging pinned 0,125 GiB. I logits completi e i
token coincidono bit per bit. I chunk target tornano a 1,94–1,95 s dopo
il primo, rispetto ai 14–19 s della prova satura. La sonda `nvidia-smi`
legge 1955 MiB liberi; CUDA, dopo graph, legge 1598 MiB. La differenza
osservata conferma che la sola somma dei tensori non descrive il picco reale.

`weights-v61-64k/report.json` completa 65503 token più 32 (totale 65535),
NextN5, usando il riferimento già finito dello stesso workload. Tutti i
logits completi prima/dopo decode sono bit-identici e KL zero. KV e snapshot
ricorrenti rimangono GPU; CUDA dopo graph legge 1607 MiB liberi, la sonda
durante prefill 1808 MiB. Prefill 370,71→393,10 token/s (+6,04%); decode
totale 1,05→0,91 (−13,33%), steady 1,08→0,94. È una diagnostica singola:
correttezza e capacità passano, ma non promuove prestazioni. Il costo dei
pesi remoti è 5,649 GiB per forward, incluso il riuso fra righe MTP;
91 forward trasferiscono 514,056 GiB complessivi. Risolta la pressione
VRAM, il decode richiede ancora riduzione/ammortamento del traffico.

La v62 aggiunge il contesto API esplicito al planner quando manca il hint
di placement. Le primitive rifiutano richieste graph/scratch impossibili
senza distruggere il piano, poi eseguono riuso di slot e tutti-residenti.
`weights-v62-wide-state.log` confronta NextN5 con un oracle indipendente
residente per tutti i cinque rollback: stato ricorrente/conv, logits completi
e continuazione sono bit-identici anche con un layer realmente trasferito.
`weights-v62-callback.log` passa cancellazione, payload/checkpoint, KV/MTP
e RNG; il complemento pesi è 0,192 GiB nel gate. La profondità non cambia.
