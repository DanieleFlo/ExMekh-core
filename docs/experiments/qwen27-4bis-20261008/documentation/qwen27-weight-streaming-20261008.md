# Qwen 27B: KV residente e streaming dei pesi, 8 ottobre 2026

Su richiesta dell'utente, il nuovo obiettivo sostituisce il KV offloaded del
4 BIS: massimo 65536 token complessivi, KV FP32 completo e stato ricorrente
in VRAM, pesi GGUF originali divisi tra residenti e complemento RAM.
NextN mantiene cinque proposte; non riprende lo sweep delle profondità.

## Percorso sperimentale

`--qwen-weight-streaming on` richiede Qwen denso 27B, CUDA e una sola GPU.
Embedding, output e tensori fuori dai layer rimangono residenti. NextN usa
il mapping ausiliario residente precedente. I layer da trasferire sono
distribuiti lungo il forward: fra due consumi remoti ci sono finestre di
calcolo sui pesi residenti. La scelta usa le dimensioni effettive dei tensori,
non una dimensione uniforme inventata.

Il budget `--qwen-weight-resident-mb N` riguarda il pack residente target.
Zero sceglie il budget dalla VRAM libera sottraendo KV target/NextN, pesi
NextN, ricorrenza, snapshot, workspace, margine e due slot di streaming.
Gli slot GPU contengono ciascuno un layer remoto completo. Il complemento
RAM contiene solo i tensori dei layer selezionati; non duplica il modello
intero. Le pagine sorgente GGUF vengono restituite dopo il caricamento.

Anche un budget esplicito viene limitato dalla VRAM effettivamente libera.
La riserva conta le allocazioni del contesto richiesto, i cinque snapshot
NextN (o la profondità esplicita), i pesi ausiliari, il massimo fra scratch
di dequantizzazione e score dell'attenzione, due slot e 2 GiB di margine.
Lo scratch viene preparato prima della sessione. Controlli prima/dopo il
graph e ai confini dei forward interrompono il percorso sotto 1 GiB libero,
evitando di proseguire con paging WSL. Le cache Q8 espanse opzionali sono
escluse anche sul supporto quando questa modalità è attiva.

Un worker persistente prepara due staging pinned da 64 MiB e accoda copie
esplicite su uno stream non bloccante. Avvia i primi due layer remoti prima
dell'embedding. Dopo l'ultimo consumatore di un layer, un evento autorizza
il riuso del suo slot per il layer remoto successivo. Lo stream di calcolo
attende soltanto l'evento del layer corrente. La sequenza vale anche per
prefill e verifica MTP multi-riga: una copia serve tutte le righe.

Gli offset GGUF vengono risolti verso pack residente o slot corrente;
una lettura fuori dal layer è un errore, senza fallback a RAM mappata.
Non si introducono kernel, quantizzazioni o precisioni differenti.
Le cache opzionali di pesi Q8 dequantizzati non possono trasformare il
complemento remoto in una seconda residenza permanente. Worker, pinned,
mappe e slot hanno lifetime dell'engine; KV e snapshot quello della sessione.

Il flag conserva il KV sul percorso residente anche con variabili del vecchio
offload ereditate. Snapshot ricorrenti MTP restano GPU. Checkpoint e restore
mantengono il formato logico e non serializzano placement o eventi dei pesi.
Il limite di 65536 viene verificato anche alla creazione della sessione.

## Provenienza e qualificazione

Da Strata si riprendono ordine di consumo noto, residenti separati da
trasferiti, ring bounded e eventi copia/consumo. Il suo `native_dense`
non costituisce un backend pronto del 27B locale. La numerica rimane quella
dei kernel DS4, con i formati locali del GGUF.

La qualificazione deve confrontare primitive con oracle indipendente,
slot riutilizzati, tutti-residenti, errori di metadata, drain, logits completi,
MTP5, rollback, cancellazione, checkpoint/restore, mRoPE e vision. Seguono
contesti effettivi piccoli, 32K e fino a 64K inclusi i token generati.
Byte H2D, attese, RAM, VRAM, prefill e decode vengono confrontati con il
precedente KV offload. L'implementazione resta sperimentale fino ai report.

La nuova capacità non dipende dalla riduzione della precisione KV. Il costo
PCIe è proporzionale ai pesi remoti per forward, invece di crescere con ogni
nuovo token storico; la sovrapposizione effettiva e il guadagno sono da misurare.

La v61 supera la diagnostica 65503+32 token/NextN5 con logits completi
bit-identici al riferimento KV offload. Il prefill misura 393,10 contro
370,71 token/s; decode totale 0,91 contro 1,05. Il margine misurato durante
prefill è circa 1,8 GiB (`nvidia-smi`), oltre 1,5 GiB dopo graph per CUDA.
La pressione VRAM iniziale è corretta, ma il trasferimento di 5,649 GiB
di pesi per forward limita il decode. È un confronto diagnostico singolo,
quindi la modalità rimane sperimentale e non sostituisce altre modalità
sulla base di questo report. Vedi `weights-v61-64k/report.json` nell'audit.
