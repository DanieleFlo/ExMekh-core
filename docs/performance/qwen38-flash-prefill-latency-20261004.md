# Flash Next: lentezza del prefill e budget RAM, 4 ottobre 2026

Il doppio prefill e la lentezza del singolo passaggio hanno cause distinte.
La correzione del confine di sistema evita il replay, ma il prompt completo
rimane costoso: gli esperti attraversano continuamente SSD, RAM e PCIe.
Questo audit identifica un limite software correggibile, il tetto RAM di
16 GiB, e un limite hardware misurato, il collegamento GPU Gen4 x2.

## Metodo e provenienza

Baseline congelata: `5d374d4978c644a3de9ce40353b3dcba9b97d235`. CUDA `sm_86`,
RTX 3090, i9-14900K, WSL Ubuntu 24.04. La RAM fisica è circa 64 GB;
WSL espone circa 47 GiB e il processo rileva circa 43,5 GiB disponibili
prima della cache degli esperti. Il GGUF Flash Q2 locale è di 147207127040
byte, inclusi circa 95 GiB di PLE; pesi e quantizzazione restano invariati.
Il file non è stato ricalcolato con SHA-256 durante questi benchmark:
si conservano dimensione e metadata locali, oltre agli hash dei binari e
all'equivalenza numerica. Non è un confronto fra diverse conversioni GGUF.

Il corpo HTTP è stato catturato prima della rete tramite l'API pubblica
`execute_agent_turn_sync`: prompt `ciao`, agente predefinito, 22 tool,
reasoning `none`, history vuota. Il server rende **4395 token**, contro i
4431 del log iniziale dell'utente: è una riproduzione del frontend reale,
non la ricostruzione byte-identica di quella sessione. Ogni variante usa
lo stesso corpo, ctx 131072, chunk 256, vision corrispondente, MTP attivo,
temperatura zero e limite di 32 token generati. `incomplete/length` è
atteso per questo limite diagnostico, non indica un errore di inferenza.

Le misure interne con `DS4_QWEN4_TIMING=2` e `DS4_QWEN4_MOE_PROFILE=1`
introducono sincronizzazioni: servono all'attribuzione, non alla promozione
di un risultato prestazionale. I benchmark `ds4-bench` e le misure HTTP
finali disabilitano entrambe le opzioni. Startup e preload denso sono
esclusi dal tempo prefill del benchmark. Un solo processo modello è attivo.
Gli artefatti grezzi restano in
`performance-results/flash-prefill-latency-20261004/`; il
[record compatto](qwen38-flash-prefill-latency-20261004.json) conserva i gate.

## Perché la GPU attende

Sul prompt HTTP profilato, un solo prefill richiede **185,241 s**.
Gli stadi MoE assorbono 179,421 s, il **97,4%** del tempo attribuito agli
stadi. Gated DeltaNet usa 2,372 s e attention 1,425 s. Quindi né PLE
né attention sono il primo bersaglio di questa diagnosi.

I contatori dell'intera richiesta, inclusi i 32 token di decode,
registrano 80296734720 byte letti da SSD, 247130565120 byte H2D,
251953 demozioni GPU→RAM e 366926 espulsioni RAM. Le letture sincrone
occupano 41,676 s e la selezione delle vittime 19,097 s di tempo host.
Questi costi sono già contenuti nello stadio MoE e possono sovrapporsi:
non si sommano ai 185 s. Una GPU poco occupata non dimostra da sola un
kernel lento: qui attende soprattutto il rifornimento degli esperti.

Windows rileva `CurrentLinkSpeed=4`, `CurrentLinkWidth=2`,
`MaxLinkSpeed=4`, `MaxLinkWidth=16`; NVIDIA mantiene x2 durante il carico.
Un microbenchmark CUDA con buffer **pinned**, blocchi di 256 MiB,
due warm-up e cinque campioni misura H2D 2,7296–2,7462 GiB/s e
D2H 2,7826–2,7868 GiB/s. È una verifica della banda effettiva,
non soltanto un'interpretazione dei contatori WSL.

La motherboard rilevata è MSI MAG Z790 TOMAHAWK MAX WIFI (MS-7E25).
La [specifica MSI](https://us.msi.com/Motherboard/MAG-Z790-TOMAHAWK-MAX-WIFI/Specification)
prevede PCI_E1 CPU x16, PCI_E3 chipset x4; un SSD in M2_1 porta PCI_E1
a x8. Questa condivisione documentata non spiega x2. La topologia Windows
riporta CPU PEG, ma non prova montaggio, contatti o assenza di riser.
La causa fisica della negoziazione resta **NOT_VERIFIED**. Va verificata
prima di spendere tempo su ottimizzazioni dei soli kernel o benchmark lunghi.

## Modifica mantenuta

`expert_ram_slot` ora usa il budget derivato dalla memoria disponibile,
con la stessa riserva di almeno 6 GiB e almeno il 25% della memoria rilevata.
Il tetto automatico di 16 GiB rimane solo il fallback quando non si può
leggere la memoria host. L'allocazione pinned resta lazily in slab da
64 MiB: il budget è un massimo, non un'allocazione immediata. Padding e
tail restano conteggiati. Gli override numerici, incluso zero, continuano
ad essere espliciti. Il budget si congela al primo utilizzo, come prima;
non sostituisce il controllo della pressione di memoria di altri processi.

Su questa macchina `auto` risulta circa **32,64 GiB**. La modifica riguarda
solo la cache CUDA `qwen4exp`; non cambia router, esperti selezionati,
pesi, precisione KV, ordine delle operazioni, payload dei checkpoint,
contesto 128K o abilitazione MTP. Qwen27 conserva il percorso `qwen35`.

Nel confronto diagnostico con lo stesso binario e override RAM 32 GiB,
il prefill HTTP profilato passa da 185,241 a **130,177 s**; le letture
SSD dell'intera richiesta scendono da circa 74,78 a 32,71 GiB e le
demozioni da 251953 a 71635. Sono due campioni profilati: la misura
finale senza profiler è registrata separatamente nel JSON.

Il confronto HTTP **senza profiler**, con il nuovo binario `auto`, misura
**184.984 → 126.103 s** di prefill (31.8% di tempo in meno) e
191,499 → 133,472 s per la risposta limitata a 32 token. Entrambi eseguono
un solo prefill; il testo generato coincide. È un campione cold per variante,
non una distribuzione di cinque run HTTP. Il monitor del nuovo processo
non rileva swap e conserva almeno circa
8.77 GiB di memoria disponibile;
questo dato riguarda il prompt breve riprodotto, non checkpoint da 128K token.

## Benchmark e correttezza

Flash: `speed-bench/promessi_sposi.txt`, ctx allocato 131072, chunk 256,
frontiere 256/512/768/1024, zero decode. Ogni frontiera aggiunge **256 token
nuovi** al prefisso; non rappresenta un nuovo prefill dell'intera lunghezza.
Due sweep di warm-up, cinque misurati, nessun profiler. Vision e MTP non
sono caricati da `ds4-bench`; il caso HTTP sopra li include entrambi.

| Frontiera | Baseline, mediana TPS | Auto RAM, mediana TPS | Rapporto |
| --- | ---: | ---: | ---: |
| 256 | 64,44 | 119,29 | 1,85× |
| 512 | 81,02 | 146,70 | 1,81× |
| 768 | 66,88 | 126,46 | 1,89× |
| 1024 | 66,28 | 125,66 | 1,90× |

CV inferiore al 2% in tutti questi gruppi. Il primo chunk freddo rimane
simile (18,69 contro 18,81 TPS): la cache più grande evita soprattutto
il ricambio successivo, non la prima lettura obbligatoria degli esperti.
Nel benchmark Flash `auto` alloca circa 19,56 GiB effettivi; il limite di
32,64 GiB non implica che il prompt breve li consumi tutti.

- PASS: build C/CUDA e test del budget su host grandi/piccoli, overflow e zero.
- PASS: primitive CUDA, vincoli di pinning, round-trip metadata e read failure.
- PASS: 200 posizioni × 248320 logit, 198656000 byte per variante,
  identici byte per byte alla baseline.
- PASS: logit completi alle quattro frontiere dei benchmark Flash, bit-exact.
- PASS: ripristino sistema/skill, rifiuto della corruzione e prefill senza
  replay per Flash e Qwen27, tramite `test_server_resident_frontier`.

Qwen27: ctx allocato 2048, chunk 256, frontiere 256/512, 16 token di decode
teacher-forced, due warm-up e cinque sweep misurati. Mediane prefill
29,70 → 29,62 e 382,94 → 382,19 TPS; decode 28,37 → 28,35 e
27,82 → 27,94 TPS. Le differenze sono inferiori allo 0,5% e compatibili
con il rumore misurato. I quattro dump completi, inclusi i logit dopo decode,
sono bit-exact. Il benchmark riporta il proprio fallback di snapshot alla
prima frontiera e ripete il prefisso: questo costo compare in entrambe
le varianti e non va scambiato per la velocità del solo kernel Qwen27.

Sono inoltre passati i 199 test Python unitari in `tests/test_*.py`
(41 skip e 188 subtest) e i 10 controlli della wiki. La raccolta indiscriminata
`pytest tests` include il vecchio harness terminale `ds4_agent_terminal_test.py`,
che richiede `pyte` non installato in questo ambiente: quel percorso opzionale
rimane NOT_VERIFIED. Nessuna dipendenza del frontend è stata cambiata
per questa verifica CUDA.

Una prova precedente che rendeva espellibili le copie RAM degli esperti
presenti in VRAM è stata **REJECT**: i logit coincidevano, ma la suite
direction non dava un guadagno e alcuni casi caldi peggioravano.
Il codice e i binari di quella prova non sono il risultato mantenuto.

## Limiti e prossime verifiche

Il verdetto **KEEP** riguarda il budget RAM e i workload Flash misurati;
non qualifica tutte le ottimizzazioni precedenti o tutte le lunghezze.
Il PCIe x2 limita ancora il prefill cold e i trasferimenti H2D rimangono
numerosi. Dopo aver verificato il link fisico, misurare di nuovo la banda
e il prompt HTTP, prima di valutare letture SSD concorrenti, demozioni
meno frequenti e indicizzazione delle vittime per capacità. Le proposte
richiedono una nuova baseline; nessun cambiamento al calcolo è implicito.

Prefill reali di decine di migliaia di token, pressione RAM da processi
contemporanei, Metal, ROCm, distributed e altri computer sono
**NOT_VERIFIED** in questa sessione. Il contesto allocato 128K non è un
benchmark di 128K token elaborati. Questa modifica usa solo la memoria
rilevata e conserva i fallback; non costituisce una qualifica su tali host.
