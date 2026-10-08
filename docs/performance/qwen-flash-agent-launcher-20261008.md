# Qwen Flash: prefill reale dall'agente e dal launcher, 8 ottobre 2026

La qualifica successiva v74 corregge il drift causato dalla geometria delle
proiezioni cuBLAS: chunk 2048/4096 conservano proiezioni dense a gruppi di 256,
mentre gli esperti riusano i trasferimenti sul batch grande. La suite completa
codice/prosa/tool, sia breve sia con oltre 4600 token di istruzioni e schemi
agentici, confronta 272 righe candidate da 248320 logits: bit-identiche,
KL e differenza di loss zero. Dopo i gate di stato/vision e cinque coppie A/B,
il profilo standard è il default del launcher Flash; le tabelle v71 sotto
sono storiche. Non servono flag sperimentali per usare le modalità promosse.
Il contesto predefinito attuale del launcher è 65536; i benchmark di questa
qualifica richiedono esplicitamente capacità 131072 e mantengono quel valore.

Il prefill lento è riprodotto chiamando il runner reale `execute_agent_turn_sync`
di `/agent`, con `DeepAgent`, provider ExMekh installato e protocollo Responses.
Il server viene avviato dal file `agent/start-ds4-server-flash.sh` richiesto
dall'utente. Il prompt iniziale, comprese istruzioni e schemi tool, contiene
4926 token effettivi; la capacità allocata è 131072, non 128K token occupati.
MTP embedded e proiettore vision sono caricati in tutti i confronti. Non sono
benchmark DSpark né chiamate dirette sostitutive del runner agentico.

Il launcher osservato all'inizio impostava tiered cache, budget VRAM/RAM automatici e chunk
256. Non abilita ring, KV streaming, ranking, grafi CUDA Flash o gli hint.
Il binario iniziale corrispondeva alla build v71: compilare le ottimizzazioni
sperimentali non le attiva. I precedenti valori di 200–300 TPS riguardavano
configurazioni e workload differenti; non descrivevano questo avvio.

## Riproduzione e isolamento

L'harness [bench_qwen_flash_agent.py](../../tools/bench_qwen_flash_agent.py)
avvia un solo server proprio su localhost, cattura i log e campiona la memoria
GPU. Usa conversazioni, directory di lavoro e cache disco di prova nel folder
dei risultati. Non legge conversazioni utente e non salva impostazioni frontend.
I due turni iniziali richiedono READY e DONE; i due aggiuntivi producono codice
e prosa per misurare la generazione, evitando di chiamare un singolo token
un benchmark di decode. I salvataggi e il riuso del prefix rimangono reali.

Comando riproducibile, con una sola GPU disponibile e nessun altro server DS4:

```sh
agent/.venv/bin/python tools/bench_qwen_flash_agent.py \
  --output performance-results/flash-strata-20261006/agent-launcher-new-trial \
  --profiles launcher stream-ring-2048 stream-ring-4096 \
  --greedy --decode-prompts
```

`--greedy` mantiene il provider e tutte le richieste reali dell'agente, fissando
soltanto temperatura zero e seed 424242 attraverso l'API pubblica del provider.
La cronologia dei turni viene ricostruita con gli stessi `ChatMessage` usati dal
frontend. `--reasoning xhigh` consente la prova separata del thinking; i risultati
qui identificati come `none` non sono risultati thinking.

## Diagnosi iniziale

Screen `agent-launcher-v71-screen-r2`: un campione per configurazione, con sampling
ordinario del server, stato esperti freddo e file cache non svuotata.

| Configurazione | Prefill medio token/s | Tempo prefill registrato |
| --- | ---: | ---: |
| Launcher originale, chunk 256 | 38,07 | 129,387 s |
| Ring e KV streaming, chunk 256 | 57,63 | 85,481 s |
| Ring e KV streaming, chunk 4096 | 185,51 | 26,554 s |

Il primo blocco originale da 256 token richiede 12,75 s: media 20,08 TPS.
Il primo blocco da 4096 della variante misura 279,26 TPS, ma il prompt intero
185,51: non si sostituisce il picco di un chunk alla media finale. La separazione
del sistema dal messaggio utente introduce anche un ultimo prefill da 42 token;
le copie dei pesi e la preparazione della cache pesano molto su questi tail.

La quota esperti riportata nel runtime passa da 10,33 GiB nell'avvio originale
a circa 11,91 con streaming/chunk 256, oppure 8,73 con streaming/chunk 4096.
Quest'ultimo dedica più memoria ai workspace. Blocchi piccoli pagano molte
volte il caricamento degli esperti; blocchi grandi ammortizzano quel lavoro
su più righe, ma possono lasciare meno cache al decode. La diagnosi non
richiede cambiare pesi, quantizzazione, PLE BF16 o la testa MTP nativa.

I checkpoint dei primi due turni costano circa 0,2–0,4 s. Non spiegano il
prefill iniziale di oltre un minuto. Per risposte lunghe il log mostra anche
replay post-response più costosi: vanno inclusi nella latenza end-to-end,
pur distinguendoli dal prefill del nuovo prompt.

Il primo tentativo `agent-launcher-v71-screen` non ha inviato inferenze per
un errore del harness: mancava la directory di lavoro. È conservato. La
prima iterazione r2 passava dict alla ricostruzione della cronologia e
produceva un warning di restore del node; i fronti server venivano riusati,
ma quei follow-up non qualificano la ricostruzione completa del frontend.
Le prove successive usano i `ChatMessage` e non hanno quel warning.

## Isolamento del chunk e generazioni reali

`agent-launcher-v71-isolate` usa sampling greedy, stesso prompt e quattro turni
per variante. Cambia prima solo il chunk del launcher originale, poi aggiunge
il bundle streaming/ring già presente. Gli output READY, DONE, codice e prosa
sono identici tra le due configurazioni.

| Configurazione | Prefill iniziale TPS | Decode codice TPS | Decode prosa TPS |
| --- | ---: | ---: | ---: |
| Launcher originale con solo chunk 4096 | 133,18 | 4,84 | 6,29 |
| Streaming/ring, chunk 4096 | 185,22 | 7,53 | 12,00 |

Il codice genera 205 token, la prosa 127. I tempi di generazione sono quelli
finali del log `decoding`, distinti dai checkpoint dopo la risposta e dal tempo
totale del turno. Picco GPU 22276 MiB per il solo chunk e 22576 per il bundle.
Le differenze non sono attribuite a una singola ottimizzazione del bundle:
esso include anche policy stable, PCIe 100%, ranking, grafi, KV decode batch
e QSA register keys. Prefill CPU e nuovi esperti NextN residenti restano spenti.

## Launcher originale contro bundle 2048

`agent-launcher-v71-balanced` ripete la configurazione originale con sampling
greedy e cronologia frontend corretta, poi misura il bundle con chunk 2048.
Il modello, la capacità 128K, il proiettore e MTP restano uguali.

| Configurazione | Prefill medio del log TPS | Prompt done, secondi | Decode codice TPS | Decode prosa TPS |
| --- | ---: | ---: | ---: | ---: |
| Launcher originale, chunk 256 | 37,74 | 130,524 | 5,26 | 7,71 |
| Streaming/ring, chunk 2048 | 147,60 | 34,225 | 8,79 | 13,91 |

Il confronto 4096 descritto sopra misura `prompt done` a 27,376 s. I relativi
TPS di prefill includendo anche la preparazione finale della cache sono
37,74 / 143,93 / 179,94 per originale / bundle 2048 / bundle 4096. Sono distinti
dal contatore `avg` mostrato durante i chunk. Nessun dato include lo startup
del processo nei TPS; i report conservano startup e durata di ciascun turno.

Il 2048 conserva meno workspace del 4096 e più quota di esperti. In questo
screen ha il decode migliore, mentre il 4096 conclude prima il prompt lungo.
Il follow-up da 45 token, a prefix riutilizzato, misura solo 22,34 / 40,24 /
33,67 TPS di chunk: i costi fissi dominano e non si applicano i TPS del prompt
lungo a una coda breve. I fronti mostrano `match=visible-prefix` con 4928 token
riusati, quindi il follow-up non viene spacciato per un secondo prefill da 5K.

Cambiare il chunk altera alcune continuazioni greedy: codice 171 token e
prosa 125 nell'originale, 205 e 134 nel bundle 2048. READY/DONE coincidono;
non si dichiara identità di tutti gli output tra chunk diversi, né si deduce
la qualità dalla plausibilità delle risposte. Il confronto a parità di chunk
4096 sopra conserva tutti gli output, ma la qualifica del cambio di geometria
richiede logits completi, KL e loss teacher-forced propri. I valori decode
qui misurati sono di continuazioni diverse, non uno stesso stream teacher-forced.
Picco GPU 22504 MiB nell'originale e 22596 nel bundle 2048; il test usa circa
5K di contesto effettivo, pur allocando capacità 128K.

## Stato della qualifica

Le prove descrivono l'avvio reale e identificano la differenza rispetto al
benchmark precedente. Sono screen diagnostici, non cinque coppie alternate
dopo warm-up. Output identici non sostituiscono i logits completi, KL e loss
teacher-forced. Nessun nuovo default del launcher viene promosso da questi
soli screen. I gate numerici della build v71 sono nel
[report principale](qwen-flash-prefill-strata-20261008.md).

## Correzione numerica e profilo standard v74

Il primo gate v72 a capacità 131072 e prompt da 4758 token trova KL massimo
0,113850 per chunk 2048 e 0,555356 per 4096. Gli argmax dei 28 teacher target
coincidono, ma questo non supera il controllo di distribuzione. Il confronto
che cambia solo il chunk riproduce esattamente il drift; quello che applica
solo il bundle mantenendo 256 righe è bit-identico. I fallimenti sono conservati
in `launcher-quality-v72-screen`, `launcher-quality-v72-plain`; la prova positiva
è `launcher-quality-v72-same-chunk`.

Le proiezioni cuBLAS possono cambiare la riduzione quando cambia il numero di
colonne token. `dense_project` ora usa la geometria qualificata da 256 colonne,
compresa la coda breve con il dispatcher originale, senza cambiare peso,
dequantizzazione, attivazioni, precisione o MTP. Non aggiunge un nuovo switch
semantico: è il percorso normale Flash CUDA. I kernel degli esperti continuano
a usare l'intero batch per ammortizzare i trasferimenti. La primitiva confronta
FP32, FP16 e Q8 su 529 righe contro chiamate indipendenti 256/256/17: bit-identità.
Il tuner usa versione 35 solo per capacità prefill superiore a 256; le preferenze
dei piccoli chunk qualificati conservano la propria chiave.

`launcher-quality-v73-dense-columns` e `launcher-quality-v73-dense-4096`
confermano il primo prompt lungo corretto. La suite finale
`launcher-quality-v74-full/report.json` contiene sei fixture per chunk:
136 righe con 2048 e 136 con 4096, tutte finite e bit-identiche al riferimento
256. KL massimo/medio e delta della loss teacher-forced sono zero. Il prompt
di qualifica deriva dalla cattura ExMekh e viene ritokenizzato dal GGUF DS4;
è una fixture della stessa forma agentica, non si presenta la resa frontend
come il prompt byte per byte del server. I test HTTP restano richieste reali.

Il profilo `DS4_FLASH_PROFILE=standard` del launcher prepara KV streaming con
32768 celle residenti, ring bounded, policy stabile, ranking/grafi CUDA e
KV decode batch/QSA. Mantiene quota PCIe 1, prefill GPU, budget auto e MTP
nativo; non abilita il prefill CPU non qualificato, gli hint anticipati o
la riserva NextN che aveva peggiorato i benchmark. Il chunk previsto è 2048;
gli override espliciti vengono rispettati. `DS4_FLASH_PROFILE=legacy`
ricostruisce il precedente avvio per i confronti.

La build `build-v74-release.log` completa CLI/server/bench/eval e i test CUDA
sm_86. `launcher-v74-gates/qualification.json` registra dodici gate PASS:
primitivi, ranking, tuner, ring, KV decode batch/QSA, compatibilità, stato,
checkpoint/restore, rewind, resampling, limiti/scratch MTP e 27B ordinario con
controlli Flash ereditati. `launcher-v74-vision/report.json` registra otto gate
PASS e venti casi API, con payload multimodali e frontiere a 786/787 token
bit-identiche anche cambiando la capacità prefill da 256 a 2048. Nessuno skip
viene contato come PASS.

Sei test dei launcher passano; verificano il nuovo default, gli override e l'isolamento del
27B, il cui wrapper conserva il contesto 28000. Cinquantacinque test unitari
dell'integrazione runner/history e il test browser di cambio modello/contesto
passano con l'ambiente installato in `/agent`. Un'asserzione iniziale del test
launcher assumeva erroneamente il contesto 22593 del launcher condiviso anziché
28000 del wrapper agente; è stata corretta dopo aver letto il wrapper, senza
cambiare il 27B.

Il primo tentativo di A/B v74 si interrompe prima di avviare la variante per
un controllo porta del harness che non gestiva `TIME_WAIT`. Il controllo usa
ora la stessa opzione di riuso indirizzo del listener; continua a rifiutare
un listener attivo. Il tentativo è conservato in `launcher-v74-paired`.
La ripetizione separata `launcher-v74-paired-r2` conserva una coppia di warm-up
esclusa, cinque coppie alternate, quattro turni reali per variante, memoria
GPU e identità di server/launcher/harness. Nessun profiler o build concorrente.

## Promozione finale e uso

Verdetto **KEEP / profilo standard promosso** per il launcher Flash CUDA sul
GGUF locale e RTX 3090/WSL. Basta `./agent/start-ds4-server-flash.sh`; gli
override espliciti rimangono validi e `DS4_FLASH_PROFILE=legacy` è un controllo
di riproduzione. Il chunk standard 2048 bilancia cache esperti e prefill; 4096
resta un override che ha superato la qualifica numerica, senza essere presentato
come un profilo prestazionale universale. CPU prefill, anticipo speculativo
degli esperti e riserva NextN non sono promossi da questi risultati.

Cinque coppie misurate dopo una coppia di warm-up, stesso binario v74 e
prompt, capacità 131072 e contesto iniziale effettivo 4926. GPU e cache esperti
ripartono per processo, file cache calda, Responses/DeepAgent reali, MTP
nativo, proiettore caricato, temperatura zero e seed 424242. Il codice genera
93 token e la prosa 55; sono output brevi, distinti dai prompt lunghi v71.

| Metrica, mediana | Precedente | Standard | Delta |
| --- | ---: | ---: | ---: |
| Prefill completo, incluso cache finish, token/s | 37,83 | 140,31 | +270,92% |
| Primo token visibile dall'agente, s | 130,65 | 35,48 | −72,84% |
| Follow-up con prefix riusato, s | 2,842 | 2,611 | −8,14% |
| Decode codice, token/s | 5,06 | 7,67 | +51,58% |
| Decode prosa, token/s | 5,33 | 8,92 | +67,35% |

CV prefill 0,68%/0,58%; codice 1,89%/3,54%; prosa 1,09%/2,78%; latenza
follow-up 2,09%/1,93%. Ogni coppia mantiene i quattro output identici, nessun
errore nei 40 turni misurati. Il guadagno supera la variabilità; nessuna metrica
misurata peggiora oltre il 5%. I tempi del primo token comprendono il runner
e la richiesta HTTP, escludendo l'avvio del server; non sono presentati come
misura del solo kernel. Il picco di un chunk non sostituisce il prefill completo.

La qualifica prova l'uso reale di circa 5K token con capacità 128K allocata;
non è un nuovo benchmark a 128K token effettivi. Gli artefatti di identità
della consegna sono in `candidate-v74-promotion`; i launcher effettivamente
usati prima del cambio della sola selezione di default sono conservati
insieme ai campioni. Questo report non promuove hardware, formati o backend
non testati.
