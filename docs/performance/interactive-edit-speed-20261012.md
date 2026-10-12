# Decoding dei documenti edit XML — 12 ottobre 2026

Il problema era il costo crescente del masking: ogni candidato ripercorreva la
stringa XML e, per una selezione letterale, ricontrollava il prefisso nello
snapshot. La presenza di un tool interattivo disabilitava inoltre i percorsi
veloci delle stringhe degli altri tool. I primi contatori dopo una ripresa
includevano token già generati prima della pausa, producendo velocità iniziali
artificialmente elevate.

## Responsabilità e percorso predefinito

- Il documento possiede i checkpoint del parser, ai confini di scalar UTF-8,
  entità complete e delimitatori risolti. L'adattatore gestisce gli escape del
  trasporto nativo e traduce gli offset senza ricopiare il prefisso accettato.
- Il vincolo dello snapshot possiede l'indice delle selezioni. Un intervallo
  nell'ordine dei suffissi identifica i candidati ancora compatibili; i probe
  confrontano soltanto i nuovi byte logici con ricerca binaria. L'indice
  esaustivo precedente resta separato come oracolo.
- Il runtime aggiorna checkpoint e frontiera solo per testo accettato. Trie,
  tentativi del vocabolario e prefissi MTP leggono lo stato. Il prefisso
  accettato è immutabile durante la chiamata; la validazione finale riparte
  dall'intero documento.
- Il server compila una maschera dei token ordinari, valida sia nel testo XML
  sia in CDATA. Markup, entità, `]`, UTF-8 frammentato e token di controllo
  restano dinamici. `maxLength` e il budget del transcript impediscono un
  bypass dei limiti. Anche i prefissi verificati di MTP possono riusare la
  maschera senza pubblicare il loro checkpoint.
- I tool ordinari mantengono i propri percorsi veloci quando un altro tool è
  configurato come interattivo. Le verifiche complete di schema avvengono dopo
  aver escluso dal checkpoint le chiusure native premature.
- I log chunk/avg escludono i token ripristinati; `gen` e l'usage API mantengono
  il conteggio cumulativo della risposta.

Restano al client filesystem, workroot, evidenze inspect/read, hash, alias,
intervalli, conflitti, encoding e commit. Questa modifica non cambia il dialetto
XML v2, non reintroduce marcatori o attributi `code` e non impone CDATA. Per
markup il formato attuale ammette entità XML oppure CDATA. Il benchmark HTML
live ha prodotto entità XML, con gli stessi byte del file richiesto.

## Misure e riproduzione

Hardware: RTX 3090, driver 610.88, WSL Ubuntu 24.04. Qwen3.8-27B UD-Q4_K_S,
CUDA, contesto allocato 22593, temperature 0, nessun modello concorrente.
Il confronto client usa un HTML di 294 byte, quattro richieste applicative e
controllo esatto del file. Alla pausa il modello ha già generato 42 token;
il corpo completato porta `gen` a 274: il decoding misurato comprende **232
token nuovi**, non 274. Un warm-up precede cinque ripetizioni.

| Misura, mediana dei cinque campioni | Server precedente | Build definitiva |
| --- | ---: | ---: |
| Decoding dopo la pausa | 2,11 token/s | 24,31 token/s |
| Tempo del decoding, 232 token | 109,97 s | 9,54 s |
| Operazione client completa | 136,79 s | 36,85 s |
| CPU dei vincoli durante il decoding | 97,31 s | 1,46 s |

Il throughput del decoding migliora di **11,5 volte**; quello dell'operazione
completa di **3,7 volte**. Il coefficiente di variazione del throughput è
0,38% prima e 1,39% dopo. Prompt e documento XML risultano identici in tutti
i campioni, oltre ai byte HTML finali. La build è il percorso predefinito,
senza un flag di ottimizzazione da attivare.

Il [report JSON](interactive-edit-speed-20261012.json) conserva campioni, dispersione, profili delle fasi,
hash dei binari, documento XML prodotto e risultati dei test. I primi tentativi
HTTP diretti producevano una sola newline: sono conservati come tentativi
rifiutati e non partecipano al confronto prestazionale.

Il benchmark del parser è separato dalla velocità di inferenza. Confronta il
parser completo e quello incrementale per testo semplice, HTML/CDATA, Unicode,
entità e selezioni letterali, anche con candidati ambigui. Le dimensioni del
prefisso vanno da circa 128 a 16384 byte. Queste misure non sono token/s del
modello e non qualificano ogni possibile testo o lunghezza.
I [360 campioni del parser](interactive-edit-speed-20261012-parser.csv) includono
nove casi, ciascuno con quattro dimensioni e due percorsi. Tutti i gruppi hanno
CV inferiore al 5%. A circa 16 KiB il probe incrementale richiede 105–112 ns
per i corpi ordinari e 1110 ns con 32 selezioni ancora ambigue, contro
106–167 microsecondi del parser completo. Il costo del probe ordinario resta
quasi costante nelle dimensioni misurate; il tempo dell'indicizzazione iniziale
dello snapshot non è incluso nel probe.

```bash
make test-server-interactive ds4_server_test
./ds4_server_test
make tests/bench_interactive_edit
./tests/bench_interactive_edit > /tmp/edit-parser.csv

# Un solo server già attivo; il driver non avvia un altro modello.
exmekh_lib/.venv/bin/python tests/bench_interactive_edit_live.py \
  --base-url http://127.0.0.1:18091/v1 --case html \
  --work-dir /tmp/edit-html-new-run --repeats 5
```

Usare `--case unicode` e `--case old_text` per gli altri casi live. Il driver
conserva messaggi e scambi e rifiuta campioni che non producono i byte attesi.
La maschera diagnostica `DS4_CONSTRAINT_MODE=compare_new_vs_oracle` confronta
il percorso nuovo con l'oracolo esaustivo; i suoi tempi includono quel confronto
e vanno tenuti distinti dalle prestazioni predefinite.
`--numeric-entities` rende espliciti nel prompt gli stessi caratteri Unicode e
CR/LF: è una variante diagnostica della richiesta, non del protocollo o del
masking. I byte attesi restano identici.

## Correttezza e intenti del modello

Passano le fixture native del server e del protocollo, i confronti a ogni
frontiera con l'oracolo completo, AddressSanitizer/UBSan e il lint del driver.
I casi coprono UTF-8 frammentato, entità, CDATA, escape nativi, CR/LF, selezioni
ambigue, comandi, intervalli, edit consecutivi, limiti e contatori dopo ripresa.
La strumentazione sanitizer riguarda il translation unit del server/test;
gli oggetti del motore collegati non sono strumentati.
La suite Python completa esegue **275 test: 214 PASS e 61 SKIP live opt-in**.

La regressione live del client passa **13 casi su 15**: restano i fallimenti
precedenti di delete e svuotamento, entrambi al limite dei 20 turni. I sei casi
MTP finali passano, con **267 confronti delle maschere e zero divergenze**,
7 cicli speculativi e 4 draft accettati. Le selezioni Unicode in `old_text`
producono i byte corretti in entrambe le prove.

Il primo caso `new_text` Unicode sostituisce `🙂` con `😂` e omette il CR
finale. Il documento XML è valido ma non rappresenta i byte richiesti: questo
campione è un fallimento applicativo ed è escluso dai confronti di velocità.
La prova diagnostica con entità numeriche passa entrambe le ripetizioni,
con gli stessi 43 byte attesi, e resta distinta nel report. Il primo
fallimento non viene trasformato in un successo.

## Limiti e prosecuzione sui chunk

Un token che lascia aperta un'entità, uno scalar UTF-8 o una chiusura ambigua
richiede il controllo della coda irrisolta. Code artificialmente lunghissime
possono ancora costare più del testo ordinario. Il parser non restringe i
contenuti per evitare questo caso e non interpreta l'intento dell'utente:
una riscrittura valida può ancora fallire una richiesta di delete o di
svuotamento esatto.
Nelle selezioni molto ambigue il costo dipende anche dal numero di candidati
rimasti, specialmente per decidere se un'entità o un escape possa proseguire.

L'unicità di `old_text` determina la posizione nello snapshot, ma non la
lunghezza della selezione: il modello può scegliere di chiuderla o cambiare
rappresentazione XML. Un chunk dello snapshot può essere una proposta da
verificare, non testo da accettare automaticamente. L'API attuale verifica
draft MTP e non espone ancora draft arbitrari forniti dal chiamante.

L'estensione successiva richiede un'API pubblica del verificatore target per
chunk esterni, logits di ogni posizione, la stessa maschera/sampling per riga,
arresto al primo disaccordo e rollback del suffisso non accettato. Deve fermarsi
alle pause e ai cambi di policy, preservare KV ricorrente, RNG e cancellazione,
e superare i confronti numerici tra verifica a batch e decode. Il normale
prefill, che espone solo i logits finali, non basta a verificare queste scelte.
Questa estensione non è abilitata dalla modifica del masking.

I kernel e le API del motore non sono modificati. Le prove locali qualificano
il server host e il percorso CUDA Qwen27 verificato; Metal, ROCm, SSD e
distribuito non sono qualificati da questi benchmark.
