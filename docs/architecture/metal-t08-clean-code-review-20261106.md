# Riesame di policy e qualificatori T08

Riesame richiesto dall'utente dopo la campagna T08 del 6 novembre, su
`1cae644` più le modifiche registrate in `t08-20261106/no-reserve.json`.
La richiesta autorizza ora commit separati dopo i test. I risultati storici
rimangono associati ai loro binari e alle rispettive configurazioni.

## Responsabilità e interventi

- Il calcolo della capacità della cache deve dipendere dai parametri ricevuti,
  senza leggere implicitamente l'ambiente. La configurazione resta nel
  proprietario della policy Metal; ammissione e log usano lo stesso valore.
- Un solo oggetto immutabile dei qualificatori deve possedere riserva,
  soglia di swap e pressione, produrre la configurazione del processo figlio
  e il report, e valutare i campioni. Campionamento OS e lifecycle restano
  nei rispettivi proprietari.
- Il messaggio di rifiuto deve essere valido anche con riserva zero.
- Il probe di ripresa deve isolare acquisizione dei logits, confronto e
  scrittura del risultato, verificando anche gli errori di I/O.
- Le fixture di sola ammissione devono restare distinte dai noise numerici
  congelati. Il qualificatore deve esprimere i dispositivi dell'oracle per
  componente senza allargare le soglie.

Non si cambiano kernel, ordine delle operazioni, forme, sampling, pesi,
algoritmo di budget o soglie. Sono refactoring dei confini e correzioni della
diagnostica, senza nuova rivendicazione di prestazioni.

## Confronto prima del refactoring

Riletti policy, ammissione/rebalance e wrapper in `ds4_metal_memory.h`,
`ds4.c`, `ds4_metal.m`, `ds4_numeric_metal.m`, suite e factory dei probe;
i report Q-P25, I-P04/I-P05 e la campagna T08 conservano l'audit numerico.
I pin pertinenti sono Transformers
`ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
`src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`;
Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`,
`src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`;
Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`,
`src/diffusers/models/transformers/transformer_qwenimage21.py` e
`src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`.
Nessun intervento nei rispettivi contratti di dtype, maschere o stato.

Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
`metal/src/engine/arena_budget.h`, `stream.cpp` e `stream.h`, separa calcolo
di capacità, ownership e trasporto. llama.cpp
`dde879bc4f954b809892fd0d660515f42536394a`, `src/llama-mmap.cpp` e i wrapper,
kernel e test Metal del manifest I-P05, mantiene distinti consigli di mmap
e disponibilità effettiva. Non si importano layout, percentuali, riduzioni
o criteri di equivalenza diversi. I file e gli hash già acquisiti sono nei
manifest delle campagne; questo riesame non usa tali confronti statici come
una misura di speedup o come spiegazione dei casi F08 ancora aperti.

## Verifica prevista

Policy e parser su CPU, report/cleanup del controller, fixture/oracle,
build e fondazioni Metal residenti/SSD. Gate reali brevi Flash poi Qwen27,
pipeline Image indipendente e Gemma in sequenza con riserva zero.
Il probe diagnostico mantiene un controllo congelato di cache da 2 GiB.
Le 50 alternanze Flash e il limite di swap Qwen rimangono risultati storici;
un refactoring equivalente non richiede di ripetere la campagna di tre ore.
I dettagli della nuova verifica saranno in un manifest separato.

## Esiti del riesame

Gli 86 test Python mirati passano in 14,761 s. Build del prodotto e
compilazione CPU isolata, fondazioni Metal residenti e SSD, lifecycle,
factory/planner/HTTP e controlli ASan/UBSan della policy passano.
Flash e Qwen27 conservano identici tutti i dieci artefatti numerici e di
stato dei gate 32+8 precedenti. Image512 supera lo stesso gate indipendente
CPU encoder/MPS DiT/MPS VAE e conserva identici gli undici artefatti nativi.
Gemma mantiene 480/480 frontiere esatte e vettori entro le soglie.
Il probe di ripresa conserva identici i tre vettori completi e il JSON
del controllo precedente; il restore è esatto e la differenza di
segmentazione rimane riprodotta, senza promuovere F08.

Un primo avvio API e il test della porta occupata sono impediti dal
sandbox prima di poter aprire un socket locale. La ripetizione autorizzata
fuori dal sandbox conserva le prove separate, senza classificare quel
rifiuto come errore di inferenza.
La Quick Chat reale passa con risposta `CONFERMATO`, contenuto restituito
identico allo streaming, riserva zero e cache automatica. Il controller
termina in 148,176 s, server exit 0 e nessun cleanup forzato.

Il report delle fondazioni ora dichiara la riserva configurata, inclusa zero;
il test relativo legge il documento attraverso il formato pubblico. La
policy C riceve la riserva esplicita, mentre il controller Python usa gli
stessi metodi pubblici della propria policy per valutazione, ambiente e
report. I test non dipendono dal layout privato di un backend.

La matrice T08 non viene promossa: F08, il limite di risorse nello stress
Qwen, gli asset mancanti e l'hardware non provato restano distinti. Restano
anche le due asserzioni sin/cos preesistenti già riprodotte sul controllo;
il riesame non modifica quella numerica. CUDA/ROCm e distribuito non sono
stati eseguiti su questa macchina. I nuovi hash e gli ultimi gate sono
registrati nel manifest T08 `review-verification.json`.

## Integrazione locale

Commit separati per responsabilità: `6bd07b0` per i proprietari della
riserva Metal, `d0f0e3a` per la policy dei qualificatori, `e56f530` per
ammissione e dispositivi dell'oracle Image, `d33d4b6` per il diagnostico
di ripresa. La documentazione e i manifest vengono integrati separatamente.
Il manifest conserva gli hash dei runner effettivamente usati nei gate:
l'ultima normalizzazione dell'ambiente del qualificatore dei comandi è
coperta dagli 86 test con processi reali; non modifica i binari nativi
già verificati né il valore risolto zero delle loro configurazioni.
