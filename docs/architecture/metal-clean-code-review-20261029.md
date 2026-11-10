# Riesame del porting Metal: responsabilità, lifetime e leggibilità

29 ottobre 2026. Riesame delle modifiche rispetto a
`837894c2af5aaf3382dfc0a0750689baee73363e`, prima dei commit. Il perimetro
comprende backend Metal, primitive native, memoria, lifecycle, scheduling
Qwen, server, launcher, build, probe e documentazione collegata. Il
[registro del riesame](../performance/metal-apple-silicon-20261029/review-verification.json)
identifica sorgenti, eseguibili, comandi ed esiti; il
[report iniziale](../performance/metal-apple-silicon-20261029/README.md)
conserva separatamente le evidenze precedenti al refactoring.

## Correzioni e confini

| Problema | Correzione e responsabilità |
| --- | --- |
| `DS4MetalNumeric` era un contenitore di ivar letti e modificati dalle funzioni esterne | Ivar privati; inizializzazione, buffer, binding, pipeline, dispatch, completamento MPS/Metal e budget appartengono alla classe. I callback C traducono l'API numerica, senza accedere alla rappresentazione Objective-C |
| Chiusura Flash espressa passando un mapping nullo alla configurazione | API esplicite `ds4_gpu_qwen4_stream_open` e `ds4_gpu_qwen4_stream_close`; apertura acquisisce il nuovo descriptor prima di rilasciare quello funzionante |
| Pesi densi ed esperti compattati potevano avere stesso offset e stessa dimensione, ma ordine differente | La chiave di cache comprende lo stride esperto; un test confronta selezioni permutate con un oracolo scalare indipendente |
| Il chiamante leggeva direttamente selezioni e numero di esperti dalla cache | Un solo accessor produce insieme binding, selezione rimappata e dominio compatto; il proprietario conserva l'invariante del layout |
| Router non valido poteva lasciare stato parziale o cambiare lo stato del command batch | Validazione completa prima dell'allocazione e del commit della selezione; ripristino del batch anche sul rifiuto. Test di ID negativo e sostituzione con descriptor chiuso |
| Dopo il completamento sincrono del batch, storage e accounting potevano attendere un ulteriore synchronize | `ds4_gpu_end_commands` ritira gli oggetti dopo il completamento riuscito. Il test libera root e view annidate durante il batch e verifica subito storage e output |
| Policy della RAM fisica ripetuta in disponibilità, prenotazione e accounting | Un solo predicato nel registro definisce quali location gravano sull'host, incluse le allocazioni device su memoria unificata |
| Query Mach conservava il diritto al port host | Rilascio del port nella funzione che lo acquisisce, anche su errore della query |
| Valori numerici anonimi descrivevano la direzione delle copie | Costanti interne comuni a CPU, CUDA e Metal; valori e ABI invariati |
| Cap Flash predefinito ripetuto fra ammissione, apertura e restore | Un helper dell'engine definisce il cap in byte per tutti e tre i percorsi |
| Errori del grafo menzionavano CUDA anche su Metal | Testo neutro nel file di risorse del dominio lifecycle |
| Probe con dipendenze transitive mancanti e target Metal esposti su altri sistemi | Dipendenze esplicite dagli header condivisi, input del linker filtrati, target Metal limitati a Darwin e clean coerente |
| Test launcher dipendente dal sistema dell'esecutore e callback `wait` in conflitto con POSIX | CUDA esplicito nelle fixture dei default CUDA; test Metal separati. Callback del transfer rinominato `wait_stage` |
| Helper numerico dei test sceglieva implicitamente Metal anche quando riceveva un booleano chiamato `cuda` | Costruzione con backend esplicito; la fixture del test sceglie la configurazione |
| Nuovo codice compresso e collegamenti a file esterni non presenti nel checkout | Formattazione dei moduli aggiunti e delle zone modificate; riferimenti storici descritti come tali, indici rigenerati |

La semantica e la preparazione delle operazioni restano nel dominio numerico
C. La classe Metal possiede gli oggetti Apple e la loro esecuzione; non
assume responsabilità per tokenizer, shape del modello o HTTP. Gli adapter
image interrogano il workspace tramite `ds4_numeric_linear_workspace`,
senza conoscere i buffer temporanei MPS. Il lifecycle resta nel dominio
engine/sessione, che conosce i layout dei checkpoint; il registro delle
risorse conserva soltanto proprietà, alias, eventi e budget.

## Verifiche e limiti

Le primitive Metal e CPU sono confrontate con PyTorch usando i gate
congelati. Le regressioni coprono cache Flash, layout compatti, errori di
apertura, alias e lifetime, GDN/quantizzazioni UD, mRoPE, checkpoint/restore,
park/release/resume, HTTP e default dei launcher. Le fixture upstream
Metal verificano inoltre gli esperti SSD e la crescita della memoria dei
command buffer. Non è stata eseguita inferenza distribuita del modello.

Il comando completo `make test` è stato eseguito: i prerequisiti e le
fixture passano, ma termina con 15 errori di apertura del modello in
`ds4_test`. Il guard Metal stima 15,91 GiB e trova circa 7,4 GiB disponibili
dopo la riserva. Non sono skip e non sono un PASS della suite completa.
I test indipendenti dal modello di `ds4_test` e tutti i comandi successivi
del target sono stati eseguiti separatamente con esito positivo.

EmbeddingGemma reale è stato rieseguito dopo il refactoring per testo,
immagine, audio, video e input interleaved: vettori bit-exact rispetto al
primo run Metal, cancellazione/riprova e storage finale nullo. Il confronto
con il primo run verifica l'assenza di regressione; non sostituisce gli
oracle indipendenti. I gate BF16 interni già falliti restano aperti.

Anche Qwen Image BF16 è stato rieseguito: generazione ed editing 256×256,
otto passi, stesso seed e input del primo run. Conditioning, latenti, RGBA
e PNG sono bit-exact. In totale il confronto comprende 529 file: 72 Gemma
testo, 434 Gemma media e 23 image. Entrambi i casi image rilasciano tutto
lo storage numerico; picchi 6,66/7,20 GiB entro piani 6,98/7,24 GiB.
Il cache planner ha usato più memoria disponibile rispetto al primo run;
questi picchi e tempi non sono un benchmark del refactoring.

La macchina CUDA non è disponibile, come confermato dall'utente. CUDA,
ROCm, altri chip Apple, inferenza Qwen27/Flash completa e flussi ExMekh
con alternanza del LLM non ricevono una nuova qualifica da questo riesame.
Il modello Flash principale è ancora in download durante le verifiche.

## Debito residuo

I principi di responsabilità singola, incapsulamento, nomi espliciti,
assenza di duplicazioni di policy e coerenza dei fallimenti sono stati
applicati ai confini modificati. Non tutti i principi Clean Code sono
soddisfatti globalmente: `ds4.c` e `ds4_server.c` restano grandi moduli con
stato legacy condiviso; il backend Metal conserva runtime globale e una
cache Flash seriale. Separarli richiede una migrazione con oracle per
famiglia e backend. Una riorganizzazione meccanica non risolverebbe queste
dipendenze e renderebbe più difficile valutare il porting corrente.

Lo streaming Flash è sincrono per layer; overlap I/O/calcolo e riuso fra
layer restano lavoro di prestazioni, con confronti Strata e parity del
modello reale ancora necessari. I gate aperti del
[piano Metal](../roadmaps/metal-apple-silicon-support-20261028.md) mantengono
il loro stato.
