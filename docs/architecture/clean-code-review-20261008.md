# Revisione del codice e delle responsabilità — 8 ottobre 2026

## Perimetro

Revisione delle modifiche Flash pendenti e dei confini fra engine, backend,
launcher, strumenti numerici, benchmark, stato, UI e integrazione ExMekh.
Il checkout di partenza è `1239542282ebcc724a94f754c181f530113ea993` con modifiche
locali già presenti. Wheel, lockfile, conversazioni e framework locale sono
conservati. L'esperimento 27B ritirato rimane nell'archivio dei documenti.

## Responsabilità corrette

| Responsabile | Contratto |
| --- | --- |
| Tuning Flash | Possiede scope, reset e selezione delle preferenze; il vocabolario pubblico distingue AUTO, decode, MTP e prefill. Engine, cache e scheduler usano valori nominati, anche per code brevi e scope annidati. |
| CPU Flash | L'header pubblico espone dimensionamento e proiezioni. Dequantizzazione, tabelle e SIMD restano nell'header privato, incluso soltanto dall'implementazione C e dai test delle primitive. |
| Launcher | È la fonte dei default. Benchmark e qualifica leggono `--print-command`, eliminando copie dei controlli che potevano divergere dalla produzione. |
| Metriche numeriche | Un unico comparatore convalida shape, valori finiti e target e calcola KL, loss, argmax e cosine. Le soglie esistenti restano invariate. Dump invalidi falliscono prima dei calcoli. |
| Provider del benchmark | Sampling e directory di cattura appartengono all'istanza. Due provider non condividono configurazione attraverso attributi di classe. Le richieste usano l'API pubblica `ProviderRequest`. |
| Server del benchmark | Possiede avvio, log, monitor e chiusura idempotente. Un fallimento di `Popen` chiude il log; una porta occupata non avvia processi; il teardown drena soltanto il processo posseduto. |
| Confronto A/B | Usa la validazione statistica comune e rifiuta coppie con un numero diverso di turni, evitando confronti troncati da `zip`. |
| Raccoglitore del turno | Espone `can_steer` e `steer`; il registro dei task non conosce il suo oggetto di steering interno. Il wiring del runner conserva le API esistenti. |
| Costruttori delle capability | Costruiscono gerarchie generiche e personalizzate attraverso lo stesso confine. Il runner coordina l'esecuzione, riceve nodi già costruiti e usa il percorso bundled definito da `config.paths`. |

Il test del launcher è allineato al default attuale di 64K, senza modificare
il launcher. Le capacità esplicite dei benchmark storici a 128K restano tali.
L'indice tecnico usa i percorsi attuali dei pacchetti Python.

## Verifiche

Risultati e comandi sono conservati in `performance-results/clean-code-20261008/`.
La revisione mantiene aritmetica, pesi, quantizzazione, precisione, ordine delle
riduzioni, profondità MTP e formato dei checkpoint. La qualifica numerica resta
distinta dalla misura delle prestazioni.

| Verifica | Esito |
| --- | --- |
| Suite completa agente dopo il refactoring della gerarchia, inclusi Playwright mock | 907 passati, 40 esclusi; copertura statement e branch 100% |
| Suite Python principale | 216 passati, 41 esclusi (257 raccolti) |
| Regressioni degli strumenti e launcher, incluse nella suite principale | 17 passati |
| Unit test C locali e gruppi engine statici | 13 binari e 4 gruppi passati |
| Primitive CPU, padding, tutti gli encoding FP16 e oracle FP64 | PASS, anche con AddressSanitizer e UndefinedBehaviorSanitizer |
| Build CPU e CUDA sm_86 dei quattro eseguibili e probe pertinenti | PASS |
| Gate CUDA, stato, rollback, MTP e protezione 27B | 13 gate passati |
| Logits codice/prosa/tool, prompt brevi e agentici, chunk 256 contro 2048 | 136 righe × 248320 valori bit-exact, KL e delta loss zero |
| Stesse righe confrontate con gli artefatti qualificati v74 | Bit-exact |
| Runner reale tramite `agent/start-ds4-server-flash.sh` | 4 turni senza errori, output identici alla v74, prefix reuse e generazioni di codice/prosa |
| Ruff, sintassi shell/Python, whitespace e indici documentali | PASS |

Il confronto numerico nuovo alloca capacità 65536 con frontiere effettive fra
22 e 4626 token; il riferimento v74 alloca 131072 con gli stessi token. Non è
una prova di 64K/128K token realmente utilizzati. La prova del runner alloca
esplicitamente 131072 e parte da 4926 token: circa 141,64 token/s di prefill
completo, 7,58 di decode codice e 8,47 di decode prosa. È un singolo smoke test,
non una nuova qualifica A/B delle prestazioni.

Gli esiti intermedi restano nei log: primo conteggio di coverage invalido per
modifica del sorgente durante il run, test rimasti al vecchio default 128K,
nomi dei gate errati, quoting del runner shell e indici documentali ancora
da rigenerare. Le ripetizioni finali usano sorgenti fermi e gli stessi assert;
nessuna tolleranza, fixture numerica o soglia di coverage è stata allentata.
I test esclusi richiedono opt-in o risorse esterne e non contano come passati.

## Limiti

La revisione applica principi di responsabilità unica, incapsulamento, coerenza
e riduzione delle duplicazioni ai confini esaminati. Non certifica ogni file
della repository. `ds4.c` e `ds4_server.c` restano moduli grandi: una separazione
generale del loader, della serializzazione o dei protocolli richiede gate propri
e non viene mescolata a questi refactoring. Metal, ROCm e distribuito con hardware
reale non sono eseguibili sulla macchina WSL/RTX 3090 di questa revisione.
