# Fixture video/audio per l'accettazione live

`video-with-audio.mp4` contiene quattro frame sintetici uniformi (48×48,
2 fps, livelli RGB 0/60/120/180) e due secondi di sinusoide mono 440 Hz,
16000 Hz, ampiezza PCM 6000. Nessuna registrazione o contenuto esterno.

Codec H264/AAC; hash, byte e versioni delle librerie sono nel JSON accanto.
I test usano il file versionato e ne verificano il checksum, senza importare
PyAV. Per ricreare offline una nuova fixture in una destinazione libera:

```sh
python tools/build_suite_e2e_video.py /new/path/video-with-audio.mp4
```

Solo il generatore richiede PyAV e NumPy. Cambi delle versioni dei codec
possono cambiare i byte; l'aggiornamento della fixture e del JSON è esplicito.
