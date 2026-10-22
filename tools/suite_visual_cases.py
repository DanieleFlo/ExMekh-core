"""Shared development fixture serialization; never imported by serving code."""

import struct
from pathlib import Path


def frames(media):
    import numpy as np
    from PIL import Image

    path = Path(media["path"])
    if media["type"] == "image":
        values = np.array(Image.open(path).convert("RGB"))[None]
        times = [0.0]
    else:
        with np.load(path) as data:
            values = data["frames"].copy()
            times = (
                data["timestamps"].tolist()
                if "timestamps" in data
                else list(range(len(values)))
            )
    if "indices" in media:
        idx = media["indices"]
        values, times = values[idx], [times[i] for i in idx]
    return values, times


def write_fixture(path, cases):
    with path.open("wb") as stream:
        stream.write(b"DS4VIS1\0" + struct.pack("<I", len(cases)))
        for case in cases:
            prompt = (case.get("prompt") or "").encode()
            stream.write(
                struct.pack(
                    "<4I",
                    len(prompt),
                    case.get("task", 0),
                    len(case["media"]),
                    case.get("timestamps", False),
                )
            )
            stream.write(prompt)
            for media in case["media"]:
                if "container" in media:
                    data = Path(media["container"]).read_bytes()
                    stream.write(
                        struct.pack("<5I", 4, len(data), 0, 0, media.get("soft", 0))
                    )
                    stream.write(data)
                    continue
                if media["type"] == "audio":
                    wave = audio(media)
                    stream.write(struct.pack("<5I", 3, len(wave), 16000, 1, 0))
                    stream.write(wave.astype("<f4").tobytes())
                    continue
                pixels, times = frames(media)
                n, h, w, _ = pixels.shape
                stream.write(
                    struct.pack(
                        "<5I",
                        1 if media["type"] == "image" else 2,
                        n,
                        w,
                        h,
                        media.get("soft", 0),
                    )
                )
                for image, timestamp in zip(pixels, times):
                    stream.write(struct.pack("<d", timestamp) + image.tobytes())


def audio(media):
    import wave

    import numpy as np

    with wave.open(str(media["path"]), "rb") as f:
        assert f.getsampwidth() == 2
        values = np.frombuffer(f.readframes(f.getnframes()), "<i2").astype(np.float32)
        values = values.reshape(-1, f.getnchannels()).mean(-1) / 32768
        rate = f.getframerate()
    if rate != 16000:
        import soxr

        values = soxr.resample(values, rate, 16000, quality="HQ")
    start = round(media.get("start", 0) * 16000)
    end = round(media["end"] * 16000) if "end" in media else len(values)
    return values[start:end].copy()
