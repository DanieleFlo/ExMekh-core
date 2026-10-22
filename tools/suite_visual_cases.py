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
