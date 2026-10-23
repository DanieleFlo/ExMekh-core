#!/usr/bin/env python3
"""Offline synthetic H264/AAC fixture; PyAV is never an inference dependency."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def build(path):
    with av.open(str(path), "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=2)
        video.width, video.height, video.pix_fmt = 48, 48, "yuv420p"
        audio = container.add_stream("aac", rate=16000)
        audio.layout = "mono"
        for index in range(4):
            pixels = np.full((48, 48, 3), index * 60, np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        samples = (6000 * np.sin(2 * np.pi * 440 * np.arange(32000) / 16000)).astype(
            np.int16
        )
        frame = av.AudioFrame.from_ndarray(
            samples[None, :], format="s16", layout="mono"
        )
        frame.sample_rate, frame.pts, frame.time_base = 16000, 0, Fraction(1, 16000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "origin": "DS4 synthetic fixture, no recordings or external content",
                "generator": "tools/build_suite_e2e_video.py",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "video": {
                    "codec": "h264",
                    "frames": 4,
                    "fps": 2,
                    "size": [48, 48],
                    "rgb_levels": [0, 60, 120, 180],
                },
                "audio": {
                    "codec": "aac",
                    "sample_rate": 16000,
                    "channels": 1,
                    "frequency": 440,
                    "samples": 32000,
                    "pcm_amplitude": 6000,
                },
                "pyav": av.__version__,
                "libraries": av.library_versions,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    build(args.output)
