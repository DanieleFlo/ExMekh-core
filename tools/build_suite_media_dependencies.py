#!/usr/bin/env python3
"""Build isolated LGPL media dependencies for Linux x86-64 or Apple Silicon.

No system installation and no invocation from inference. Downloads are pinned
and checked before extraction. Building does not qualify model inference;
platform qualification is recorded separately after real media tests.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import tarfile
import urllib.request
from pathlib import Path

FFMPEG_SHA256 = "733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1"
SOXR_SHA256 = "b111c15fdc8c029989330ff559184198c161100a59312f5dc19ddeb9b5a15889"
SOXR_URL = "https://deb.debian.org/debian/pool/main/libs/libsoxr/libsoxr_0.1.3.orig.tar.xz"
PACKAGES = {
    "libsoxr-dev_0.1.3-4build3_amd64.deb": "6fc0cf4cdd5afc009ef06e032cc4f51849820b636605ef6a4724c40d7eb1aa81",
    "libsoxr0_0.1.3-4build3_amd64.deb": "2ddb6d21ad8ae0c850d380b551f109fc6b50c9a9b62c64ce37607c115bd3382a",
}


def checked(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"dependency hash mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("build/suite-media-dependencies")
    )
    parser.add_argument("--prefix", type=Path, default=Path("build/suite-media"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the checked archives/packages already in directory",
    )
    parser.add_argument("--cmake", default="cmake")
    args = parser.parse_args()
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    linux = platform.system() == "Linux" and platform.machine() == "x86_64"
    if not (apple or linux) or args.jobs < 1:
        parser.error("the pinned builder targets Linux x86-64 or Darwin arm64; jobs must be positive")
    directory, prefix = args.directory.resolve(), args.prefix.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prefix.mkdir(parents=True, exist_ok=True)
    archive = directory / "ffmpeg-7.1.1.tar.xz"
    url = "https://ffmpeg.org/releases/ffmpeg-7.1.1.tar.xz"
    if not archive.exists() and not args.offline:
        urllib.request.urlretrieve(url, archive)
    checked(archive, FFMPEG_SHA256)
    source = directory / "ffmpeg-7.1.1"
    if not source.exists():
        with tarfile.open(archive) as stream:
            stream.extractall(directory, filter="data")
    if apple:
        soxr_archive = directory / "libsoxr_0.1.3.orig.tar.xz"
        if not soxr_archive.exists() and not args.offline:
            urllib.request.urlretrieve(SOXR_URL, soxr_archive)
        checked(soxr_archive, SOXR_SHA256)
        soxr_source = directory / "soxr-0.1.3-Source"
        if not soxr_source.exists():
            with tarfile.open(soxr_archive) as stream:
                stream.extractall(directory, filter="data")
        soxr_build = directory / "soxr-build"
        subprocess.run([
            args.cmake, "-S", str(soxr_source), "-B", str(soxr_build),
            f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON", "-DWITH_OPENMP=OFF",
            "-DBUILD_TESTS=OFF", "-DBUILD_EXAMPLES=OFF",
        ], check=True)
        subprocess.run([args.cmake, "--build", str(soxr_build), "--parallel", str(args.jobs)], check=True)
        subprocess.run([args.cmake, "--install", str(soxr_build)], check=True)
    else:
        packages = directory / "packages"
        packages.mkdir(exist_ok=True)
        for name, digest in PACKAGES.items():
            package = packages / name
            if not package.exists() and not args.offline:
                subprocess.run(
                    ["apt-get", "download", name.split("_")[0] + "=0.1.3-4build3"],
                    cwd=packages,
                    check=True,
                )
            checked(package, digest)
            subprocess.run(["dpkg-deb", "-x", str(package), str(prefix)], check=True)
    configure = [
        "./configure",
        f"--prefix={prefix}",
        "--disable-doc",
        "--disable-programs",
        "--disable-avdevice",
        "--disable-avfilter",
        "--disable-network",
        "--disable-autodetect",
        "--disable-x86asm",
        "--disable-everything",
        "--enable-shared",
        "--disable-static",
        "--enable-avformat",
        "--enable-avcodec",
        "--enable-swscale",
        "--enable-swresample",
        "--enable-decoder=h264,hevc,vp8,vp9,ffv1,rawvideo,aac,mp3,mp3float,flac,vorbis,opus,pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le,pcm_f64le",
        "--enable-demuxer=mov,matroska,avi,nut,wav,flac,mp3,ogg,mpegts",
        "--enable-parser=h264,hevc,vp8,vp9,aac,mpegaudio,flac,vorbis,opus",
    ]
    subprocess.run(configure, cwd=source, check=True)
    subprocess.run(["make", f"-j{args.jobs}"], cwd=source, check=True)
    subprocess.run(["make", "install"], cwd=source, check=True)
    (prefix / "provenance.json").write_text(
        json.dumps(
            {
                "ffmpeg": {"url": url, "sha256": FFMPEG_SHA256, "configure": configure},
                "soxr": {"url": SOXR_URL, "sha256": SOXR_SHA256} if apple else PACKAGES,
                "platform": platform.platform(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
