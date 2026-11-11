"""Pinned artifacts and conservative launch profiles for Core's managed mode."""

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 4 * 2**20


class Problem(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class Artifact:
    path: str
    url: str
    size: int
    digest: str


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


class PinnedStore:
    def __init__(self, root, token=None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._token = token

    def _path(self, artifact, suffix=""):
        path = self.root / (artifact.path + suffix)
        if not path.resolve().is_relative_to(self.root):
            raise Problem(
                "ARTIFACT_PATH", "The artifact path is outside the model directory."
            )
        return path

    def has_local_file(self, artifact):
        return self._path(artifact).is_file()

    def discard_partial(self, artifact):
        with self._claim(artifact):
            self._path(artifact, ".part").unlink(missing_ok=True)

    @staticmethod
    def _fingerprint(path):
        stat = path.stat()
        return [
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ]

    def installed(self, artifact):
        try:
            path = self._path(artifact)
            receipt = json.loads(self._path(artifact, ".verified.json").read_text())
            return receipt == [artifact.digest, self._fingerprint(path)]
        except (OSError, ValueError, Problem):
            return False

    def remaining(self, artifact):
        if self.installed(artifact):
            return 0
        path = self._path(artifact)
        if path.is_file() and path.stat().st_size == artifact.size:
            return 0  # The existing file can be verified in place.
        part = self._path(artifact, ".part")
        return (
            artifact.size - min(part.stat().st_size, artifact.size)
            if part.is_file()
            else artifact.size
        )

    def _verify(self, path, artifact, report, check):
        before = self._fingerprint(path)
        if before[2] != artifact.size:
            return False
        digest = hashlib.sha256()
        processed = 0
        with path.open("rb") as stream:
            while data := stream.read(CHUNK):
                check()
                digest.update(data)
                processed += len(data)
                report("verifying", processed, artifact.size)
        return digest.hexdigest() == artifact.digest and before == self._fingerprint(
            path
        )

    @contextlib.contextmanager
    def _claim(self, artifact):
        lock = self._path(artifact, ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Problem(
                    "ARTIFACT_BUSY",
                    "Another Core instance is preparing this file.",
                    409,
                ) from exc
            yield

    def ensure(self, artifact, report, check, *, import_only=False):
        with self._claim(artifact):
            return self._ensure_claimed(artifact, report, check, import_only)

    def _ensure_claimed(self, artifact, report, check, import_only):
        path = self._path(artifact)
        if self.installed(artifact):
            return path
        receipt = self._path(artifact, ".verified.json")
        if path.is_file():
            if not self._verify(path, artifact, report, check):
                raise Problem(
                    "CHECKSUM_MISMATCH",
                    "The existing file failed verification. Remove it and download again.",
                )
        elif import_only:
            raise Problem(
                "ARTIFACT_MISSING",
                "Place the expected file in the server model directory and retry.",
            )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = self._path(artifact, ".part")
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset > artifact.size:
                raise Problem(
                    "CHECKSUM_MISMATCH", "The partial file exceeds the expected size."
                )
            if offset < artifact.size:
                self._download(artifact, partial, offset, report, check)
            if not self._verify(partial, artifact, report, check):
                partial.unlink(missing_ok=True)
                raise Problem(
                    "CHECKSUM_MISMATCH",
                    "The downloaded file failed verification. Retry the download.",
                )
            check()
            partial.replace(path)
        atomic_json(receipt, [artifact.digest, self._fingerprint(path)])
        return path

    def _download(self, artifact, partial, offset, report, check):
        if shutil.disk_usage(self.root).free < artifact.size - offset:
            raise Problem(
                "DISK_SPACE_REQUIRED", "Free space on the model drive and retry."
            )
        headers = {"Accept-Encoding": "identity", "User-Agent": "ExMekh-Core/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        request = urllib.request.Request(artifact.url, headers=headers)
        report("downloading", offset, artifact.size)
        started = time.monotonic()
        processed = offset
        try:
            response = urllib.request.build_opener(SafeRedirect()).open(
                request, timeout=20
            )
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise Problem(
                    "DOWNLOAD_ACCESS",
                    "Grant access to this model on Hugging Face and configure HF_TOKEN.",
                ) from exc
            raise Problem(
                "DOWNLOAD_HTTP", f"The model source responded with HTTP {exc.code}."
            ) from exc
        with response:
            expected_range = f"bytes {offset}-{artifact.size - 1}/{artifact.size}"
            if offset and (
                response.status != 206
                or response.headers.get("Content-Range") != expected_range
            ):
                raise Problem(
                    "RESUME_UNAVAILABLE",
                    "The source cannot resume this partial file. Discard the partial download to restart.",
                )
            if not offset and response.status != 200:
                raise Problem(
                    "DOWNLOAD_HTTP",
                    "The model source did not return a complete download.",
                )
            with partial.open("ab" if offset else "wb") as output:
                while data := response.read(CHUNK):
                    check()
                    if processed + len(data) > artifact.size:
                        raise Problem(
                            "DOWNLOAD_SIZE", "The source exceeded the manifest size."
                        )
                    output.write(data)
                    processed += len(data)
                    rate = (processed - offset) / max(time.monotonic() - started, 0.1)
                    report("downloading", processed, artifact.size, rate)
                output.flush()
                os.fsync(output.fileno())
        if processed != artifact.size:
            raise Problem(
                "DOWNLOAD_INCOMPLETE",
                "The connection ended before the file was complete. Resume the download.",
            )

    def remove(self, artifact):
        with self._claim(artifact):
            for suffix in ("", ".part", ".verified.json"):
                self._path(artifact, suffix).unlink(missing_ok=True)


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new and new.host != req.host:
            new.remove_header("Authorization")
        if new and new.type != "https":
            raise Problem(
                "DOWNLOAD_REDIRECT", "The source redirected to an insecure address."
            )
        return new
