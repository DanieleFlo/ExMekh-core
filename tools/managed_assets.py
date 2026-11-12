"""Inspect, download and verify the pinned artifacts used by managed Core."""

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import stat
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


@dataclass(frozen=True)
class ArtifactState:
    present: bool
    installed: bool
    missing_bytes: int


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
    def __init__(self, root, token=None, *, local_receipts=None):
        """Verify existing bundles in place when local_receipts is supplied."""
        self.root = root.resolve()
        self._local_only = local_receipts is not None
        self._receipt_root = local_receipts.resolve() if self._local_only else self.root
        self._receipt_root.mkdir(parents=True, exist_ok=True)
        self._token = token

    def _path(self, artifact, suffix=""):
        root = (
            self._receipt_root if suffix in {".lock", ".verified.json"} else self.root
        )
        path = root / (artifact.path + suffix)
        if not path.resolve().is_relative_to(root):
            raise Problem(
                "ARTIFACT_PATH", "The artifact path is outside the model directory."
            )
        return path

    def has_complete_file(self, artifact):
        try:
            metadata = self._regular_file(self._path(artifact))
            return metadata is not None and metadata.st_size == artifact.size
        except (OSError, Problem):
            return False

    def discard_partial(self, artifact):
        self._require_writable()
        with self._claim(artifact):
            self._path(artifact, ".part").unlink(missing_ok=True)

    @staticmethod
    def _fingerprint(metadata):
        return [
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ]

    def _matches_receipt(self, artifact, metadata):
        try:
            receipt = json.loads(self._path(artifact, ".verified.json").read_text())
            return receipt == [artifact.digest, self._fingerprint(metadata)]
        except (OSError, ValueError, Problem):
            return False

    @staticmethod
    def _regular_file(path):
        try:
            metadata = path.stat()
        except FileNotFoundError:
            return None
        return metadata if stat.S_ISREG(metadata.st_mode) else None

    def inspect(self, artifact):
        metadata = self._regular_file(self._path(artifact))
        if metadata is not None and metadata.st_size == artifact.size:
            return ArtifactState(True, self._matches_receipt(artifact, metadata), 0)
        partial = (
            None
            if self._local_only
            else self._regular_file(self._path(artifact, ".part"))
        )
        downloaded = min(partial.st_size, artifact.size) if partial else 0
        return ArtifactState(metadata is not None, False, artifact.size - downloaded)

    def installed(self, artifact):
        try:
            return self.inspect(artifact).installed
        except (OSError, Problem):
            return False

    def remaining(self, artifact):
        return self.inspect(artifact).missing_bytes

    def _verify(self, path, artifact, report, check):
        digest = hashlib.sha256()
        processed = 0
        with path.open("rb") as stream:
            before = self._fingerprint(os.fstat(stream.fileno()))
            if before[2] != artifact.size:
                return None
            while data := stream.read(CHUNK):
                check()
                digest.update(data)
                processed += len(data)
                report("verifying", processed, artifact.size)
            after = self._fingerprint(os.fstat(stream.fileno()))
        return (
            before
            if before == after and digest.hexdigest() == artifact.digest
            else None
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
            return self._ensure_claimed(
                artifact, report, check, import_only or self._local_only
            )

    def _ensure_claimed(self, artifact, report, check, import_only):
        path = self._path(artifact)
        if self.installed(artifact):
            return path
        receipt = self._path(artifact, ".verified.json")
        if path.is_file():
            fingerprint = self._verify(path, artifact, report, check)
            if fingerprint is None:
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
            fingerprint = self._verify(partial, artifact, report, check)
            if fingerprint is None:
                partial.unlink(missing_ok=True)
                raise Problem(
                    "CHECKSUM_MISMATCH",
                    "The downloaded file failed verification. Retry the download.",
                )
            check()
            self._require_unchanged(partial, fingerprint)
            partial.replace(path)
            # Renaming can change ctime while leaving the verified bytes intact.
            fingerprint = self._require_unchanged(path, fingerprint, renamed=True)
        self._require_unchanged(path, fingerprint)
        atomic_json(receipt, [artifact.digest, fingerprint])
        return path

    def _require_unchanged(self, path, fingerprint, *, renamed=False):
        try:
            current = self._fingerprint(path.stat())
        except FileNotFoundError:
            current = []
        fields = 4 if renamed else len(fingerprint)
        if current[:fields] != fingerprint[:fields]:
            raise Problem(
                "ARTIFACT_CHANGED",
                "The file changed during verification. Verify it again.",
                409,
            )
        return current

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
        self.remove_all((artifact,))

    def remove_all(self, artifacts):
        self._require_writable()
        artifacts = tuple(artifacts)
        with contextlib.ExitStack() as locks:
            for artifact in artifacts:
                locks.enter_context(self._claim(artifact))
            for artifact in artifacts:
                for suffix in ("", ".part", ".verified.json"):
                    self._path(artifact, suffix).unlink(missing_ok=True)

    def _require_writable(self):
        if self._local_only:
            raise Problem(
                "ARTIFACT_READ_ONLY",
                "The existing suite directory is read-only to Core.",
            )


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
