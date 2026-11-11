"""Headless Core supervisor for model preparation and activation."""

import argparse
import contextlib
import copy
import fcntl
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from managed_assets import ROOT, PinnedStore, Problem, atomic_json
from managed_catalog import SUITE_IDS, SUITE_MANIFESTS, HardwareProbe, ModelCatalog
from managed_http import ManagedHTTP
from managed_worker import (
    STARTUP_PHASES,
    NativeLaunch,
    NativeWorker,
    Stopped,
    run_command,
)

LOG = logging.getLogger("exmekh.core")
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
MAX_HISTORY = 64


class Core:
    def __init__(self, state_dir, model_dir, backend, gemma_media=None):
        self._directory = state_dir.resolve()
        self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock_file = (self._directory / "instance.lock").open("a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise Problem(
                "CORE_ALREADY_RUNNING",
                "Another Core instance owns this state directory.",
            ) from None
        self._store = PinnedStore(model_dir, os.environ.get("HF_TOKEN"))
        self._gemma_media = gemma_media
        self._catalog = ModelCatalog(self._store, gemma_media)
        self._hardware = HardwareProbe(backend, self._store.root)
        self._lock = threading.Condition()
        self._tasks = queue.Queue()
        self._closing = threading.Event()
        self._boot_id = uuid.uuid4().hex
        self._revision = 0
        self._active = None
        self._saved = None
        self._worker = NativeWorker()
        self._previous_launch = None
        self._inflight = 0
        self._draining = False
        self._activation = None
        self._error = None
        self._feature_errors = {}
        self._jobs = {}
        self._plans = {}
        self._read_state()
        self._thread = threading.Thread(target=self._run_jobs, name="core-management")
        self._thread.start()

    def _read_state(self):
        profile_path = self._directory / "profile.json"
        try:
            data = json.loads(profile_path.read_text())
            if data["version"] != 1:
                raise ValueError("unknown profile format")
            self._saved = self._catalog.parse_profile(data["profile"])
            self._revision = data["revision"]
            if type(self._revision) is not int or self._revision < 0:
                raise ValueError("invalid revision")
        except FileNotFoundError:
            pass
        except (OSError, TypeError, ValueError, KeyError, Problem):
            self._saved = None
            self._revision = 0
            self._error = {
                "code": "PROFILE_INVALID",
                "message": "The saved profile could not be read. Configure a new profile.",
            }
        try:
            data = json.loads((self._directory / "jobs.json").read_text())
            if data["version"] == 1:
                for job in data["jobs"]:
                    if job["state"] not in TERMINAL:
                        job["state"] = "interrupted"
                    self._jobs[job["id"]] = job
        except (OSError, TypeError, ValueError, KeyError):
            LOG.debug("No usable activity history")

    def _write_jobs(self):
        while len(self._jobs) > MAX_HISTORY:
            first = next(
                (key for key, job in self._jobs.items() if job["state"] in TERMINAL),
                None,
            )
            if first is None:
                break
            del self._jobs[first]
        try:
            atomic_json(
                self._directory / "jobs.json",
                {
                    "version": 1,
                    "jobs": [
                        {k: v for k, v in job.items() if k != "_start"}
                        for job in self._jobs.values()
                    ],
                },
            )
        except OSError:
            LOG.exception("Could not persist activity history")

    def host(self, refresh=False):
        return self._hardware.snapshot(refresh=refresh)

    def catalog(self):
        hardware = self.host()
        models = self._catalog.describe(self._planning_host(hardware))
        with self._lock:
            for model in models:
                model["removable"] = self._can_remove(model["id"], model["installed"])
        return {"hardware": hardware, "models": models}

    def _planning_host(self, hardware=None):
        host = self.host() if hardware is None else copy.deepcopy(hardware)
        # A replacement reuses the active worker's allocations after draining it.
        # The current free-memory sample is checked again before launch.
        with self._lock:
            profile = self._running_profile()
            if profile and profile["mode_id"] == "vram" and host["devices"]:
                # The resident model's bytes are reclaimable. VRAM used by other
                # processes is still excluded by the live sample.
                device = host["devices"][0]
                weights = self._catalog.model_file(profile["model_id"]).size
                device["free_bytes"] = min(
                    device["total_bytes"], device["free_bytes"] + weights
                )
        return host

    def _can_remove(self, model, installed):
        active = self._running_profile()
        if active and active["model_id"] == model:
            return False
        # A broken, inactive installation must be removable even if a saved
        # profile points at it, so the same file can be downloaded again.
        return not (installed and self._saved and self._saved["model_id"] == model)

    def plan(self, payload):
        profile = self._catalog.parse_profile(payload)
        host = self._planning_host()
        self._catalog.validate(profile, host)
        files = [
            {
                "name": item.path,
                "missing_bytes": self._store.remaining(item),
                "installed": self._store.installed(item),
            }
            for item in self._catalog.artifacts(profile)
        ]
        missing = sum(item["missing_bytes"] for item in files)
        if missing > host["disk_free_bytes"]:
            raise Problem(
                "DISK_SPACE_REQUIRED",
                "The model drive does not have enough free space.",
            )
        with self._lock:
            self._plans = {
                key: plan
                for key, plan in self._plans.items()
                if plan[0] > time.monotonic()
            }
            if len(self._plans) >= MAX_HISTORY:
                del self._plans[next(iter(self._plans))]
            key = uuid.uuid4().hex
            self._plans[key] = (time.monotonic() + 300, self._revision, profile)
            return {
                "id": key,
                "revision": self._revision,
                "profile": profile,
                "files": files,
                "missing_bytes": missing,
                "expires_in_seconds": 300,
            }

    def _running_profile(self):
        """Observe worker exit under the state lock before publishing readiness."""
        if self._active and not self._worker.is_running():
            self._active = None
            self._error = {
                "code": "WORKER_EXITED",
                "message": "The inference worker exited. Retry the saved profile.",
            }
        return self._active

    def status(self):
        with self._lock:
            self._running_profile()
            activation = self._jobs.get(self._activation)
            state = (
                "ready"
                if self._active and not self._draining
                else "starting"
                if self._activation
                else "error"
                if self._error
                else "empty"
            )
            return {
                "version": 1,
                "boot_id": self._boot_id,
                "revision": self._revision,
                "state": state,
                "chat_ready": state == "ready",
                "active_profile": copy.deepcopy(self._active),
                "saved_profile": copy.deepcopy(self._saved),
                "job_id": self._activation,
                "activation_job": self._job_snapshot(activation)
                if activation
                else None,
                "error": self._error,
                "feature_errors": self._feature_errors.copy(),
            }

    @staticmethod
    def _job_actions(job):
        state = job["state"]
        actions = []
        if state in ("queued", "running") and job["kind"] in ("download", "import"):
            actions.append("pause")
        if state == "paused":
            actions.append("resume")
        if state in ("queued", "running", "paused") and job["phase"] in (
            "queued",
            "downloading",
            "verifying",
        ):
            actions.append("cancel")
        return actions

    @classmethod
    def _job_snapshot(cls, job):
        internal = {
            "request",
            "idempotency",
            "native_load_seconds",
            "loaded_profile",
            "_start",
        }
        snapshot = {
            **{
                key: copy.deepcopy(value)
                for key, value in job.items()
                if key not in internal
            },
            "allowed_actions": cls._job_actions(job),
        }
        if "_start" in job:
            snapshot["elapsed_seconds"] = int(time.monotonic() - job["_start"])
        return snapshot

    def get_jobs(self, key=None):
        with self._lock:
            if key is None:
                return [
                    self._job_snapshot(job) for job in reversed(self._jobs.values())
                ]
            if key not in self._jobs:
                raise Problem("JOB_NOT_FOUND", "The activity could not be found.", 404)
            return self._job_snapshot(self._jobs[key])

    def submit(self, payload, idempotency=None):
        return self._submit(payload, idempotency, persist=True)

    def _submit(self, payload, idempotency=None, *, persist):
        with self._lock:
            for old in self._jobs.values():
                if idempotency and old.get("idempotency") == idempotency:
                    if old["request"] != payload:
                        raise Problem(
                            "IDEMPOTENCY_CONFLICT", "This key is already in use.", 409
                        )
                    return self._job_snapshot(old)
            kind = payload.get("kind")
            model = payload.get("model_id")
            profile = None
            if kind in ("apply", "restore"):
                if self._activation:
                    raise Problem(
                        "ACTIVATION_BUSY", "A profile is already being applied.", 409
                    )
                if kind == "apply":
                    key = payload.get("plan_id")
                    plan = self._plans.get(key) if isinstance(key, str) else None
                    if (
                        not plan
                        or plan[0] <= time.monotonic()
                        or plan[1] != self._revision
                    ):
                        raise Problem(
                            "PLAN_STALE",
                            "Review the profile again before applying it.",
                            409,
                        )
                    profile = plan[2]
                else:
                    profile = self._saved
                    if profile is None:
                        raise Problem("PROFILE_MISSING", "Choose a profile first.")
                model = profile["model_id"]
            elif kind not in (
                "download",
                "import",
                "remove",
                "discard",
            ) or not self._catalog.has_model(model):
                raise Problem("JOB_INVALID", "Choose a supported operation and model.")
            else:
                for old in self._jobs.values():
                    if old["model_id"] == model and old["state"] not in TERMINAL:
                        if old["kind"] == kind:
                            return self._job_snapshot(old)
                        if kind in ("remove", "discard") or old["kind"] in (
                            "remove",
                            "discard",
                        ):
                            raise Problem(
                                "MODEL_BUSY",
                                "The model has an operation in progress.",
                                409,
                            )
            if kind == "remove":
                installed = self._store.installed(self._catalog.model_file(model))
                if not self._can_remove(model, installed):
                    raise Problem(
                        "MODEL_IN_USE",
                        "Switch to another model before removing this one.",
                        409,
                    )
            if sum(job["state"] not in TERMINAL for job in self._jobs.values()) >= 16:
                raise Problem("QUEUE_FULL", "Wait for an activity to finish.", 409)
            job = {
                "id": uuid.uuid4().hex,
                "kind": kind,
                "model_id": model,
                "profile": profile,
                "state": "queued",
                "phase": "queued",
                "created_at": time.time(),
                "elapsed_seconds": 0,
                "progress": None,
                "error": None,
                "request": payload,
                "idempotency": idempotency,
            }
            self._jobs[job["id"]] = job
            if profile:
                self._activation = job["id"]
                self._error = None
            if persist:
                self._write_jobs()
            self._tasks.put(job["id"])
            return self._job_snapshot(job)

    def action(self, key, action):
        with self._lock:
            job = self._jobs.get(key)
            if not job:
                raise Problem("JOB_NOT_FOUND", "The activity could not be found.", 404)
            if action not in self._job_actions(job):
                raise Problem(
                    "JOB_ACTION_UNAVAILABLE",
                    "This action is unavailable at the current phase.",
                    409,
                )
            if action == "pause":
                job["state"] = "pausing" if job["state"] == "running" else "paused"
            elif action == "resume":
                job["state"] = "queued"
                self._tasks.put(key)
            elif job["state"] == "running":
                job["state"] = "cancelling"
            else:
                job["state"] = "cancelled"
                if self._activation == key:
                    self._activation = None
            self._write_jobs()
            return self._job_snapshot(job)

    def _run_jobs(self):
        while (key := self._tasks.get()) is not None:
            if self._closing.is_set():
                return
            with self._lock:
                job = self._jobs.get(key)
                if not job or job["state"] != "queued":
                    continue
                job["state"] = "running"
                job["started_at"] = time.time()
                job["_start"] = time.monotonic()
            error = None
            try:
                self._run_job(job)
            except Stopped:
                pass
            except Problem as exc:
                error = {"code": exc.code, "message": str(exc)}
            except Exception:
                LOG.exception("Core operation failed: %s", key)
                error = {
                    "code": "OPERATION_FAILED",
                    "message": "Core could not finish the operation. Check server logs and retry.",
                }
            finally:
                with self._lock:
                    state = {"pausing": "paused", "cancelling": "cancelled"}.get(
                        job["state"]
                    )
                    job["state"] = state or (
                        "interrupted"
                        if self._closing.is_set()
                        else "failed"
                        if error
                        else "completed"
                    )
                    job["error"] = error
                    if not error and job["state"] == "completed":
                        job["phase"] = "completed"
                    job["elapsed_seconds"] = time.monotonic() - job.pop("_start")
                    if self._activation == key:
                        self._activation = None
                        self._error = error
                    self._write_jobs()

    def _check_job(self, job):
        if self._closing.is_set() or job["state"] in ("pausing", "cancelling"):
            raise Stopped

    def _progress(self, job, phase, done=None, total=None, rate=None, unit="bytes"):
        with self._lock:
            job["phase"] = phase
            job["elapsed_seconds"] = time.monotonic() - job["_start"]
            job["progress"] = {
                "completed": done,
                "total": total,
                "unit": unit,
                "percent": round(100 * done / total, 1) if total else None,
                "eta_seconds": round((total - done) / rate) if total and rate else None,
                "bytes_per_second": round(rate) if rate and unit == "bytes" else None,
            }

    def _ensure(self, artifact, job, import_only=False):
        with self._lock:
            job["file"] = artifact.path
        return self._store.ensure(
            artifact,
            lambda *args: self._progress(job, *args),
            lambda: self._check_job(job),
            import_only=import_only,
        )

    def _run_job(self, job):
        kind = job["kind"]
        artifact = self._catalog.model_file(job["model_id"])
        self._check_job(job)
        if kind in ("download", "import"):
            self._ensure(artifact, job, import_only=kind == "import")
        elif kind == "remove":
            self._store.remove(artifact)
        elif kind == "discard":
            self._store.discard_partial(artifact)
        else:
            self._apply(job, restore=kind == "restore")

    def _prepare_suite(self, feature, job, restore):
        for artifact in self._catalog.files_for_suite(feature):
            self._ensure(artifact, job, import_only=restore)
        self._progress(job, "preparing_assets")
        root = self._store.root / "suite"
        directory = self._directory / feature
        directory.mkdir(mode=0o700, exist_ok=True)
        index = directory / "bundle.ds4b"
        if not index.is_file():
            run_command(
                [
                    sys.executable,
                    str(ROOT / "tools" / "suite_bundle.py"),
                    "build",
                    "--manifest",
                    str(ROOT / "config" / "suite-bundles" / SUITE_MANIFESTS[feature]),
                    "--root",
                    str(root),
                    "--output",
                    str(directory),
                ],
                self._closing,
            )
        if feature == "images":
            asset = directory / "tokenizer.ds4q"
            source = (
                root / "Qwen" / "Qwen-Image-2.1-Turbo" / "processor" / "tokenizer.json"
            )
            compiler = "compile_qwen_image_tokenizer.py"
        else:
            asset = directory / "gemma.ds4g"
            source = root / "google" / "embeddinggemma-2"
            compiler = "compile_embedding_gemma2.py"
        if not asset.is_file():
            run_command(
                [
                    sys.executable,
                    str(ROOT / "tools" / compiler),
                    str(source),
                    str(asset),
                ],
                self._closing,
            )
        result = {
            "model": SUITE_IDS[feature],
            "index": str(index),
            "root": str(root),
            "text_asset": str(asset),
        }
        if feature == "embeddings":
            result["media_asset"] = str(self._gemma_media)
        return result

    def _apply(self, job, restore):
        self._progress(job, "checking_profile")
        profile = job["profile"]
        host = self._planning_host()
        self._catalog.validate({**profile, "features": []}, host)
        model = profile["model_id"]
        model_path = self._ensure(
            self._catalog.model_file(model), job, import_only=restore
        )
        effective = copy.deepcopy(profile)
        optional_errors = {}
        vision = None
        suite = []
        for feature in profile["features"]:
            try:
                self._catalog.validate({**profile, "features": [feature]}, host)
                if feature == "vision":
                    vision = self._ensure(
                        self._catalog.vision_file(model), job, import_only=restore
                    )
                else:
                    suite.append(self._prepare_suite(feature, job, restore))
            except (Problem, OSError, ValueError, subprocess.SubprocessError) as exc:
                if isinstance(exc, Problem):
                    message = str(exc)
                else:
                    LOG.exception("Could not prepare optional feature %s", feature)
                    message = "Check server logs and retry preparing this feature."
                optional_errors[feature] = message
                effective["features"].remove(feature)
        self._check_job(job)
        suite_config = self._suite_config(suite) if suite else None
        self._progress(job, "building_worker")
        try:
            binary = self._build(host["backend"], bool(suite))
        except Problem:
            if not suite:
                raise
            for feature in ("images", "embeddings"):
                if feature in effective["features"]:
                    optional_errors[feature] = (
                        "The suite build failed. Install the native suite dependencies and try again."
                    )
                    effective["features"].remove(feature)
            suite_config = None
            binary = self._build(host["backend"], False)

        with self._lock:
            previous_launch = self._previous_launch if self._running_profile() else None
            self._draining = True
            self._progress(job, "draining")
            self._lock.wait_for(lambda: self._inflight == 0 or self._closing.is_set())
        try:
            self._check_job(job)
            self._stop_worker()
            self._catalog.validate(effective, self.host(refresh=True))
            launch = self._launch_for(
                binary, host["backend"], effective, model_path, vision, suite_config
            )
            try:
                ready = self._start_worker(launch, job, effective)
            except Problem:
                if not effective["features"]:
                    raise
                for feature in effective["features"]:
                    optional_errors[feature] = (
                        "Core could not start with this feature enabled. Review its requirements."
                    )
                effective["features"] = []
                self._stop_worker()
                self._catalog.validate(effective, self.host(refresh=True))
                binary = self._build(host["backend"], False)
                launch = self._launch_for(
                    binary, host["backend"], effective, model_path, None, None
                )
                ready = self._start_worker(launch, job, effective)
            with self._lock:
                revision = self._revision + 1
                atomic_json(
                    self._directory / "profile.json",
                    {"version": 1, "revision": revision, "profile": profile},
                )
                self._revision = revision
                self._saved = profile
                self._active = ready
                self._previous_launch = (launch, effective)
                self._feature_errors = optional_errors
        except Exception:
            self._stop_worker()
            if previous_launch and not self._closing.is_set():
                try:
                    launch, effective = previous_launch
                    ready = self._start_worker(launch, job, effective, recovering=True)
                    with self._lock:
                        self._active = ready
                except Exception:
                    LOG.exception("Could not restore the previous worker")
            raise
        finally:
            with self._lock:
                self._draining = False
                self._lock.notify_all()

    def _build(self, backend, suite):
        configured = os.environ.get(
            "DS4_MANAGED_SUITE_BIN" if suite else "DS4_MANAGED_SERVER_BIN"
        )
        if configured:
            binary = Path(configured).resolve()
            if not binary.is_file():
                raise Problem(
                    "WORKER_MISSING", "The configured native binary was not found."
                )
            return binary
        target = "ds4-server-suite" if suite else "ds4-server"
        if backend == "cpu":
            run_command(["make", "-j2", "cpu"], self._closing)
        else:
            args = ["make", "-j2", target]
            if backend == "cuda":
                args.append("CUDA_ARCH=" + os.environ.get("CUDA_ARCH", "native"))
            run_command(args, self._closing)
        return ROOT / target

    def _suite_config(self, models):
        media = self._directory / "media"
        media.mkdir(mode=0o700, exist_ok=True)
        return {
            "version": 1,
            "media_directory": str(media),
            "storage_bytes": 4 * 2**30,
            "timeout_seconds": 600,
            "image_timeout_seconds": 1200,
            "models": models,
        }

    def _launch_for(self, binary, backend, profile, model, vision, suite):
        argv = [
            str(binary),
            f"--{backend}",
            "--host",
            "127.0.0.1",
            "-m",
            str(model),
            "--ctx",
            str(profile["context_tokens"]),
            "--no-mtp",
            "--prefill-chunk",
            "64",
            "--kv-disk-dir",
            str(self._directory / "kv"),
        ]
        if profile["mode_id"] == "ssd":
            argv.extend(["--ssd-streaming", "--ssd-streaming-cold"])
        elif profile["mode_id"] == "unified":
            argv.append("--warm-weights")
        if vision:
            argv.extend(["--vision", str(vision)])
        return NativeLaunch(tuple(argv), suite)

    def _start_worker(self, launch, job, profile, *, recovering=False):
        milestone_count = len(STARTUP_PHASES)
        self._progress(
            job,
            "rolling_back" if recovering else "opening_model",
            0,
            milestone_count,
            unit="milestones",
        )
        started = time.monotonic()
        with self._lock:
            historical = next(
                (
                    old["native_load_seconds"]
                    for old in reversed(list(self._jobs.values()))
                    if old.get("native_load_seconds")
                    and old.get("loaded_profile") == profile
                ),
                None,
            )

        def report(phase, elapsed):
            with self._lock:
                if phase is not None:
                    step = STARTUP_PHASES.index(phase) + 1
                    self._progress(
                        job,
                        "rolling_back" if recovering else phase,
                        step,
                        milestone_count,
                        unit="milestones",
                    )
                if historical and elapsed < historical:
                    job["progress"]["eta_seconds"] = max(1, round(historical - elapsed))
                else:
                    job["progress"]["eta_seconds"] = None

        native = self._worker.start(launch, report, lambda: self._check_job(job))
        self._hardware.invalidate()
        if not recovering:
            with self._lock:
                job["native_load_seconds"] = time.monotonic() - started
                job["loaded_profile"] = copy.deepcopy(profile)

        configured = profile["features"]
        operations = native.ready_operations
        available = {
            "vision": "vision" in configured,
            "images": {"image_generate", "image_edit"} <= operations,
            "embeddings": "embed" in operations,
        }
        return {
            **profile,
            "api_model": native.name,
            "owned_by": native.owner,
            "available_features": [
                name for name, enabled in available.items() if enabled
            ],
            "runtime_id": uuid.uuid4().hex,
        }

    def _stop_worker(self):
        with self._lock:
            self._active = None
        self._worker.stop()

    def is_closing(self):
        return self._closing.is_set()

    @contextlib.contextmanager
    def open_inference(self):
        """Keep the current worker alive for the lifetime of this request."""
        with self._lock:
            if not self._running_profile() or self._draining or self.is_closing():
                raise Problem("MODEL_NOT_READY", "The chat model is not ready.", 503)
            self._inflight += 1
        try:
            try:
                connection = self._worker.connect()
            except OSError as exc:
                raise Problem(
                    "MODEL_NOT_READY", "The inference worker is unavailable.", 503
                ) from exc
            with connection:
                yield connection
        finally:
            with self._lock:
                self._inflight -= 1
                self._lock.notify_all()

    def restore_saved_profile(self):
        with self._lock:
            if self._saved:
                # Startup restores are recreated on every boot. The worker
                # records their result, so the listener can start serving now.
                self._submit({"kind": "restore"}, persist=False)

    def close(self):
        self._closing.set()
        with self._lock:
            self._lock.notify_all()
        self._tasks.put(None)
        self._thread.join()
        self._stop_worker()
        self._lock_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("DS4_SERVER_PORT", "8080"))
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "metal", "cuda", "cpu"),
        default=os.environ.get("DS4_BACKEND", "auto"),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=os.environ.get("DS4_STATE_DIR", str(Path.home() / ".ds4" / "managed")),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=os.environ.get("DS4_GGUF_DIR", str(ROOT / "gguf")),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    gemma_media = os.environ.get("DS4_GEMMA_MEDIA_ASSET")
    core = Core(
        args.state_dir,
        args.model_dir,
        args.backend,
        Path(gemma_media).resolve() if gemma_media else None,
    )
    try:
        with ManagedHTTP(
            ("127.0.0.1", args.port), core, os.environ.get("DS4_MANAGE_TOKEN")
        ) as server:

            def stop(_signum, _frame):
                threading.Thread(target=server.shutdown).start()

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            # The management listener is bound before any native model work.
            core.restore_saved_profile()
            LOG.info(
                "Core management listening on http://127.0.0.1:%s/v1",
                server.server_port,
            )
            server.serve_forever()
    finally:
        core.close()


if __name__ == "__main__":
    main()
