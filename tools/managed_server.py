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
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from managed_assets import ROOT, PinnedStore, Problem, atomic_json
from managed_catalog import ModelCatalog
from managed_hardware import HardwareProbe
from managed_http import ManagedHTTP
from managed_memory import ReplacementMemory
from managed_suite import SuiteAssets
from managed_worker import (
    STARTUP_PHASES,
    NativeLaunch,
    NativeWorker,
    Stopped,
    build_native_server,
)

LOG = logging.getLogger("exmekh.core")
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
MAX_HISTORY = 64
PLAN_TTL_SECONDS = 300
ARTIFACT_REMOVALS = {"remove", "discard"}
ACTIVATIONS = ("apply", "restore")


@dataclass(frozen=True)
class ProfilePlan:
    expires_at: float
    revision: int
    profile: dict
    files: list[dict]

    def is_current(self, revision, now):
        return self.revision == revision and self.expires_at > now

    def requires_more_download(self, files):
        reviewed = {item["name"]: item["missing_bytes"] for item in self.files}
        return any(
            item["missing_bytes"] > reviewed.get(item["name"], 0) for item in files
        )


@dataclass
class PreparedProfile:
    profile: dict
    model_path: Path
    feature_errors: dict[str, str]
    vision_path: Path | None = None
    suite: dict[str, dict] = field(default_factory=dict)

    def disable_feature(self, name, reason):
        self.profile["features"].remove(name)
        self.feature_errors[name] = reason
        if name == "vision":
            self.vision_path = None
        self.suite.pop(name, None)


@dataclass(eq=False)
class InferenceRequest:
    connection: socket.socket | None = None
    stopping: bool = False

    def attach(self, connection):
        self.connection = connection
        if self.stopping:
            self.stop()

    def stop(self):
        self.stopping = True
        if self.connection is not None:
            with contextlib.suppress(OSError):
                self.connection.shutdown(socket.SHUT_WR)


class Core:
    def __init__(
        self, state_dir, model_dir, backend, gemma_media=None, *, suite_config=None
    ):
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
        token = os.environ.get("HF_TOKEN")
        self._store = PinnedStore(model_dir, token)
        self._closing = threading.Event()
        self._suite = SuiteAssets(
            self._directory,
            self._store.root,
            token,
            suite_config,
            gemma_media,
            self._closing,
        )
        self._catalog = ModelCatalog(
            self._store,
            self._suite.gemma_media,
            suite_store=self._suite.store,
            native_stores=self._suite.native_stores,
        )
        self._hardware = HardwareProbe(backend, self._store.root)
        self._lock = threading.Condition()
        self._tasks = queue.Queue()
        self._boot_id = uuid.uuid4().hex
        self._revision = 0
        self._active = None
        self._saved = None
        self._worker = NativeWorker()
        self._replacement_memory = None
        self._previous_launch = None
        self._inflight = 0
        self._requests = set()
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
                        self._finish_files(job)
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

    def catalog(self, *, refresh=False):
        hardware, planning = self._planning_snapshot(refresh=refresh)
        models, optional_models = self._catalog.inventory(planning)
        with self._lock:
            for model in models:
                model["removable"] = self._can_remove(model["id"], model["installed"])
            active = self._running_profile()
            for entry in optional_models:
                feature = entry["id"]
                entry["active"] = bool(
                    active and feature in active.get("available_features", ())
                )
                if error := self._feature_errors.get(feature):
                    entry["activation_error"] = error
                entry["removable"] = (
                    entry.get("present", False)
                    and entry.get("managed", False)
                    and not self._suite_model_in_use(feature)
                )
            profile = copy.deepcopy(self._saved or active)
        return {
            "hardware": hardware,
            "models": models,
            "optional_models": optional_models,
            "profile": profile,
        }

    def _planning_snapshot(self, *, refresh=False):
        with self._lock:
            pid = self._worker.pid if self._running_profile() else None
            reclaim = self._replacement_memory
        replacing = pid is not None and reclaim is not None
        if replacing:
            sample = reclaim.sample(self._hardware, self._worker, pid)
        # Pair the worker allocation count with a fresh system-wide sample.
        hardware = self.host(refresh=refresh or replacing)
        planning = copy.deepcopy(hardware)
        if replacing:
            with self._lock:
                if (
                    self._worker.pid == pid
                    and self._replacement_memory is reclaim
                    and self._running_profile()
                ):
                    reclaim.credit(planning, *sample)
        return hardware, planning

    def _can_remove(self, model, installed):
        active = self._running_profile()
        if active and active["model_id"] == model:
            return False
        # A broken, inactive installation must be removable even if a saved
        # profile points at it, so the same file can be downloaded again.
        return not (installed and self._saved and self._saved["model_id"] == model)

    def _suite_model_in_use(self, feature):
        activation = self._jobs.get(self._activation)
        pending = activation["profile"] if activation else None
        return any(
            profile and feature in profile.get("features", ())
            for profile in (self._saved, self._running_profile(), pending)
        )

    def plan(self, payload):
        profile = self._catalog.parse_profile(payload)
        with self._lock:
            revision = self._revision
        _, host = self._planning_snapshot(refresh=True)
        self._catalog.validate(profile, host)
        files = self._catalog.inspect_profile_files(profile)
        missing = sum(item["missing_bytes"] for item in files)
        if missing > host["disk_free_bytes"]:
            raise Problem(
                "DISK_SPACE_REQUIRED",
                "The model drive does not have enough free space.",
            )
        with self._lock:
            if revision != self._revision:
                raise self._stale_plan()
            now = time.monotonic()
            self._plans = {
                key: plan
                for key, plan in self._plans.items()
                if plan.is_current(self._revision, now)
            }
            if len(self._plans) >= MAX_HISTORY:
                del self._plans[next(iter(self._plans))]
            key = uuid.uuid4().hex
            self._plans[key] = ProfilePlan(
                now + PLAN_TTL_SECONDS, self._revision, profile, files
            )
            return {
                "id": key,
                "revision": self._revision,
                "profile": copy.deepcopy(profile),
                "files": copy.deepcopy(files),
                "missing_bytes": missing,
                "expires_in_seconds": PLAN_TTL_SECONDS,
            }

    @staticmethod
    def _stale_plan():
        return Problem(
            "PLAN_STALE", "Review the profile again before applying it.", 409
        )

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
                "active_requests": len(self._requests),
                "stopping_requests": sum(
                    request.stopping for request in self._requests
                ),
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

    @staticmethod
    def _job_file(name, feature, size_bytes, installed):
        return {
            "name": name,
            "feature": feature,
            "size_bytes": size_bytes,
            "state": "verified" if installed else "queued",
        }

    def _activation_request(self, kind, plan_id):
        if self._activation:
            raise Problem("ACTIVATION_BUSY", "A profile is already being applied.", 409)
        if self._requests:
            raise Problem(
                "INFERENCE_BUSY",
                "Wait for active requests to finish or stop them before changing the model.",
                409,
            )
        plan = None
        if kind == "apply":
            plan = self._plans.get(plan_id) if isinstance(plan_id, str) else None
            if not plan or not plan.is_current(self._revision, time.monotonic()):
                raise self._stale_plan()
            profile = plan.profile
        else:
            profile = self._saved
            if profile is None:
                raise Problem("PROFILE_MISSING", "Choose a profile first.")
        targets = {profile["model_id"], *profile["features"]}
        if any(
            old["model_id"] in targets
            and old["state"] not in TERMINAL
            and old["kind"] in ARTIFACT_REMOVALS
            for old in self._jobs.values()
        ):
            raise Problem("MODEL_BUSY", "The model has an operation in progress.", 409)
        if plan is None:
            return profile, []
        try:
            files = self._catalog.inspect_profile_files(profile)
        except (OSError, Problem) as error:
            raise self._stale_plan() from error
        if plan.requires_more_download(files):
            raise self._stale_plan()
        return profile, [
            self._job_file(
                item["name"], item["feature"], item["size_bytes"], item["installed"]
            )
            for item in files
        ]

    def _existing_artifact_job(self, kind, model, suite_removal):
        if kind not in ("download", "import", "remove", "discard") or not (
            self._catalog.has_model(model) or suite_removal
        ):
            raise Problem("JOB_INVALID", "Choose a supported operation and model.")
        for old in self._jobs.values():
            if old["model_id"] != model or old["state"] in TERMINAL:
                continue
            if old["kind"] == kind:
                return old
            if kind in ARTIFACT_REMOVALS or old["kind"] in ARTIFACT_REMOVALS:
                raise Problem(
                    "MODEL_BUSY", "The model has an operation in progress.", 409
                )
        if kind == "remove":
            self._check_removable(model, suite_removal)
        return None

    def _check_removable(self, model, suite_removal):
        if suite_removal:
            if self._suite_model_in_use(model):
                raise Problem(
                    "MODEL_IN_USE",
                    "Disable this feature in the startup profile before removing its files.",
                    409,
                )
            return
        installed = self._store.installed(self._catalog.model_file(model))
        if not self._can_remove(model, installed):
            raise Problem(
                "MODEL_IN_USE", "Switch to another model before removing this one.", 409
            )

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
            suite_removal = kind == "remove" and self._catalog.has_suite_model(model)
            profile = None
            files = []
            if kind in ACTIVATIONS:
                profile, files = self._activation_request(kind, payload.get("plan_id"))
                model = profile["model_id"]
            elif old := self._existing_artifact_job(kind, model, suite_removal):
                return self._job_snapshot(old)
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
            if profile:
                job["model_name"] = self._catalog.model_name(model)
                job["files"] = files
            elif suite_removal:
                job["model_name"] = self._catalog.suite_model_name(model)
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
                self._finish_files(job)
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
                    self._finish_files(job)
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

    @staticmethod
    def _finish_files(job):
        for entry in job.get("files", ()):
            if entry["state"] == "queued":
                entry["state"] = "skipped"
            elif entry["state"] in {"downloading", "verifying"}:
                entry.update(state="interrupted", progress=None)

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
            return job["progress"]

    def _ensure(self, artifact, job, import_only=False, *, store=None):
        with self._lock:
            job["file"] = artifact.path
            entry = next(
                (
                    item
                    for item in job.get("files", ())
                    if item["name"] == artifact.path
                ),
                None,
            )

        def report(phase, *args):
            with self._lock:
                progress = self._progress(job, phase, *args)
                if entry is not None:
                    entry.update(state=phase, progress=progress)

        try:
            path = (store or self._store).ensure(
                artifact, report, lambda: self._check_job(job), import_only=import_only
            )
        except Stopped:
            raise
        except Exception:
            if entry is not None:
                with self._lock:
                    entry.update(state="failed", progress=None)
            raise
        if entry is not None:
            with self._lock:
                entry.update(state="verified", progress=None)
        return path

    def _run_job(self, job):
        kind = job["kind"]
        model = job["model_id"]
        self._check_job(job)
        if kind in ACTIVATIONS:
            self._apply(job, restore=kind == "restore")
            return
        if kind == "remove" and self._catalog.has_suite_model(model):
            self._catalog.remove_suite_model(model)
            return
        artifact = self._catalog.model_file(model)
        if kind in ("download", "import"):
            self._ensure(artifact, job, import_only=kind == "import")
        elif kind == "remove":
            self._store.remove(artifact)
        else:
            self._store.discard_partial(artifact)

    def _apply(self, job, restore):
        self._progress(job, "checking_profile")
        profile = job["profile"]
        _, host = self._planning_snapshot(refresh=True)
        self._catalog.validate({**profile, "features": []}, host)
        prepared = self._prepare_profile(job, profile, host, restore)
        self._activate_profile(job, profile, prepared, host["backend"])

    def _prepare_profile(self, job, profile, host, restore):
        model = profile["model_id"]
        effective = copy.deepcopy(profile)
        optional_errors = {}
        chat_store, chat_files = self._catalog.files_for(model, None)
        sources = {None: (chat_store, chat_files)}
        for feature in profile["features"]:
            try:
                sources[feature] = self._catalog.files_for(model, feature)
            except (Problem, KeyError, OSError, ValueError):
                LOG.exception("Could not read the artifact list for %s", feature)
                optional_errors[feature] = (
                    "Core could not read the file list for this feature."
                )
                effective["features"].remove(feature)
        files = [
            self._job_file(
                artifact.path, feature, artifact.size, store.installed(artifact)
            )
            for feature, (store, artifacts) in sources.items()
            for artifact in artifacts
        ]
        with self._lock:
            job["files"] = files
        model_path = self._ensure(
            chat_files[0], job, import_only=restore, store=chat_store
        )
        prepared = PreparedProfile(effective, model_path, optional_errors)
        for feature in effective["features"][:]:
            try:
                self._catalog.validate({**profile, "features": [feature]}, host)
                store, artifacts = sources[feature]
                if feature == "vision":
                    prepared.vision_path = self._ensure(
                        artifacts[0], job, import_only=restore, store=store
                    )
                else:
                    for artifact in artifacts:
                        self._ensure(artifact, job, import_only=restore, store=store)
                    self._progress(job, "preparing_assets")
                    prepared.suite[feature] = self._suite.prepare(feature, store.root)
            except (Problem, OSError, ValueError, subprocess.SubprocessError) as exc:
                if isinstance(exc, Problem):
                    message = str(exc)
                else:
                    LOG.exception("Could not prepare optional feature %s", feature)
                    message = "Check server logs and retry preparing this feature."
                prepared.disable_feature(feature, message)
                with self._lock:
                    for item in job["files"]:
                        if item["feature"] == feature and item["state"] == "queued":
                            item["state"] = "skipped"
        return prepared

    def _activate_profile(self, job, requested, prepared, backend):
        self._check_job(job)
        self._progress(job, "building_worker")
        try:
            binary = build_native_server(backend, bool(prepared.suite), self._closing)
        except Problem:
            if not prepared.suite:
                raise
            for feature in tuple(prepared.suite):
                prepared.disable_feature(
                    feature,
                    "The suite build failed. Install the native suite dependencies and try again.",
                )
            binary = build_native_server(backend, False, self._closing)

        with self._lock:
            previous_launch = self._previous_launch if self._running_profile() else None
            self._draining = True
            self._progress(job, "draining")
            self._lock.wait_for(lambda: self._inflight == 0 or self._closing.is_set())
        try:
            self._check_job(job)
            self._stop_worker()
            self._revalidate_prepared(prepared)
            launch = self._prepared_launch(binary, backend, prepared)
            try:
                ready = self._start_worker(launch, job, prepared.profile)
            except Problem:
                if not prepared.profile["features"]:
                    raise
                for feature in prepared.profile["features"][:]:
                    prepared.disable_feature(
                        feature,
                        "Core could not start with this feature enabled. Review its requirements.",
                    )
                self._stop_worker()
                self._revalidate_prepared(prepared)
                binary = build_native_server(backend, False, self._closing)
                launch = self._prepared_launch(binary, backend, prepared)
                ready = self._start_worker(launch, job, prepared.profile)
            for feature in prepared.profile["features"]:
                if feature not in ready["available_features"]:
                    prepared.feature_errors[feature] = (
                        "The worker did not report this feature as ready."
                    )
            with self._lock:
                revision = self._revision + 1
                atomic_json(
                    self._directory / "profile.json",
                    {"version": 1, "revision": revision, "profile": requested},
                )
                self._revision = revision
                self._saved = requested
                self._active = ready
                self._previous_launch = (launch, prepared.profile)
                self._feature_errors = prepared.feature_errors
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

    def _revalidate_prepared(self, prepared):
        host = self.host(refresh=True)
        self._catalog.validate({**prepared.profile, "features": []}, host)
        for feature in prepared.profile["features"][:]:
            try:
                self._catalog.validate(
                    {**prepared.profile, "features": [feature]}, host
                )
            except Problem as error:
                prepared.disable_feature(feature, str(error))

    def _prepared_launch(self, binary, backend, prepared):
        suite = (
            self._suite.config(list(prepared.suite.values()))
            if prepared.suite
            else None
        )
        return NativeLaunch.for_profile(
            binary,
            backend,
            prepared.profile,
            prepared.model_path,
            prepared.vision_path,
            suite,
            self._directory / "kv",
        )

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
                job["progress"]["elapsed_seconds"] = int(elapsed)
                if historical and elapsed < historical:
                    job["progress"]["eta_seconds"] = max(1, round(historical - elapsed))
                else:
                    job["progress"]["eta_seconds"] = None

        before = self.host(refresh=True)
        native = self._worker.start(launch, report, lambda: self._check_job(job))
        device_bytes = self._worker.device_memory()
        after = self.host(refresh=True)
        model = profile["model_id"]
        vision = (
            self._catalog.vision_file(model).size
            if "vision" in profile["features"]
            else 0
        )
        with self._lock:
            self._replacement_memory = ReplacementMemory.observed(
                before,
                after,
                profile,
                self._catalog.model_file(model).size,
                vision,
                device_bytes,
            )
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
            self._replacement_memory = None
        self._worker.stop()
        self._hardware.invalidate()

    def is_closing(self):
        return self._closing.is_set()

    def stop_inference(self):
        with self._lock:
            requests = [request for request in self._requests if not request.stopping]
            for request in requests:
                request.stop()
            return {"requested": len(requests)}

    @contextlib.contextmanager
    def open_inference(self, *, processing=False):
        """Keep the current worker alive for the lifetime of this request."""
        with self._lock:
            if not self._running_profile() or self._draining or self.is_closing():
                raise Problem("MODEL_NOT_READY", "The chat model is not ready.", 503)
            self._inflight += 1
            request = InferenceRequest() if processing else None
            if request is not None:
                self._requests.add(request)
        try:
            try:
                connection = self._worker.connect()
            except OSError as exc:
                raise Problem(
                    "MODEL_NOT_READY", "The inference worker is unavailable.", 503
                ) from exc
            with connection:
                if request is not None:
                    with self._lock:
                        request.attach(connection)
                yield connection
        finally:
            with self._lock:
                self._inflight -= 1
                if request is not None:
                    self._requests.remove(request)
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
        default=ROOT / (os.environ.get("DS4_GGUF_DIR") or "gguf"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    gemma_media = os.environ.get("DS4_GEMMA_MEDIA_ASSET")
    core = Core(
        args.state_dir,
        args.model_dir,
        args.backend,
        Path(gemma_media).resolve() if gemma_media else None,
        suite_config=os.environ.get("DS4_SUITE_CONFIG")
        or "config/native-suite.example.json",
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
