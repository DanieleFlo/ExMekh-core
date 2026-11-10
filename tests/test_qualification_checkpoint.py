import argparse
import base64
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qualify_metal_server as server
from tools.qualification_checkpoint import (
    ReleaseCheckpoint,
    atomic_json,
    qualification_output,
)
from tools.qualification_client import QualificationClient
from tools.qualification_memory import MemoryPolicy
from tools.qualification_suite import SuiteReleaseHistory
from tools.qualify_suite_release import _run_cycles, run


def identity(bounded=True, cycles=3):
    return {
        "provenance": {"binary": "frozen"},
        "configuration": {"bounded": bounded, "cycles": cycles},
    }


def append(checkpoint, cycle, *, warmup=False, native_bytes=0):
    dimensions = (128, 256, 512, 768)[cycle % 4]
    matrix = [[1.0] + [0.0] * (dimensions - 1)] * 5
    png = f"image-{SuiteReleaseHistory.seed(cycle)}".encode()
    name = checkpoint.save_image(cycle, warmup, png)
    checkpoint.save_vectors(dimensions, matrix)
    owner = {
        "owner": 500,
        "roots": 1,
        "views": 0,
        "allocated": [native_bytes],
        "live": [native_bytes],
        "reserved": [0],
    }
    return checkpoint.append(
        {
            "cycle": cycle,
            "warmup": warmup,
            "status": "PASS",
            "seconds": 1.0,
            "image_seconds": 0.4,
            "embedding_seconds": 0.4,
            "dimensions": dimensions,
            "image_file": name,
            "image_sha256": hashlib.sha256(png).hexdigest(),
            "embedding_sha256": hashlib.sha256(
                json.dumps(matrix, separators=(",", ":")).encode()
            ).hexdigest(),
            "embedding": {"resources": {"owners": [owner]}},
        }
    )


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)

    def test_resume_preserves_completed_artifacts_and_recovers_torn_tail(self):
        with ReleaseCheckpoint(self.output, identity(), "one", False) as checkpoint:
            append(checkpoint, 0)
        image = self.output / "image-000.png"
        before = image.stat().st_mtime_ns
        with (self.output / "cycles.jsonl").open("ab") as out:
            out.write(b'{"cycle":1')
        with ReleaseCheckpoint(self.output, identity(), "two", True) as checkpoint:
            self.assertEqual(checkpoint.completed, 1)
            append(checkpoint, 1)
        self.assertEqual(image.stat().st_mtime_ns, before)
        self.assertEqual(len(list(self.output.glob("interrupted-tail-*"))), 1)
        with ReleaseCheckpoint(self.output, identity(), "three", True) as checkpoint:
            self.assertEqual(checkpoint.completed, 2)

    def test_records_do_not_expose_mutable_journal_state(self):
        contract = identity()
        with ReleaseCheckpoint(self.output, contract, "one", False) as checkpoint:
            contract["configuration"]["cycles"] = 1
            row = append(checkpoint, 0)
            row["warmup"] = True
            row["embedding"]["resources"]["owners"][0]["reserved"][0] = 100
            snapshot = checkpoint.records[0]
            snapshot["warmup"] = True
            self.assertEqual(checkpoint.completed, 1)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                checkpoint.completion()
            self.assertFalse(checkpoint.records[0]["warmup"])
            self.assertEqual(
                checkpoint.records[0]["embedding"]["resources"]["owners"][0][
                    "reserved"
                ],
                [0],
            )

    def test_completed_and_pending_cycles_keep_continuity_per_server(self):
        contract = identity(bounded=False, cycles=50)
        with ReleaseCheckpoint(self.output, contract, "one", False) as checkpoint:
            self.assertEqual(checkpoint.pending_cycles()[0], (0, True))
            append(checkpoint, 0, warmup=True)
            append(checkpoint, 1)
            self.assertEqual(checkpoint.pending_cycles()[0], (2, False))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                checkpoint.completion()
        with ReleaseCheckpoint(self.output, contract, "two", True) as checkpoint:
            self.assertEqual(checkpoint.pending_cycles()[:2], [(0, True), (2, False)])
            append(checkpoint, 0, warmup=True)
            append(checkpoint, 2)
        with (
            ReleaseCheckpoint(self.output, contract, "one", True) as checkpoint,
            self.assertRaisesRegex(ValueError, "not continuous"),
        ):
            append(checkpoint, 3)

    def test_changed_identity_artifact_or_complete_record_is_refused(self):
        with ReleaseCheckpoint(self.output, identity(), "one", False) as checkpoint:
            append(checkpoint, 0)
        for changed in (
            identity(cycles=2),
            {**identity(), "provenance": {"binary": "different"}},
        ):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, "identity"),
                ReleaseCheckpoint(self.output, changed, "two", True),
            ):
                pass
        artifact = self.output / "image-000.png"
        artifact.write_bytes(b"changed")
        with (
            self.assertRaisesRegex(ValueError, "artifact"),
            ReleaseCheckpoint(self.output, identity(), "two", True),
        ):
            pass
        artifact.write_bytes(b"image-1000")
        journal = self.output / "cycles.jsonl"
        row = json.loads(journal.read_text())
        row["dimensions"] = 768
        journal.write_text(json.dumps(row) + "\n")
        with (
            self.assertRaisesRegex(ValueError, "record"),
            ReleaseCheckpoint(self.output, identity(), "two", True),
        ):
            pass

    def test_resume_before_first_checkpoint_and_refuse_unversioned_cycles(self):
        # Includes interruption between the initial journal and manifest writes.
        for empty_journal in (False, True):
            with self.subTest(empty_journal=empty_journal):
                directory = self.output / str(empty_journal)
                directory.mkdir()
                if empty_journal:
                    (directory / "cycles.jsonl").touch()
                with ReleaseCheckpoint(
                    directory, identity(), "new", True
                ) as checkpoint:
                    self.assertEqual(checkpoint.completed, 0)
                    append(checkpoint, 0)
                with ReleaseCheckpoint(
                    directory, identity(), "again", True
                ) as checkpoint:
                    self.assertEqual(checkpoint.completed, 1)
        legacy = self.output / "legacy"
        legacy.mkdir()
        (legacy / "cycles.jsonl").write_text('{"cycle": 0}\n')
        with (
            self.assertRaisesRegex(FileExistsError, "unversioned"),
            ReleaseCheckpoint(legacy, identity(), "new", True),
        ):
            pass

    def test_active_owner_prevents_duplicate_client_and_server(self):
        with (
            ReleaseCheckpoint(self.output, identity(), "one", False),
            self.assertRaisesRegex(RuntimeError, "already running"),
            ReleaseCheckpoint(self.output, identity(), "two", True),
        ):
            pass
        root = self.output / "server"
        with qualification_output(root, False):
            atomic_json(root / "qualification-profile.json", {"identity": identity()})
            with (
                self.assertRaisesRegex(RuntimeError, "already running"),
                qualification_output(root, True),
            ):
                pass
        with qualification_output(root, True) as second:
            self.assertEqual(second, root / "attempts/0001")
        with qualification_output(root, True) as third:
            self.assertEqual(third, root / "attempts/0002")

    def test_gate_failure_cannot_be_retried_as_a_successful_resume(self):
        with ReleaseCheckpoint(self.output, identity(), "one", False) as checkpoint:
            append(checkpoint, 0)
            checkpoint.reject_gate(AssertionError("numerical mismatch"))
        with (
            self.assertRaisesRegex(ValueError, "validation gate failed"),
            ReleaseCheckpoint(self.output, identity(), "two", True),
        ):
            pass

    def test_resting_state_and_continuity_remain_strict_across_segments(self):
        contract = identity(bounded=False, cycles=50)
        with ReleaseCheckpoint(self.output, contract, "one", False) as checkpoint:
            append(checkpoint, 0, warmup=True)
            for cycle in range(1, 6):
                append(checkpoint, cycle)
        with ReleaseCheckpoint(self.output, contract, "two", True) as checkpoint:
            append(checkpoint, 0, warmup=True)
            for cycle in range(6, 11):
                append(checkpoint, cycle, native_bytes=1)
            self.assertEqual(checkpoint.completed, 10)
            self.assertEqual(checkpoint.continuity()["longest_continuous_cycles"], 5)
            with self.assertRaisesRegex(AssertionError, "bounded warm state"):
                SuiteReleaseHistory(checkpoint.records, "8x8").validate_resting_state()


class FakeClient:
    def __init__(self, png, interrupt_after=None):
        self.png, self.interrupt_after, self.generated = png, interrupt_after, []
        model = types.SimpleNamespace(model_dump=lambda **_: {"id": "frozen-chat"})
        self.models = types.SimpleNamespace(
            list=lambda: types.SimpleNamespace(data=[model])
        )
        self.files = types.SimpleNamespace(
            create=lambda **_: types.SimpleNamespace(id="media"),
            content=lambda _: types.SimpleNamespace(read=lambda: png),
        )
        self.images = types.SimpleNamespace(generate=self.generate)
        self.responses = types.SimpleNamespace(create=self.respond)
        self.step, self.code = 0, ""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    @staticmethod
    def metadata():
        return {
            "resources": {
                "owners": [
                    {
                        "owner": 500,
                        "roots": 1,
                        "views": 0,
                        "allocated": [0],
                        "live": [0],
                        "reserved": [0],
                    }
                ]
            }
        }

    def respond(self, **request):
        if self.step == 0:
            self.code = request["input"][0]["content"].split()[1].rstrip(".")
        output = (
            []
            if self.step == 2
            else [
                types.SimpleNamespace(
                    type="function_call",
                    name=("suite_image", "suite_embedding")[self.step],
                    arguments="{}",
                    call_id="call",
                )
            ]
        )
        result = types.SimpleNamespace(
            output=output,
            output_text=self.code,
            usage=types.SimpleNamespace(
                input_tokens_details=types.SimpleNamespace(cached_tokens=1)
            ),
        )
        self.step = (self.step + 1) % 3
        return result

    def generate(self, **request):
        if (
            self.interrupt_after is not None
            and len(self.generated) >= self.interrupt_after
        ):
            raise KeyboardInterrupt("fixture interruption")
        self.generated.append(request["extra_body"]["ds4"]["seed"])
        data = {
            "data": [
                {
                    "b64_json": base64.b64encode(self.png).decode(),
                    "ds4": {"file_id": "image"},
                }
            ],
            "ds4": self.metadata(),
        }
        return types.SimpleNamespace(model_dump=lambda **_: data)

    def post(self, _, **request):
        dimensions = request["body"]["dimensions"]
        return {
            "data": [{"embedding": [1.0] + [0.0] * (dimensions - 1)} for _ in range(5)],
            "ds4": self.metadata(),
        }


class ReleaseResumeTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for relative in (
            "tests/fixtures/native-suite-v1/media/image-0.png",
            "tests/fixtures/native-suite-audio-v2/it.wav",
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"frozen fixture")
        video = self.root / "video.mp4"
        video.write_bytes(b"frozen video")
        self.profile = self.root / "profile.json"
        self.server_instance("one")
        image = io.BytesIO()
        Image.new("RGBA", (8, 8), "red").save(image, format="PNG")
        self.png = image.getvalue()
        self.args = argparse.Namespace(
            bounded=True,
            cycles=3,
            image_size="8x8",
            output=self.root / "client",
            chat_model=None,
            url="http://fixture/v1",
            video=video,
            pid=None,
            resume=False,
        )
        self.sdk = types.SimpleNamespace(OpenAI=lambda **_: FakeClient(self.png))
        self.enterContext(patch.dict("sys.modules", {"openai": self.sdk}))
        self.enterContext(
            patch.dict("os.environ", {"DS4_QUALIFICATION_PROFILE": str(self.profile)})
        )
        self.enterContext(patch("tools.qualify_suite_release.ROOT", self.root))

    def server_instance(self, name):
        atomic_json(
            self.profile, {"identity": {"native": "frozen"}, "server_instance": name}
        )

    def summary(self):
        return json.loads((self.args.output / "summary.json").read_text())

    def test_interrupted_http_sequence_continues_at_next_cycle_without_rewriting(self):
        first = FakeClient(self.png, interrupt_after=1)
        self.sdk.OpenAI = lambda **_: first
        with self.assertRaises(KeyboardInterrupt):
            run(self.args)
        saved = self.args.output / "image-000.png"
        before = saved.stat().st_mtime_ns
        self.assertEqual(self.summary()["completed_cycles"], 1)
        self.args.resume = True
        self.server_instance("two")
        second = FakeClient(self.png)
        self.sdk.OpenAI = lambda **_: second
        report = run(self.args)
        self.assertEqual(report["completed_cycles"], 3)
        self.assertEqual(second.generated, [17, 1002])
        self.assertEqual(saved.stat().st_mtime_ns, before)
        self.assertEqual(report["longest_continuous_cycles"], 2)
        self.assertEqual(report["status"], "PASS")
        third = FakeClient(self.png)
        self.sdk.OpenAI = lambda **_: third
        self.assertEqual(run(self.args)["completed_cycles"], 3)
        self.assertEqual(third.generated, [])

    def test_upload_interruption_records_recoverable_progress(self):
        first = FakeClient(self.png)

        def interrupt(**_):
            raise KeyboardInterrupt("upload interruption")

        first.files.create = interrupt
        self.sdk.OpenAI = lambda **_: first
        with self.assertRaises(KeyboardInterrupt):
            run(self.args)
        self.assertEqual(self.summary()["status"], "INTERRUPTED")
        self.assertEqual(self.summary()["completed_cycles"], 0)
        self.args.resume = True
        self.server_instance("two")
        self.sdk.OpenAI = lambda **_: FakeClient(self.png)
        self.assertEqual(run(self.args)["completed_cycles"], 3)

    def test_interrupted_fixture_generation_does_not_publish_partial_video(self):
        self.args.video = None

        def interrupted(path):
            path.write_bytes(b"partial video")
            raise KeyboardInterrupt("fixture interrupted")

        with patch(
            "tools.qualify_suite_release.fixture_video", side_effect=interrupted
        ), self.assertRaises(KeyboardInterrupt):
            run(self.args)
        fixture = self.args.output / "fixture.mp4"
        self.assertFalse(fixture.exists())
        self.args.resume = True
        with patch(
            "tools.qualify_suite_release.fixture_video",
            side_effect=lambda p: p.write_bytes(b"complete video"),
        ):
            self.assertEqual(run(self.args)["completed_cycles"], 3)
        self.assertEqual(fixture.read_bytes(), b"complete video")

    def test_failed_numeric_gate_stays_failed_on_resume(self):
        bad = FakeClient(self.png)
        original = bad.post

        def invalid_vectors(*args, **kwargs):
            response = original(*args, **kwargs)
            response["data"][0]["embedding"][0] = float("nan")
            return response

        bad.post = invalid_vectors
        self.sdk.OpenAI = lambda **_: bad
        with self.assertRaisesRegex(AssertionError, "finite gate"):
            run(self.args)
        self.assertEqual(self.summary()["status"], "FAIL")
        self.assertTrue((self.args.output / "gate-failure.json").exists())
        self.args.resume = True
        self.sdk.OpenAI = lambda **_: FakeClient(self.png)
        with self.assertRaisesRegex(ValueError, "validation gate failed"):
            run(self.args)


class ServerProvenanceTests(unittest.TestCase):
    def test_resumption_pins_model_assets_and_server_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "tools/qualify_metal_server.py",
                "tools/qualify_suite_release.py",
                "tools/qualification_checkpoint.py",
                "tools/qualification_suite.py",
                "tools/qualification_memory.py",
                "tools/qualification_client.py",
                "tools/qualification_timeouts.py",
                "tools/qualify_metal_command.py",
                "model.gguf",
                "bundle.ds4b",
                "tokens.ds4q",
                "ds4-server-suite",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            configuration = root / "suite.json"
            atomic_json(
                configuration,
                {"models": [{"index": "bundle.ds4b", "text_asset": "tokens.ds4q"}]},
            )
            output = root / "run"
            output.mkdir()
            args = argparse.Namespace(
                output=output,
                suite_config=configuration,
                resume=False,
                mtp=False,
                vision=False,
            )
            invocation = QualificationClient.prepare(
                ["python", "tools/qualify_suite_release.py", "--output", "client"],
                root,
                root / "junit.xml",
            )
            report = {
                "command": [
                    str(root / "ds4-server-suite"),
                    "-m",
                    str(root / "model.gguf"),
                ],
                "client_sources_sha256": invocation.source_hashes(),
                "executables_sha256": {"server": "frozen"},
                "qualification_profile": {"context_tokens": 8192},
                **MemoryPolicy().as_dict(),
                "suite_config_sha256": hashlib.sha256(
                    configuration.read_bytes()
                ).hexdigest(),
                "resource_metrics": True,
                "phase_profile": True,
                "kv_seed": None,
                "footprint_limit_gib": None,
            }
            with (
                patch.object(server, "ROOT", root),
                patch.object(server.platform, "system", return_value="Linux"),
                patch.dict("os.environ", {"DS4_METAL_TENSOR": "1"}),
            ):
                first = json.loads(
                    server.resume_profile(args, report, invocation, output).read_text()
                )
                args.resume = True
                attempt = output / "attempts/0001"
                attempt.mkdir(parents=True)
                continued = QualificationClient.prepare(
                    [*invocation.command, "--resume"], root, root / "junit.xml"
                )
                second = json.loads(
                    server.resume_profile(args, report, continued, attempt).read_text()
                )
                self.assertEqual(first["identity"], second["identity"])
                self.assertNotEqual(first["server_instance"], second["server_instance"])
                self.assertEqual(
                    first["identity"]["runtime_environment"]["DS4_METAL_TENSOR"],
                    hashlib.sha256(b"1").hexdigest(),
                )
                with (
                    patch.dict("os.environ", {"DS4_METAL_TENSOR": "0"}),
                    self.assertRaisesRegex(ValueError, "identity changed"),
                ):
                    server.resume_profile(args, report, continued, attempt)
                with patch.dict(
                    "os.environ", {"DS4_SUITE_E2E_ARTIFACTS": "new-attempt"}
                ):
                    server.resume_profile(args, report, continued, attempt)
                (root / "model.gguf").write_bytes(b"changed model")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    server.resume_profile(args, report, continued, attempt)
                (root / "model.gguf").write_bytes(b"model.gguf")
                (root / "tokens.ds4q").write_bytes(b"changed tokenizer")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    server.resume_profile(args, report, continued, attempt)


class ContinuousGateTests(unittest.TestCase):
    def test_old_completed_cycles_do_not_claim_continuity_in_a_new_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "client"
            output.mkdir()
            video = root / "video.mp4"
            video.write_bytes(b"frozen")
            contract = identity(bounded=False, cycles=50)
            contract["configuration"]["chat_model"] = "frozen-chat"
            with ReleaseCheckpoint(output, contract, "one", False) as checkpoint:
                append(checkpoint, 0, warmup=True)
                for cycle in range(1, 51):
                    append(checkpoint, cycle)
            args = argparse.Namespace(
                bounded=False,
                cycles=50,
                image_size="8x8",
                output=output,
                video=video,
                resume=True,
                pid=None,
            )
            with ReleaseCheckpoint(output, contract, "two", True) as checkpoint:
                client = FakeClient(b"")
                report = _run_cycles(
                    args, client, checkpoint, [video] * 3, "frozen-chat"
                )
                self.assertEqual(client.generated, [])
                self.assertEqual(report["completed_cycles"], 50)
                self.assertEqual(report["status"], "PASS_SEGMENTED")
                self.assertEqual(report["continuous_lifetime"], "NOT_VERIFIED")


if __name__ == "__main__":
    unittest.main()
