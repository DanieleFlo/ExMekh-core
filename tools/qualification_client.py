"""Client invocation and per-test evidence for native qualification runs."""

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path


def _entrypoint(command) -> tuple[str, int]:
    """Locate the program and its arguments, excluding interpreter options."""
    if not command:
        return "", 0
    if not re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", Path(command[0]).name):
        return command[0], 1
    index = 1
    while index < len(command):
        argument = command[index]
        if argument in ("-c", "-"):
            return "", len(command)
        if argument in ("-m", "--"):
            index += 1
            return (command[index], index + 1) if index < len(command) else ("", index)
        if argument in ("-W", "-X"):
            index += 2
        elif argument.startswith("-"):
            index += 1
        else:
            return argument, index + 1
    return "", index


@dataclass(frozen=True)
class QualificationClient:
    command: tuple[str, ...]
    cwd: Path
    junit: Path | None

    @property
    def is_suite_release(self) -> bool:
        program, _ = _entrypoint(self.command)
        return Path(program).name in (
            "qualify_suite_release.py",
            "tools.qualify_suite_release",
        )

    def _option_arguments(self):
        _, start = _entrypoint(self.command)
        arguments = self.command[start:]
        return arguments[: arguments.index("--")] if "--" in arguments else arguments

    def option(self, name: str) -> str | None:
        """Use argparse's last-occurrence rule for value-taking options."""
        value = None
        arguments = self._option_arguments()
        for index, argument in enumerate(arguments):
            if argument.startswith(name + "="):
                value = argument.split("=", 1)[1]
            elif argument == name:
                if index + 1 == len(arguments) or arguments[index + 1].startswith("-"):
                    raise ValueError(f"{name} requires a value")
                value = arguments[index + 1]
        return value

    @property
    def suite_output(self) -> Path:
        output = self.option("--output")
        if not self.is_suite_release or output is None:
            raise ValueError("release client requires its checkpoint output")
        return (self.cwd / output).resolve()

    def for_suite_server(self, url: str, resume: bool) -> "QualificationClient":
        if not self.is_suite_release:
            return self
        configured = self.option("--url")
        if configured is not None and configured.rstrip("/") != url.rstrip("/"):
            raise ValueError(
                "release client URL must address the owned qualification server"
            )
        command = list(self.command)
        _, start = _entrypoint(command)
        end = command.index("--", start) if "--" in command[start:] else len(command)
        extra = [] if configured is not None else ["--url", url]
        if resume and "--resume" not in self._option_arguments():
            extra.append("--resume")
        return replace(self, command=(*command[:end], *extra, *command[end:]))

    def identity_command(self) -> list[str]:
        """Resume controls recovery, without changing the qualified workload."""
        return [argument for argument in self.command if argument != "--resume"]

    @classmethod
    def prepare(
        cls, command: list[str], cwd: Path, default_junit: Path
    ) -> "QualificationClient":
        """Honor pytest's output path and resolve evidence from the client's cwd."""
        cwd = cwd.resolve()
        program, start = _entrypoint(command)
        if Path(program).name not in ("pytest", "py.test"):
            return cls(tuple(command), cwd, None)
        pytest_index = start - 1
        option_end = (
            command.index("--", pytest_index + 1)
            if "--" in command[pytest_index + 1 :]
            else len(command)
        )
        junit = None
        options = ("--junitxml", "--junit-xml")
        for index in range(pytest_index + 1, option_end):
            argument = command[index]
            option, separator, value = argument.partition("=")
            if option not in options:
                continue
            if not separator:
                value = command[index + 1] if index + 1 < option_end else ""
            if not value or (not separator and value.startswith("-")):
                raise ValueError("pytest JUnit option requires an output path")
            junit = Path(value)
        if junit is None:
            junit = (cwd / default_junit).resolve()
            command = [
                *command[:option_end],
                f"--junitxml={junit}",
                *command[option_end:],
            ]
        return cls(tuple(command), cwd, (cwd / junit).resolve())

    def source_hashes(self) -> dict[str, str]:
        hashes = {}
        for argument in self.command:
            name = argument.split("::", 1)[0]
            path = self.cwd / name
            if path.suffix == ".py" and path.is_file():
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes

    def read_test_results(self) -> list[dict]:
        if self.junit is None or not self.junit.exists():
            return []
        cases = []
        for case in ET.parse(self.junit).iter("testcase"):
            failure, error, skipped = (
                case.find("failure"),
                case.find("error"),
                case.find("skipped"),
            )
            detail = next(
                (item for item in (failure, error, skipped) if item is not None), None
            )
            seconds = float(case.get("time", "0"))
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("invalid JUnit testcase duration")
            cases.append(
                {
                    "class": case.get("classname"),
                    "name": case.get("name"),
                    "seconds": seconds,
                    "status": "FAIL"
                    if failure is not None or error is not None
                    else "NOT_VERIFIED"
                    if skipped is not None
                    else "PASS",
                    "detail": detail.get("message", "") if detail is not None else "",
                }
            )
        return cases

    def record_results(self, report: dict) -> None:
        """Missing, partial or skipped test evidence can never promote a run."""
        report["test_results"] = []
        if self.junit is None:
            return
        report["all_selected_tests_passed"] = False
        try:
            cases = self.read_test_results()
        except (ET.ParseError, OSError, ValueError) as error:
            report["test_results_error"] = repr(error)
            if report["status"] in ("PASS", "PASS_SEGMENTED"):
                report["status"] = "FAIL"
            return
        report["test_results"] = cases
        if not cases:
            report["test_results_error"] = (
                "JUnit report is missing or contains no testcases"
            )
            if report["status"] in ("PASS", "PASS_SEGMENTED"):
                report["status"] = "NOT_VERIFIED"
            return
        statuses = {case["status"] for case in cases}
        report["all_selected_tests_passed"] = statuses == {"PASS"}
        if report["status"] in ("PASS", "PASS_SEGMENTED"):
            if "FAIL" in statuses:
                report["status"] = "FAIL"
            elif "NOT_VERIFIED" in statuses:
                report["status"] = "NOT_VERIFIED"
