"""Run independent modules concurrently, then run exclusive tests alone."""

from __future__ import annotations

import os

import pytest
from xdist.remote import Producer
from xdist.scheduler import LoadFileScheduling
from xdist.workermanage import WorkerController

from tests.execution_policy import is_live_provider_test

_SERIAL_SUFFIX = "@fe_serial"
_SERIAL_SCOPE = "exclusive tests"


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:
    """Use the logical CPUs available to pytest, unless explicitly overridden."""
    requested = os.getenv("PYTEST_XDIST_AUTO_NUM_WORKERS")
    return int(requested) if requested else (os.process_cpu_count() or 1)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not hasattr(config, "workerinput"):
        return
    for item in items:
        if is_live_provider_test(item) or item.get_closest_marker("serial") is not None:
            # Like xdist's loadgroup marker, expose the group to the controller.
            item._nodeid += _SERIAL_SUFFIX


@pytest.hookimpl(tryfirst=True)
def pytest_xdist_make_scheduler(
    config: pytest.Config, log: Producer
) -> ExclusiveFileScheduling:
    return ExclusiveFileScheduling(config, log)


class ExclusiveFileScheduling(LoadFileScheduling):
    """Reserve a worker for tests that must follow all other worker teardown."""

    def __init__(self, config: pytest.Config, log: Producer | None = None) -> None:
        super().__init__(config, log)
        self._serial_work: dict[str, bool] = {}
        self._serial_node: WorkerController | None = None
        self._exclusive_phase = False

    @property
    def tests_finished(self) -> bool:
        return not self._serial_work and super().tests_finished

    @property
    def has_pending(self) -> bool:
        return bool(self._serial_work) or super().has_pending

    def _split_scope(self, nodeid: str) -> str:
        if nodeid.endswith(_SERIAL_SUFFIX):
            return _SERIAL_SCOPE
        return super()._split_scope(nodeid)

    def add_node(self, node: WorkerController) -> None:
        super().add_node(node)
        if (self._serial_work or self._exclusive_phase) and self._serial_node is None:
            # Replace the exclusive worker before or during its phase.
            self._serial_node = node

    def remove_node(self, node: WorkerController) -> str | None:
        if node is self._serial_node:
            self._serial_node = None
        crashitem = super().remove_node(node)
        self._start_serial_phase_if_ready()
        return crashitem

    def _reschedule(self, node: WorkerController) -> None:
        if self._exclusive_phase and node is not self._serial_node:
            node.shutdown()
            return
        if self._serial_work and node is self._serial_node:
            return
        super()._reschedule(node)

    def _start_serial_phase_if_ready(self) -> None:
        node = self._serial_node
        if not self._serial_work or node not in self.registered_collections:
            return
        if any(worker is not node for worker in self.nodes):
            return

        # Worker removal happens after session/module fixtures have torn down.
        # Requeued work from a crashed worker, if any, precedes exclusive tests.
        self.workqueue[_SERIAL_SCOPE] = self._serial_work
        self._exclusive_phase = True
        self._serial_work = {}
        super()._reschedule(node)
        # xdist needs lookahead or a shutdown sentinel to run the last item.
        super()._reschedule(node)

    def schedule(self) -> None:
        if self.collection is not None:
            super().schedule()
            self._start_serial_phase_if_ready()
            return

        assert self.collection_is_completed
        if not self._check_nodes_have_same_collection():
            return

        collection = list(next(iter(self.registered_collections.values())))
        serial = {item: False for item in collection if item.endswith(_SERIAL_SUFFIX)}
        if not serial or len(self.nodes) == 1:
            super().schedule()
            return

        self.collection = collection
        self._serial_work = serial
        self._serial_node = self.nodes[-1]
        parallel_work: dict[str, dict[str, bool]] = {}
        for nodeid in collection:
            if nodeid not in serial:
                parallel_work.setdefault(self._split_scope(nodeid), {})[nodeid] = False
        work_units = parallel_work.items()
        if self.config.option.loadscopereorder:
            work_units = sorted(work_units, key=lambda unit: -len(unit[1]))
        self.workqueue.update(work_units)

        parallel_nodes = [node for node in self.nodes if node is not self._serial_node]
        for node in parallel_nodes:
            if self.workqueue:
                self._assign_work_unit(node)
            else:
                node.shutdown()
        for node in parallel_nodes:
            self._reschedule(node)
        if not self.workqueue:
            for node in parallel_nodes:
                node.shutdown()
