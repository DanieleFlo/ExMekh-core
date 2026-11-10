# ExMekh Lib patches in the bundled wheel

`exmekh-lib-budget.patch` applies to ExMekh Lib commit
`1a94657fd1127375a221349087ce33cae0c9edcc`. It includes implementation and
regression tests. The bundled `agent/wheels/exmekh_lib-0.1.0-py3-none-any.whl`
was rebuilt with `uv build --wheel` from that revision plus this patch.
The wheel metadata and dependency versions are unchanged; `agent/uv.lock`
records the new wheel SHA256:
`97e0119e2b7767bd4af6ec74d39987c73a00b7fe5671d0406bbae5b4c2b76139`.

The node runtime owns automatic history replacement. Before changing history,
scope anchors or emitting an archive event, it checks the complete candidate
request with the caller's token estimator. A budgeted automatic replacement
must be smaller and fit the prompt limit. Otherwise original messages remain
available, the policy observes the unchanged context, and the existing capacity
guard still stops an oversized request. Explicit manual compaction retains its
existing contract. No frontend access to SDK internals is required.

`FileEditingSettings.from_environment()` supports the explicit temporary
`EXMEKH_INTERACTIVE_TIMEOUT_SECONDS` override. The default remains 300 seconds;
only finite positive values are accepted. The local Metal qualification profile
selects 1200 seconds. Unset the variable to restore the default. Reinstalling
the locked wheel preserves support for this setting.

The patch also removes external DNS from one mocked browser workflow test;
URL validation still runs against a fixed public address.

To rebuild, start from a clean checkout of the pinned SDK revision, apply the
patch with `git apply`, and use the existing `sync-exmekh-lib` command from
`agent/`. That command builds the wheel and refreshes its locked hash before
installing it. Do not replace the patched wheel with an unpatched upstream
build when reproducing these qualification results.

Verification: 1510 SDK owner tests pass from source; 159 compaction/settings
tests pass against the installed wheel. The frontend suite is checked
separately. Reports are under the T06–T07 Metal qualification evidence.
