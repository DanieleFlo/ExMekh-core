# ExMekh Lib patches in the bundled wheel

The current wheel is built from ExMekh Lib revision
`260bc59397872c391333c3f9fd00a2832ff90b46`. It contains the native progress
protocol and the provider request adapters used by the node runtime. The
`exmekh-lib-native-image-upload.patch` adds PNG conversion at the upload
boundary for raster formats outside the native service's MIME list. The SDK
uses the first frame of an animated image and preserves the original file.
The wheel SHA256 in `agent/uv.lock` is
`76396a549db4184c168f093d2cadd6cece497807604f811c1ae3b09a08bcdd95`. Its dependency metadata is unchanged.

To rebuild, check out that SDK revision, apply the upload patch with
`git apply`, and run `uv build --wheel`. Copy
the resulting wheel to `agent/wheels/` and refresh its locked hash before
running `uv sync --locked --reinstall-package exmekh-lib`.

The archived `exmekh-lib-image-progress.patch` describes the earlier wheel
built from SDK revision `ed3a744c9843ce9980f2fdf87db32c73d104cd91`. The
patch includes stream validation, model-independent discovery and the optional
`active_items` field for shared batch progress. It is retained for reproducing
the qualification evidence for that earlier wheel.

Previous wheel baseline:

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
