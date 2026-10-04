# ExMekh Agent — Local AI Studio Frontend for ExMekh-Lib

A Python frontend inspired by modern local LLM interfaces such as **LM Studio**, **Unsloth**, and **BionicGPT**. It is designed to integrate directly with the hierarchical agentic framework `exmekh_lib` and its distributed wheel package.

---

## Key Features

- **Dark Theme Inspired by LM Studio & Unsloth**: High-contrast obsidian/charcoal palette with support for markdown syntax, code blocks, and modern aesthetics.
- **Real-Time Streaming**:
  - **Collapsible Thinking Block**: visualization of reasoning tokens with an expandable animated view, identical to DeepSeek-R1 / LM Studio.
  - **Tool & Skill Execution Cards**: monitoring of hierarchical tools invoked by `DeepAgent` with depth tracking and skill paths.
  - **Turn Steering**: send a queued message as guidance to the active DeepAgent run. Accepted guidance and its delivery position remain visible after a page refresh and in saved conversation history. Guidance delivered inside a skill retains its canonical output ID in `get_execution_history` and `read_tool_output` across later turns, including its attached images and rendered PDF pages.
- **Images and PDFs**:
  - Pick, drop, or paste several images in one message, including queued guidance.
  - Click thumbnails in the composer or saved chat for a large preview with a blurred backdrop; close it with the button or Escape.
  - PDFs render locally into images: at most the first **10 pages per document**, in page order. The composer shows the selected/total page count and warns when pages are omitted.
  - Images keep their orientation and are resized to at most **2048 pixels per side**. Animated images use their first frame.
  - Original files stay in scoped temporary storage; normalized PNG content is also saved in chat JSON so vision history and previews survive refreshes and file cleanup. Chat messages show only previews for these visual attachments.
  - Chat Completions and Responses both use ExMekh-Lib mixed user inputs. The endpoint/model must support vision; DS4 needs its matching encoder loaded with `--vision`.

- **Local Provider Control**:
  - Dynamic health check of the local server endpoint (LM Studio, Ollama, llama.cpp, vLLM).
  - Dynamic discovery and selection of installed models.
  - Project conversations automatically use **DeepAgent**; standalone **Quick Chats** use the typed client without skills or tools.
- **Projects and Quick Chats**:
  - Projects are named after their selected work folder. The folder is fixed after creation.
  - Shared HDS, capabilities, coding-agent history, and default bash approval settings open beside each project name.
  - Bash approval changes inside a chat stay local; the project default applies to newly created conversations.
  - Persistent chat history lives under `data/chats/`, and project settings under `data/projects/`.
  - Temporary uploads and pasted files use `<work_root>/tmp/projects/<project_id>/`; Quick Chats use the frontend code folder and `tmp/quickchats/`.
  - Input drafts and uploads stay with their original conversation when switching chats. Quick Chats include text attachments directly in the prompt and send images and rendered PDF pages as native vision inputs.
  - Deleting a project deletes its conversations and stops their active turns. The selected work folder remains available.
  - Existing agent conversations are automatically grouped by work folder. When their previous settings differ, the most recently updated chat supplies the shared project settings; each chat keeps its bash approval mode and approval history.
- **Automatic `exmekh_lib` Wheel Synchronization**:
  - WSL shell script to build the sibling ExMekh-Lib sources, update the frontend wheel and lockfile, and reinstall ExMekh-Lib.

---

## Wheel Synchronization from `exmekh_lib`

From the repository root in WSL:

```sh
./agent/sync_exmekh_lib.sh
```

The script builds the current sources in the sibling `exmekh_lib/` directory, copies the
resulting wheel into `agent/wheels/`, updates its path in `pyproject.toml` and
`uv.lock`, and reinstalls ExMekh-Lib. After a successful sync, it removes older ExMekh-Lib wheels.
No previous build or frontend virtual environment is required. Wheel cleanup runs
only after a successful installation.

The installed sync command is `uv run sync-exmekh-lib` from `agent/`.
To start the framework's interactive CLI with the frontend's locked wheel:

```sh
cd agent
uv run --frozen --env-file .env exmekh-lib
```

The Python namespace is `exmekh_lib`; the distribution and console command are
`exmekh-lib`. The frontend integration module is `exmekh_agent.exmekh_bridge`. The CLI
loads `BASE_URL` and `API_KEY` from `.env` with the command above. Set `EXMEKH_LIB_MODEL`
and `EXMEKH_LIB_API` to override its model and API; these are the environment variable
names exposed by the renamed library. After updating the library sources, sync the
wheel so the installed configuration matches them.

Both shell launchers resolve paths from their own location, so they can be called
from any working directory. They use `uv` from `PATH` or `~/.local/bin/uv`;
set `UV_BIN=/path/to/uv` to choose another executable.

---

## Starting the Frontend

From the repository root in WSL:

```sh
./agent/run.sh
```

The launcher installs the locked dependencies when needed and starts the frontend.
From `agent/`, the equivalent command is `uv run --frozen exmekh-agent`.
The distribution and UI command are `exmekh-agent`; Python imports use
`exmekh_agent`. The framework CLI remains `exmekh-lib`.

Copy `.env.example` to `.env` to configure the application. Its web settings
are `EXMEKH_AGENT_HOST`, `EXMEKH_AGENT_PORT`, and `EXMEKH_AGENT_TITLE`. Existing
`.env` files using `FE_HOST`, `FE_PORT`, and `FE_TITLE` must use those new names.
Provider settings such as `BASE_URL`, `API_KEY`, and `DEFAULT_MODEL` stay the same.
The model selector and context bar refresh the provider's `/models` metadata at
startup and every 10 seconds. If the saved model is unavailable, the running
client selects an advertised model; supported aliases remain selectable. DS4
reports the allocated `--ctx`, so switching between Qwen Flash and Qwen 27B also
updates the displayed context capacity. Automatic discovery does not rewrite
`.env`; the Save button persists settings.
On startup, saved projects and chats that point to the previous library's bundled
`site-packages/qs/capabilities` are migrated to the installed library defaults.
Custom capabilities paths, conversation history, and approvals are preserved.

The application will open at:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Running Tests

### Offline Unit Tests
Run the standard (offline) tests:
```shell
cd agent
uv run pytest
```

Tests run by module on all logical CPUs available to pytest by default, keeping
module fixtures on one worker. Real-provider `live` tests (excluding `mock`) and tests marked
`@pytest.mark.serial(reason="...")` run one at a time after all parallel workers
have finished, including fixture teardown. Clipboard tests use this marker
because the operating system clipboard is shared between browsers.
Use `-n 0` to debug without workers, or `-n 16` to choose a worker count.
`PYTEST_XDIST_AUTO_NUM_WORKERS=16 uv run pytest` sets the default for `-n auto`.
On WSL, automatic detection follows the CPU limit configured in `.wslconfig`;
Windows and WSL share the host CPU resources.
Browser servers use available localhost ports and isolated storage, so separate
pytest invocations can run concurrently. Server logs are retained in `.test-tmp`.

The test suite automatically enforces branch coverage and requires 100% on the
Python configuration, state, and ExMekh-Lib integration code. The calculation excludes the
`exmekh_agent.main` and `exmekh_agent.sync` entrypoints and the UI modules listed in `pyproject.toml`, which are verified by
functional and Playwright tests in the same suite. To run only a subset of tests,
use `--no-cov`; the threshold applies to the full suite.

To run quality checks and build distribution separately:

```shell
uv run ruff check src tests
uv run ruff format --check src tests
uv build
```

### Live End-to-End Tests
Tests with mock providers exercise the agent workflow without an LLM server:
```shell
cd agent
uv run pytest --no-cov -m mock
```

For a vision-enabled DS4 server, verify multiple images, PDF conversion and
follow-up turns through both APIs:

```shell
uv run pytest --no-cov --run-live -n 0 tests/test_images_live.py
```

DS4 accepts finite local `#/$defs/name` references emitted by nested Pydantic
output models, including nullable branches. All constraints remain active;
external references, recursive schemas and unsupported constraints return an
explicit error. Too little output budget fails instead of accepting partial JSON.

For a Flash server, verify strict Pydantic output through the installed client
with both Chat and Responses, streaming, thinking, greedy/sampled decoding,
nullable nested fields, Unicode and truncated-output rejection. Run against
one server at a time, first target-only and then MTP if enabled:

```shell
DS4_FLASH_STRUCTURED_BASE_URL=http://127.0.0.1:8080/v1 \
  uv run pytest --no-cov --run-live -n 0 tests/test_flash_structured_live.py
```

With the provider configured in `.env` available, run all tests with a real model (conversation, streaming, tools, persistence, title generation, and adversarial cases of `exit_agent`):
```shell
uv run pytest --no-cov --run-live -m "live and not mock" -s -v
```

`--run-live` enables tests that send requests to the model. Without this option (or `EXMEKH_AGENT_RUN_LIVE_TESTS=1`), the standard suite skips them; mock tests remain runnable.
Each adversarial case of `exit_agent` has a 90-second timeout. A timeout terminates the process and marks the case as skipped because recovery was not verified. Cases where the model does not produce the requested malformed payload are also skipped. Deterministic argument tests independently check omitted `items` and invalid item types.

---

## Package Structure

```
agent/
├── .env                  # BASE_URL, API_KEY, MODEL configuration
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── pyproject.toml        # uv configuration and dependencies
├── README.md             # Project documentation
├── .python-version      # Python 3.13
├── run.sh               # WSL launcher for the UI
├── sync_exmekh_lib.sh    # Build and install the sibling ExMekh-Lib sources
├── start-ds4-server.sh  # WSL launcher for Qwen27B
├── start-ds4-server-flash.sh  # WSL launcher for Qwen Flash Next
├── wheels/               # Synchronized exmekh_lib wheel
│   └── exmekh_lib-0.1.0-py3-none-any.whl
├── data/
│   ├── chats/           # JSON storage for conversations
│   └── projects/        # Shared project settings
└── src/
    └── exmekh_agent/
        ├── __init__.py
        ├── main.py       # NiceGUI entry point
        ├── config.py     # Configuration and capabilities paths
        ├── sync.py       # Python sync script
        ├── exmekh_bridge/
        │   ├── __init__.py
        │   ├── agent_runner.py  # Asynchronous exmekh_lib runner (Client and DeepAgent)
        │   ├── image_input.py    # Image/PDF decoding and multimodal history
        │   └── models.py        # Pydantic models for messages and sessions
        ├── state/
        │   ├── __init__.py
        │   ├── json_store.py      # Atomic JSON storage
        │   ├── temp_storage.py    # Scoped attachment storage and cleanup
        │   └── session_manager.py # Projects, chat state, and migration
        └── ui/
            ├── __init__.py
            ├── page_controller.py # Page coordination, async tasks, and active session
            ├── styles.py         # LM Studio obsidian theme CSS
            ├── header.py         # Branding, sidebar toggle, and global settings
            ├── sidebar.py        # Projects and Quick Chats sidebar
            ├── chat_view.py      # Chat timeline with thinking and tool blocks
            ├── image_preview.py   # Shared thumbnails and modal preview
            ├── input_bar.py      # Input bar with send and stop buttons
            └── settings_drawer.py # Advanced settings drawer
```

## DS4 integration

This checkout lives in `agent/`. The sibling `exmekh_lib/` source copy is ignored by
Git; the versioned wheel in `wheels/` is the runtime dependency. In WSL/Linux,
start either `./start-ds4-server.sh` (Qwen27B) or
`./start-ds4-server-flash.sh` (Flash Next), then `./run.sh` in a separate terminal. The model
server uses port 8080 and the frontend uses port 8000. Set `API_TYPE=responses`
and keep the client context at or below the server allocation reported by `/v1/models`.
See [the DS4 frontend guide](../docs/agentic/frontend.md).

Browser tests use installed Edge when available and otherwise use Playwright Chromium.

The `.python-version` pins Python 3.13, matching the source environment and test baseline.

### Selecting the local inference model

From `agent/`, use a dedicated server launcher for each model:

```sh
./start-ds4-server.sh        # Qwen27B: MTP, context 28000
./start-ds4-server-flash.sh  # Flash Next: MTP, context 131072, prefill chunk 256
```

Start the frontend with `./run.sh` in another terminal. Each launcher selects
its model profile even if `DS4_MODEL_PROFILE` is set in the environment.
Model discovery reads the real server model and allocated context. Flash uses
its matching projector and CUDA tiered cache. Both launchers enable MTP.

Both scripts accept `--print-command` to review the exact command without
loading weights. `DS4_CTX`, `DS4_PREFILL_CHUNK`, `DS4_MODEL`, `DS4_VISION_MODEL`
and `DS4_KV_DISK_DIR` can override the profile. An empty vision path disables
vision; an empty KV directory disables the shared launcher's KV disk arguments.
See [Flash pipeline qualification](../docs/performance/qwen38-flash-pipeline-20261004.md)
for measured scenarios and remaining limits.
