# AGENTS.md

## Purpose

* Write code that is easy to read, understand, test, and maintain.
* Prefer clarity and simplicity over cleverness or unnecessary abstraction.

## Imports and Module Structure

* **No In-Function Imports**: Always place all library and module imports at the very top of the file (module level). Never import libraries, packages, or module symbols inside function or method bodies.
* **Import Ordering**: Group imports cleanly following standard conventions:
  1. Standard library imports (e.g. `asyncio`, `os`, `time`, `pathlib`).
  2. Third-party packages (e.g. `nicegui`, `pydantic`, `httpx`, `pytest`).
  3. Local application and project modules (e.g. `fe.config`, `fe.exmekh_bridge`, `fe.ui`).
* Keep imports explicit and clean; avoid wildcard (`*`) imports.

## Architecture & Code Organization

* Maintain clear separation of concerns between UI layers (`fe.ui`), state management (`fe.state`), and framework integration (`fe.exmekh_bridge`).
* The UI is powered by NiceGUI and must remain responsive; long-running operations or synchronous model calls must be offloaded to worker threads via an executor (`asyncio.get_running_loop().run_in_executor`).
* When updating UI elements from asynchronous tasks or callbacks, ensure execution is within the proper client context (`with client:`).

## Agent & LLM Calls

* Use the typed agentic client (`exmekh_lib.core.models.client.Client`) for interaction with language models.
* Use Pydantic schemas and structured outputs (`client.call_structured`) when deterministic structured payloads are required.
* Prompts provided to models for automated system tasks (such as title generation or metadata extraction) should be written in English with clear instructions to respect the user's conversation language.

## Testing and Verification

* Validate all changes continuously with the automated test suite (`uv run pytest`).
* Ensure both fast unit tests and end-to-end Playwright UI tests pass before committing.
* Never leave broken tests or regression failures in the codebase.
