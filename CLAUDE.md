# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & commands

The `Makefile` starts with `include .env`, so **every `make` target fails without a `.env` file**: `cp .env.sample .env` first (CI does exactly this). Dependencies are managed by `uv` (not Poetry/pip, despite the Poetry-era comments in `pyproject.toml`).

```bash
make install         # installs uv, `uv sync --all-groups --all-extras`, pre-commit hooks
make tests-basic     # the CI suite: tests/basic + test_api, test_runner_caching,
                     # test_pandas_accessors, test_output_modes (< $0.01 of OpenAI spend)
make tests           # everything except tests/ranking and tests/test_ollama.py
make lint            # ruff check docetl/* --fix
make mypy            # strict-ish: disallow_untyped_defs, warn_return_any, files = docetl
make docs-serve      # mkdocs on :8001
```

Single test / targeted runs:

```bash
uv run pytest tests/basic/test_basic_map.py::test_map_operation -s
uv run pytest tests/test_plan_rules.py -x
```

**Most tests make real LLM calls** and need `OPENAI_API_KEY` in `.env`. Tests whose assertions depend on nondeterministic model output are marked `@pytest.mark.flaky(reruns=N)` (pytest-rerunfailures) rather than loosened — follow that convention. `tests/conftest.py` auto-isolates the model-cascade cache per test; without it, a disk cache hit makes zero LLM calls and breaks call-count assertions.

DocWrangler UI (Next.js `website/` + FastAPI `server/`):

```bash
make install-ui && make run-ui-dev   # backend on :8000, Next dev server on :3000
make docker                          # both, in one container
```

Pre-commit runs black, isort (black profile), ruff, and pycln over `docetl/` only — `website/` is formatted by prettier and linted by `next lint`.

## Architecture

### One config dict, three front ends

Everything funnels into a single pipeline **config dict** (`datasets`, `operations`, `pipeline.steps`, `pipeline.output`) that `DSLRunner` (`docetl/runner.py`) executes. The front ends only differ in how they build that dict:

- **Frame API** (`docetl/frame.py`) — the recommended `docetl.read_json(...).map(...).collect()` chain. Each operator returns a new immutable `Frame`; `_build_config` emits the dict, `_memo_key` hashes it so repeated `.show()`/`.collect()` reuse results.
- **YAML** (`docetl/cli.py` → `DSLRunner.from_yaml`), with `docetl/api.py`'s `Pipeline` dataclasses as the typed intermediate.
- **pandas accessors** (`docetl/apis/pd_accessors.py`) — `df.semantic.map(...)`.

Module-level settings (`docetl.default_model`, `rate_limits`, `max_threads`, `plan_rewrites`, …) live in `docetl/_config.py` and are exposed as `docetl.<attr>` via a module-class swap in `docetl/__init__.py`; `_config.runner_settings()` is the single place they get folded into a runner config.

### The config → plan → config round trip

`docetl/plan/` lifts the config into a typed DAG, rewrites it, and lowers it back:

```
config ──lift──▶ LogicalPlan ──apply_rules──▶ LogicalPlan ──lower──▶ config'
```

The contract that everything else depends on: `lower(lift(c))` is deep- and yaml-dump-equal to `c`, and **when no rule fires the original config object is returned untouched** — checkpoint hashes and Frame memo keys are computed from the config, so any churn here silently invalidates caches. `lift` never raises; malformed configs become `PlanIssue`s collected on the plan.

Rewrite rules (`plan/rewrite.py`, e.g. `SelectionPushdown`, `LimitPushdown`) implement `find` + `apply_at` and run to fixpoint. Their safety is decided by **operator traits** — classmethods on `BaseOperation` (`docetl/operations/base.py`): `cardinality()`, `fields_read()`, `fields_written()`, `fields_removed()`, `is_llm()`, `is_row_local()`, `preserves_order()`. Read the `Cardinality` docstring before relying on it: `ONE_TO_ONE` is an *at-most* contract (a row can be dropped on failure), which is why filter pushdown is legal but a positional head is not.

### Pull-based execution

`containers.py:build_operation_graph` turns the plan into a DAG of `OpContainer`s (plus a synthetic `scan_<dataset>` head per step and a `StepBoundary` tail per step). Execution is **demand-driven**: the runner calls `next()` on the last container, which recursively `_pull_children()`. Consequences worth knowing:

- Operations run strictly one at a time; parallelism is *within* an operation (a `ThreadPoolExecutor` over documents). `progress/tracker.py` relies on this to keep a single "current operation".
- Each container tracks **selectivity**, so a downstream request for N sample rows is scaled into a larger upstream request — this is what makes the optimizer's `is_build` sampling passes cheap.
- `CheckpointStore` (`checkpoint.py`) persists per-op results under `intermediate_dir`, keyed by a hash chain over op configs computed in `DSLRunner._compute_hashes`; a changed op invalidates itself and everything downstream.

### Operation registry

`docetl/operations/__init__.py` has a static `mapping` of type-name → class, but `get_operation()` **checks `importlib.metadata` entry points (`docetl.operation` group) first**, so plugins override built-ins. A new built-in operation needs: the class in `docetl/operations/`, an entry in `mapping`, and — if it should be usable as a plugin/by the plan layer's `_resolve_op_class` — an entry point in `pyproject.toml`. Parsers register the same way under `docetl.parser`.

### Two separate optimizers

- **V1** (`docetl/optimizer.py` + `docetl/optimizers/`): sample-based, per-operation (`MapOptimizer`, `ReduceOptimizer`, `JoinOptimizer`), driven from `OpContainer.optimize()` using `SAMPLE_SIZE_MAP`. Largely superseded; its docs are commented out of `mkdocs.yml`.
- **MOAR** (`docetl/moar/` + `docetl/reasoning_optimizer/`): the current optimizer, reached via `docetl build file.yaml` (which requires an `optimizer_config:` block with `evaluation_file` + `metric_key`) or `pipeline.optimize(eval_fn=...)`. `MOARSearch` does graph search over pipeline rewrites, each node scored on cost/accuracy and kept on a `ParetoFrontier`. The rewrites themselves are **directives** in `docetl/reasoning_optimizer/directives/` — each is an LLM-instantiated transformation (`chaining`, `gleaning`, `change_model`, `doc_chunking`, …) with a Pydantic schema in `instantiate_schemas.py` and registration in the directives `__init__.py`. See `docetl/reasoning_optimizer/AGENT.md` and `docs/developer-reference/moar-extensibility.md` for the add-a-directive walkthrough.

### LLM plumbing

`operations/utils/api.py` (`APIWrapper`) wraps litellm for every model call; `fallback_models` become a litellm `Router`. Structured output has two modes (`OutputMode`: tool calls vs. native structured output) — `tests/test_output_modes.py` parametrizes over both. Responses are cached on disk under `~/.cache/docetl` (override with `DOCETL_HOME_DIR`; clear with `docetl clear-cache`); `cache_key()` in `operations/utils/cache.py` defines what counts as the same call. `operations/utils/cascade.py`/`cascade_runner.py` implement BARGAIN model cascades (cheap proxy model + oracle escalation), mixed into operations via `CascadeMixin`.

### Progress UI

`docetl/progress/` is a headless, thread-safe `ProgressTracker` emitting `RunState`/`OpState` snapshots; `docetl/tui/` is a Textual app that polls those snapshots (`DSLRunner._should_use_tui` decides between it and the plain Rich console in `console.py`). Keep instrumentation in the tracker, not the TUI — `tests/test_tui.py` and `tests/tui_screenshots.py` render it headlessly.

### DocWrangler

`website/` is the Next.js app (package name `docwrangler`); `server/app/` is a thin FastAPI layer (`routes/pipeline.py` runs pipelines and streams progress over websockets, `convert.py` does document conversion, `filesystem.py` file access). The UI talks to the same `DSLRunner`, so runner changes can break the playground.

## Conventions

- `from __future__ import annotations` plus PEP 604 (`str | None`) in newer modules; mypy enforces annotated defs under `docetl/`.
- Docstrings carry the non-obvious invariants (the `Cardinality` at-most contract, the plan round-trip byte-stability guarantee, the one-op-at-a-time property the tracker assumes). When changing those behaviors, update the docstring — downstream code is written against it.
- User-facing output goes through `docetl/console.py` (Rich), not `print`.
