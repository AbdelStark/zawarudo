---
description: Drive the end-to-end Push-T demo epic (#1) and its child issues #2–#10 to completion, in dependency order, with TDD + validation + per-issue commits/PRs.
argument-hint: "[all | <issue#> | from <issue#> | next]  (default: next ready issue)"
allowed-tools: Read, Edit, Write, Bash, Grep, Glob, Task
---

# Goal: ship the end-to-end Push-T demo (epic #1)

You are the orchestrator for **`AbdelStark/zawarudo`** epic **#1** — a one-command demo where
`docker compose up` launches a **client + backend**, serving a **real Push-T LeWorldModel checkpoint**
with **full monitoring** (Prometheus + OpenTelemetry + Grafana). Conformance target: `rfc/0005-pusht-demo-profile.md`.
Engine decision: ADR-0001 (Ray Serve + FastAPI + PyTorch).

**Always load context first:** read `CLAUDE.md` and the relevant skill(s) in `.codex/skills/` before
touching code. The skill registry is `.codex/skills/_index.md`.

## Scope (from `$ARGUMENTS`, default = `next`)
- `next` / empty → pick the lowest-numbered **ready** issue (all deps closed) and do exactly that one.
- `<n>` → do only issue #n (verify its deps are closed first; if not, say so and stop).
- `from <n>` → do #n then continue forward through the DAG.
- `all` → walk the entire DAG to completion, one issue at a time.

## Dependency DAG (do in this order; respect blockers)
```
#2 packaging ──┬─▶ #3 runtime ──┬─▶ #5 golden
               │                ├─▶ #7 compose ──▶ #9 monitoring ──▶ #10 acceptance
#6 client ─────┘                ├─▶ #8 benchmark
#4 observability ───────────────┴─▶ #9 monitoring
```
- #2, #4, #6 have **no blockers** — safe to start/parallelize.
- #3 needs #2. #5 needs #2+#3. #7 needs #3+#6. #8 needs #3. #9 needs #4+#7. #10 needs #2–#9.
- Independent issues with no shared files may be delegated to parallel sub-agents (Task). Serialize
  anything touching the same files, the WMCP contract, or `deployment/`.

## Per-issue loop (run for each issue you take on)
1. **Sync**: `gh issue view <n>` — read the full body, tasks, and acceptance criteria. Confirm deps closed.
2. **Branch**: `git checkout -b feat/issue-<n>-<slug>` off `main`. Never commit straight to `main` for code.
3. **Plan**: restate the acceptance criteria as a checklist. Load the matching skill
   (`runtime-backend`, `wmcp-api`, `observability`, `testing`, `model-packaging`, `deployment`).
4. **TDD**: write/extend tests first where feasible (`tests/`), then implement under `src/` or new
   packages (`client/`). Keep the `mock` backend path working — it is the contract-test fixture.
5. **Validate** — all must pass before commit:
   - `uv run --extra dev pytest -q`
   - `uv run --extra dev ruff check .`  and  `uv run --extra dev mypy src`
   - if deployment touched: `cd deployment && docker compose config --quiet`
   - if a service/client added: build it and smoke it (`curl :8080/readyz`, run the client once)
6. **Commit**: Conventional Commits, **no Claude attribution** (repo convention). Reference the issue:
   `feat(<scope>): <summary> (#<n>)`.
7. **Ship**: push the branch and open a PR with `gh pr create` linking `Closes #<n>`; or, if the user
   asked to push to `main`, fast-forward after validation. Use the `describe_pr` skill for the PR body.
8. **Track**: tick the corresponding box in epic **#1**'s checklist (`gh issue edit 1`), and let the PR
   close the child on merge.

## Hard boundaries (from CLAUDE.md — do NOT bypass)
- **GATED, needs explicit approval before editing**: `src/wmcp_jepa_service/schemas.py`, `api/openapi.yaml`,
  `schemas/*.schema.json` (the WMCP contract must stay in sync across all three), `deployment/**`,
  `pyproject.toml` deps, `adr/**`, `rfc/**`. Pause and ask.
- **Forbidden**: `.env*`/secrets; never load model weights or code from an inference request payload.
- Declare new deps in `pyproject.toml` extras — no ad-hoc `pip install`.

## Real-model reality (issues #2/#3/#5)
- The HF checkpoint `quentinll/lewm-pusht` + upstream `le-wm`/`stable-worldmodel` are heavy and the real
  runtime wants a GPU. If weights/GPU are unavailable in this environment:
  - implement and unit-test everything that does **not** require the weights (loader, packaging, shapes,
    span wiring, client, compose),
  - keep `WMCP_BACKEND=mock` as the default and the contract-test path,
  - mark golden/GPU acceptance as **blocked-on-environment** in the issue and escalate rather than fake it.
- Correctness before optimization: no AMP / `torch.compile` until #5 golden validation passes (fp32 first).

## Escalate to the user (don't guess) when
- a GATED file must change, deps must be added, or the WMCP contract must evolve;
- weights/GPU are required but unavailable;
- an acceptance criterion can't be met without a design decision;
- you'd otherwise be blocked > one validation cycle.

## Definition of done (epic #1)
`make demo` (or `docker compose up`) brings up client **and** backend on the real `lewm-pusht` checkpoint;
all RFC-0005 required endpoints respond; Prometheus metrics **and** OTel traces are visible in Grafana for
every operation; a `score-medium` benchmark report is committed; every RFC-0005 acceptance criterion is checked.

---
**Start now:** resolve `$ARGUMENTS` to the target issue(s), print the chosen plan (issue #, branch name,
acceptance checklist), then execute the per-issue loop. After each issue, report status and stop if the
scope was a single issue; otherwise continue to the next ready issue in the DAG.
