Run a thorough pre-PR quality check of the current branch and produce a report.

## Step 1 — Determine diff scope

Run:
```
git fetch origin main
git diff --name-only origin/main...HEAD
git log origin/main..HEAD --oneline
```

If there are no commits ahead of main, stop and report that there is nothing to check.

Classify the changed files into buckets — everything below is scoped to only the buckets that are non-empty:
- **code**: `src/**`
- **tests**: `tests/**`
- **docs**: `docs/**`, `README.md`, in-tree `**/README.md`
- **deps**: `pyproject.toml`, `uv.lock`
- **other**: anything else (`.claude/**`, `.github/**`, etc.)

## Step 2 — Code quality


If **code** is non-empty, inspect `git diff origin/main...HEAD -- src` for issues the linters won't catch:
- Debug statements (`print(`, `pdb.set_trace`, `breakpoint()`)
- Commented-out code (not genuine explanatory comments)
- Leftover `TODO`/`FIXME`/`XXX` without context
- New code that doesn't follow the task pattern in `.claude/CLAUDE.md` (`_base.py` / `local.py` / `slurm.py` split, `Params` extending `HFParams`)
- Potential bugs

If **code** is empty, skip this diff scan and say so.

## Step 3 — Testing

Skip this step entirely if **code** and **tests** are both empty (e.g. a docs-only or config-only PR) — say so and move on.

Otherwise run:
```
uv run pytest tests
```
Then check coverage: for each changed/added file under `src/tigerflow_ml/**`, check whether a corresponding file under `tests/unit/**` or `tests/integration/**` was touched. Flag new params/behavior with no test change. Note if the change would benefit from integration coverage (skipped locally without `TIGERFLOW_ML_TEST_DIR`).

## Step 4 — Documentation quality

Skip this step if **docs** and **code** are both empty — say so and move on.

Otherwise invoke the `check-docs` skill and incorporate its findings verbatim — list every issue it reports, don't summarize them away.

Additionally, if **code** touches params, entry points, or vllm requirements but **docs** is empty, flag the diff as likely introducing stale docs even before `check-docs` catches it (e.g. a brand-new task with no docs page yet).

## Step 5 — Dependencies & lockfile

Skip if **deps** is empty.

Otherwise run:
```
uv lock --check
```
and cross-check any new/bumped dependency against the Dependency Policy in `.claude/CLAUDE.md` (should it be capped? does it actually do what it's meant to do — check the package's actual PyPI identity, not just its name).

## Step 6 — Final steps

- Commits: skim `git log origin/main..HEAD --oneline` for clear messages; flag "wip"/"fixup"/"tmp" commits that should be squashed
- Branch currency: `git rev-list --left-right --count origin/main...HEAD` — if main is ahead, flag that the branch should be rebased before opening the PR

## Report

One section per step actually run above (mark skipped steps as "skipped — <reason>", don't pad the report with empty sections). For each non-skipped section, state ✓ if clean or list concrete findings (file:line where possible).

End with:
1. **Overall verdict** — ready to open, or blocked (and by what)
2. **Recommended PR title** — conventional-commit style, matching recent `git log`
3. **Recommended label** — one of the categories in `.github/release.yml` (`breaking-change`, `feature`, `bug`, `refactor`, `documentation`, or none)
