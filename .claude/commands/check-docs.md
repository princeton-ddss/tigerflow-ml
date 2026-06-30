Audit all documentation in this repo for correctness against the source code.

## Step 1 — Discover tasks and read source files

Run the following to find all task source files:
```
find src/tigerflow_ml -name "_base.py" | sort
```

Each `_base.py` at path `src/tigerflow_ml/<domain>/<taskname>/_base.py` corresponds to the docs page `docs/mkdocs/tasks/<taskname>.md`. Build this mapping dynamically from the find output.

Also read:
- `src/tigerflow_ml/params.py` — base param classes (`HFParams`, `VLLMParams`)
- `pyproject.toml` — task entry points under `[project.entry-points."tigerflow.tasks"]`

Then read every discovered `_base.py` and its corresponding docs page.

## Step 2 — Check the mkdocs task pages

For each discovered task, compare the `## Parameters` table in `docs/mkdocs/tasks/<taskname>.md` against the `Params` class in `_base.py` (including inherited fields from `HFParams` or `VLLMParams`). If the docs page does not exist, flag it as missing.

Report for each task:
1. **Missing from docs** — params in code absent from the table
2. **Extra in docs** — params in the table that don't exist in code
3. **Wrong default** — default shown in docs doesn't match code
4. **Description drift** — help text significantly diverges from the `help=` string
5. **Missing docs page** — `docs/mkdocs/tasks/<taskname>.md` does not exist

Comparison rules:
- **Skip `hidden=True` params** — never appear in docs
- **Override resolution** — when a task's `Params` redeclares an inherited field, use the task-specific `default` and `help`
- **Name conversion** — Python `_` → CLI `-` (e.g. `cache_dir` → `--cache-dir`)
- **Bool flags** — `False` default → `--no-<flag>`; `True` default → `--<flag>`
- **`None` or `""` default** — treat as blank/absent in docs

## Step 3 — Check `docs/mkdocs/index.md`

Read `docs/mkdocs/index.md`. Verify:
- Every task discovered in Step 1 is listed under `## Available Tasks`
- Each entry's description matches the intro of the corresponding mkdocs task page
- All links point to existing task pages

## Step 4 — Check the root `README.md`

Read `README.md`. Verify the `## Tasks` table:
- Every entry point in `pyproject.toml` (`[project.entry-points."tigerflow.tasks"]`) has a corresponding row
- Entry point names in the table match `pyproject.toml` exactly
- Task descriptions are accurate

## Step 5 — Check in-tree READMEs

Run:
```
find src/tigerflow_ml -name "README.md" | sort
```

For each found README, read it and check it against the corresponding `_base.py`:
- If the README contains a param table or bullet-point param list, verify that each param exists in the code with a matching default and description
- Flag params described in the README that no longer exist in code
- Flag params that have changed their defaults without the README being updated
- Note: some READMEs intentionally omit shared base-class params (e.g. HFParams fields) and defer to `--help` — this is correct, not an error

## Step 6 — Check vllm installation instructions

The `vllm` optional dependency (defined in `pyproject.toml` under `[project.optional-dependencies]`) is only required by tasks whose `Params` class inherits from `VLLMParams`. Tasks that inherit from `HFParams` do not need it.

From the `_base.py` files read in Step 1, determine which tasks use `VLLMParams` vs `HFParams`. This is the authoritative list of vllm-requiring tasks.

Check every place in the docs that names which tasks need the `vllm` extra:
- `README.md` — the `pip install tigerflow-ml[vllm]` block and any surrounding prose
- `docs/mkdocs/index.md` — the installation section
- `docs/mkdocs/tasks/<taskname>.md`- if vllm install is needed, this should be mentioned before `## Parameters`

For each location, verify:
1. **Missing task** — a vllm-requiring task is not mentioned
2. **Extra task** — a task is listed as needing vllm but its `Params` inherits `HFParams`
3. **Stale prose** — the surrounding sentence names specific tasks and that list is wrong

## Summary

Print a grouped summary:
- Per-file: list of issues found (or "✓ OK")
- Overall: total issue count
