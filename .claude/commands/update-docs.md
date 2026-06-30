Regenerate all documentation in this repo to match the source code.

## Step 1 — Discover tasks and read source files

Run the following to find all task source files:
```
find src/tigerflow_ml -name "_base.py" | sort
```

Each `_base.py` at path `src/tigerflow_ml/<domain>/<taskname>/_base.py` corresponds to the docs page `docs/mkdocs/tasks/<taskname>.md`. Build this mapping dynamically from the find output.

Also read:
- `src/tigerflow_ml/params.py` — base param classes (`HFParams`, `VLLMParams`)
- `pyproject.toml` — task entry points under `[project.entry-points."tigerflow.tasks"]`

Then read every discovered `_base.py` and its corresponding docs page (if it exists).

## Step 2 — Update the mkdocs task pages

For each discovered task, read its docs page and replace the `## Parameters` table with one generated from the `Params` class. If the docs page doesn't exist yet, note it but don't create it (that requires human-authored content beyond the params table).

Preserve all other content (intro, `## Supported Input Formats`, `## Output Format`, `## Models`, `## Examples`, etc.).

Rules for building each table:

**What to include / exclude:**
- Include all params from the inherited base class and the task-specific `Params` subclass
- **Exclude `hidden=True` params**
- **Override resolution** — when a task's `Params` redeclares an inherited field, use the task-specific `default` and `help`; do not emit the parent's version separately

**Column: Parameter** — `--<flag-name>` with `_` → `-`

**Column: Default:**
- `None` or `""` → leave blank
- `False` bool → `--no-<flag>`
- `True` bool → `--<flag>`
- Complex expressions → evaluate or show as a formula when that's more informative (follow existing translate docs style for `--max-model-len`)
- All other values → render literally

**Column: Description** — use the `help=` string, lightly edited for prose

**Ordering** — base-class params first (definition order), then task-specific params in definition order

## Step 3 — Update `docs/mkdocs/index.md`

Read `docs/mkdocs/index.md`. Update the `## Available Tasks` list so that:
- Every task discovered in Step 1 is listed
- Each description matches the intro sentence of the corresponding mkdocs task page

Preserve all other content (badges, installation, Next Steps).

## Step 4 — Update the root `README.md`

Read `README.md`. Update the `## Tasks` table so that:
- Every entry point from `pyproject.toml` has a row (add missing, remove stale)
- Entry point names match `pyproject.toml` exactly
- Task descriptions are accurate

Preserve all other content (badges, Installation, Usage, Container, Development).

## Step 5 — Update in-tree READMEs

Run:
```
find src/tigerflow_ml -name "README.md" | sort
```

For each found README, read it and update any param documentation to match the corresponding `_base.py`:
- Update param tables or bullet-point descriptions with correct defaults and descriptions
- Remove mentions of params that no longer exist
- Do not add new params that aren't already described (the README author chose what to include)
- Preserve the existing format and prose style of each README; do not convert between formats (e.g. don't turn a bullet list into a table)
- Some READMEs intentionally omit shared base-class params — preserve this design choice

## Step 6 — Update vllm installation instructions

The `vllm` optional dependency (defined in `pyproject.toml` under `[project.optional-dependencies]`) is only required by tasks whose `Params` class inherits from `VLLMParams`. Tasks that inherit from `HFParams` do not need it.

From the `_base.py` files read in Step 1, determine which tasks use `VLLMParams` vs `HFParams`. This is the authoritative list of vllm-requiring tasks.

Update every place in the docs that names which tasks need the `vllm` extra:
- `README.md` — the `pip install tigerflow-ml[vllm]` block and surrounding prose
- `docs/mkdocs/index.md` — the installation section
- `docs/mkdocs/tasks/<taskname>.md`- if vllm install is needed, this should be mentioned before `## Parameters`

In each location, rewrite the task list to exactly match the vllm-requiring tasks derived from the code. Preserve all surrounding prose structure and formatting; only update the task names within it.

## Summary

After all edits, print one line per file: what changed (e.g. "detect.md: added --dtype, --compile") or "no changes needed".
