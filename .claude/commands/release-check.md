Run a thorough pre-release check and produce a report on release readiness.

None of the steps below use the `gh` CLI — it isn't a project dependency. They use the public GitHub REST API instead (unauthenticated, since this repo is public). Compute the repo slug once and reuse it:
```
REPO=$(git remote get-url origin | sed -E 's#.*github\.com[:/]##; s#\.git$##')
```

## Step 1 — Pre-release state

```
git fetch origin main
LAST_TAG=$(git describe --tags --abbrev=0)
git log $LAST_TAG..origin/main --oneline
curl -s "https://api.github.com/repos/$REPO/actions/runs?branch=main&status=completed&per_page=1" | jq '.workflow_runs[0] | {name, conclusion, html_url}'
curl -s "https://api.github.com/repos/$REPO/pulls?state=open&base=main" | jq '[.[] | {number, title, user: .user.login}]'
```
- List every commit merged to main since `$LAST_TAG` — this is what's about to ship
- Confirm the latest completed run on main has `conclusion: success`
- Flag any open PRs targeting main that look intended for this release but aren't merged yet

## Step 2 — Version bump

```
uv version --short
echo "$LAST_TAG"
uv lock --check
```
- Confirm the version in `pyproject.toml` was bumped relative to `$LAST_TAG` (CD verifies the tag matches `pyproject.toml` — see `.github/workflows/cd.yml` — so a mismatch fails the release)
- Confirm `uv.lock` is in sync; if not, note that `uv lock` needs to be run and committed

## Step 3 — Testing

```
uv run pytest tests
```
If `TIGERFLOW_ML_TEST_DIR` is set, integration tests run too. If not set, note explicitly that integration tests were skipped, and recommend running them if any task's model-loading/inference path changed since `$LAST_TAG`:
```
TIGERFLOW_ML_TEST_DIR=<path> uv run pytest tests
```

## Step 4 — Documentation quality

Invoke the `check-docs` skill and incorporate its findings verbatim. Treat any finding here as **blocking** — docs must be fully in sync before a release ships. If issues are found, recommend running `update-docs` first.

## Step 5 — Dependencies & release note labels

```
curl -s "https://api.github.com/repos/$REPO/pulls?state=open&base=main" | jq '[.[] | select(.user.login == "dependabot[bot]") | {number, title}]'
```
- Note whether any open Dependabot PRs are close to landing and should be included in this release
- Re-confirm the Dependency Policy caps in `.claude/CLAUDE.md` still match `pyproject.toml` for `tigerflow`, `vllm`, `transformers`, `torch`

Then check that every merged PR since `$LAST_TAG` has a category label (so it doesn't fall into "Other Changes" in the auto-generated release notes — see `.github/release.yml`):
```
git log $LAST_TAG..origin/main --oneline | grep -oE '#[0-9]+' | tr -d '#' | while read -r n; do
  curl -s "https://api.github.com/repos/$REPO/issues/$n" | jq -c "{number: $n, labels: [.labels[].name]}"
done
```
Flag any PR whose `labels` doesn't include one of: `breaking-change`, `feature`, `bug`, `refactor`, `documentation`.

## Report

One section per step. State ✓ if clean or list concrete blocking issues.

End with:
1. **Overall verdict** — ready to release, or blocked (and by what, in priority order)
2. **Next version** — the version that should be tagged
3. **Release notes highlights** — 2-4 bullets summarizing the most user-relevant changes since `$LAST_TAG`, drawn from the commit/PR list in Step 1
