# Contributing to EdgeLens

Thanks for helping. The most valuable contribution right now is **data from
real boards**.

## Report a board (5 minutes, no code)

Open a *Hardware report* issue with the output of:

```bash
edgelens doctor
bash scripts/validate_on_jetson.sh      # from a source checkout; attach the .tar.gz
```

Every telemetry source EdgeLens supports came from parsing real board output,
and Jetson Orin, Xavier and Orin Nano reports are especially wanted.

## Development setup

```bash
git clone https://github.com/Infinity-ops/edgelens
cd edgelens
pip install -e ".[dev,onnx]"     # on Jetson: pip install -e ".[dev]" + NVIDIA's onnxruntime-gpu wheel
python -m pytest -q
ruff check --select E9,F edgelens tests examples
```

CI runs the tests on Python 3.8–3.13, with and without onnxruntime, and
smoke-tests the built wheel. Please keep Python 3.8 compatible (no `match`,
no `X | Y` type unions, no walrus-heavy code).

## Principles for changes

- **Never fabricate data.** If something can't be measured, report it as
  unavailable with the reason. Demo data is always labelled.
- **Evidence, not confidence.** Diagnosis findings carry the numbers behind
  them and a categorical strength, never an uncalibrated percentage.
- **Evidence before any AI.** Findings come from measured data and
  transparent rules. A model may one day help navigate the evidence; it
  never replaces it, and EdgeLens does not become a chatbot that guesses.
- **Fixtures from real hardware.** A new rule or parser comes with a test
  using numbers/output from a real board, noted in the test.
- **Schema changes are versioned.** Fields in result JSON only change
  meaning with a `SCHEMA_VERSION` bump; readers keep working with older files.

## Releasing (maintainers)

One-time setup (trusted publishing, no API tokens stored anywhere):

1. On pypi.org and test.pypi.org: *Your projects → Publishing → Add a new
   pending publisher*: owner `Infinity-ops`, repository `edgelens`, workflow
   `release.yml`, environment `pypi` (resp. `testpypi`).
2. On GitHub: *Settings → Environments*: create `pypi` and `testpypi`
   (optionally require your approval for `pypi`).

Each release:

1. Set the version in `pyproject.toml` and `edgelens/__init__.py`, and the
   date in `CHANGELOG.md`.
2. *Actions → release → Run workflow* publishes to **TestPyPI**. Check:
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "edgelens==X.Y.Z"`
3. Tag and publish a GitHub Release `vX.Y.Z`; the workflow publishes to
   **PyPI** (it refuses if the tag doesn't match the package version).
