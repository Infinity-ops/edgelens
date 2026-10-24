# Release checklist

Run top to bottom before tagging a release. A release is tagged only when
every **P0** line is checked; the GitHub release notes link this file with
the boxes as they were.

## P0: gates

- [ ] **Clean installation.** The `package` job of the `tests` workflow is
      green on the release commit: wheel and sdist build, `twine check
      --strict`, the wheel installs in a fresh venv, `pip check`, and
      `edgelens --help`, `doctor`, `benchmark --demo`, `diagnose` and the
      README quick start run from the installed wheel.
- [ ] **Test suite.** The `tests` workflow is green on the release commit
      for Python 3.8, 3.9, 3.10, 3.11, 3.12 and 3.13 (with onnxruntime),
      3.12 without onnxruntime, the lint job, and the test suite run from the
      source distribution. Subprocess tests included: none may be skipped
      for a reason other than missing hardware or an optional package.
- [ ] **Hardware truth.** Every claim about a board in README.md,
      CHANGELOG.md and docs/ matches a row in [jetson.md](jetson.md), and
      every **measured** row has its `SUMMARY.txt` / result files archived
      (not in the repository; in the maintainer's records). Board names are
      the module name `edgelens doctor` prints.
- [ ] **Measurement integrity.** `tests/test_release_integrity.py`,
      `tests/test_v02_trust.py` (statistical compare) and
      `tests/test_core_engine.py` (statistics, deadlines) pass; no open
      issue labelled `measurement` without a note in
      [limitations.md](limitations.md).
- [ ] **Interfaces frozen.** `tests/test_frozen_interfaces.py` passes
      against the snapshot of the release line. A removed or renamed
      command, option, contract / stream key or result field needs a new
      minor version and an "Upgrading" entry.
- [ ] **Version.** `pyproject.toml`, `edgelens/__init__.py`, `CITATION.cff`
      and the CHANGELOG heading agree; the tag is `v` + that version (the
      release workflow checks the first and the last).
- [ ] **No third-party material** in the repository that its owners have
      not cleared: data, models, results, names of partners' systems.

## P1: first-time users

- [ ] The README quick start, copied from the rendered README into a fresh
      Linux machine (no Jetson), works as written.
- [ ] The README "On a Jetson" block works as written on a real Jetson; note
      the board and JetPack in the release notes.
- [ ] [examples/quickstart/README.md](../examples/quickstart/README.md):
      the expected output still looks like a current run.
- [ ] The compatibility statement in the README (Python versions, extras,
      OS, boards) and [limitations.md](limitations.md) are current.

## Publishing

1. Merge to `main` (squash: one commit per release keeps `main`'s history
   readable).
2. Run the `release` workflow by hand: it publishes to **TestPyPI**. In a
   fresh venv: `pip install -i https://test.pypi.org/simple/
   --extra-index-url https://pypi.org/simple/ edgelens==X.Y.Z` and run the
   quick start.
3. Tag `vX.Y.Z` on `main` and publish a GitHub Release with the CHANGELOG
   section: the workflow publishes to PyPI.
4. `pip install edgelens==X.Y.Z` in a fresh venv; run the quick start once
   more.

## P2: after the release

- [ ] Ask two or three people outside the project to install it from PyPI
      and run the quick start and one command on their own system; collect
      friction and measurement doubts as issues (the *hardware report* and
      *bug report* issue templates).
- [ ] Hardware reports from boards not yet in [jetson.md](jetson.md): add a
      row only from a run whose output was attached.
