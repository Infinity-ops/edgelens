# Performance CI on a Jetson (v0.2 Layer 3)

A one-off profiler answers "how fast is it today?". A performance gate in CI
answers it for every change, and stops the change that breaks a requirement
before it is merged.

The complete example repository is in
[`examples/ci-jetson/`](../examples/ci-jetson/README.md): a GitHub Actions
workflow for a self-hosted Jetson runner, an assumption contract
([contracts.md](contracts.md)), the gate script and a DEM2-style pipeline.
Copy the folder into a new private repository to start.

How the gate works:

1. `edgelens contract check` — the contract file is valid (no typos).
2. Pre-run check (bounded wait, `EDGELENS_COOLDOWN_S`): board, power mode,
   pinned clocks, start temperature and background load are as the contract
   assumes. A hot board gets time to cool down; a board in the wrong power
   mode fails fast with INCONCLUSIVE, without measuring.
3. `edgelens contract run` — measures what the contract needs (period,
   deadline, iterations, duration) and validates.
4. The evidence record becomes the job summary; the result, the verdict and
   the Perfetto timeline are uploaded as an artifact.
5. The exit code is the verdict: 0 PASS, 1 FAIL, 2 INCONCLUSIVE. Anything but
   PASS fails the job, so a branch-protection rule blocks the merge.

Design choices worth knowing:

- **INCONCLUSIVE blocks the merge too.** If the runner is not set up as the
  contract assumes, the run proves nothing about the change; letting it pass
  would make the check meaningless exactly when the setup drifts.
- **One job per board at a time** (`concurrency`): a second benchmark on the
  same board is background load.
- **The runner never switches the power mode** (that can need a reboot); it
  pins clocks with `jetson_clocks` and restores them afterwards. The contract
  checks the power mode on every run.
- **Self-hosted runners execute the job's code**: use a private repository or
  require approval for fork pull requests.
- **Long checks nightly:** soak guarantees (drift, memory growth, throttling)
  belong in a scheduled workflow, not in every pull request.
