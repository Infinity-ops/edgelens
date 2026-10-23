# Performance CI on a Jetson with EdgeLens

A complete example repository: every pull request is measured on a real
Jetson and **cannot be merged when it breaks its assumption contract**.

```
.github/workflows/edgelens-perf.yml   the workflow (self-hosted Jetson runner)
perf/anomaly.contract.yaml               requirements + the conditions they hold under
ci/perf_gate.sh                       measure -> validate -> evidence (exit code = verdict)
pipeline.py                           the pipeline under test (replace with yours)
```

What a run produces (job summary + the `edgelens-evidence-*` artifact):

| File | What |
| --- | --- |
| `evidence.md` | PASS / FAIL / INCONCLUSIVE, one row per requirement ID, the assumptions measured — also shown as the job summary |
| `run.json` | the full EdgeLens result |
| `run.validation.json` | the verdict as JSON |
| `timeline.perfetto.json` | spans + CPU/GPU/power/temperature on one timeline: open at https://ui.perfetto.dev |

| Verdict | Exit | CI |
| --- | --- | --- |
| PASS — every assumption held, every guarantee met | 0 | ✅ |
| FAIL — a guarantee broken under the contract's assumptions | 1 | ❌ (with the limiting stage and why) |
| INCONCLUSIVE — the board was not set up as the contract assumes (clocks not pinned, wrong power mode, hot, busy), or too short a run | 2 | ❌ — fix the runner, not the code |

## Try it locally first (any machine)

```bash
pip install "edgelens>=0.2"            # on a Jetson: without the [onnx] extra
edgelens contract check perf/anomaly.contract.yaml
bash ci/perf_gate.sh                   # off a Jetson: INCONCLUSIVE (board / power mode / clocks)
```

## Set up the Jetson runner (once)

1. **Register a self-hosted runner** on the board: repository → Settings →
   Actions → Runners → *New self-hosted runner* → Linux / ARM64; run the shown
   commands as a normal user, then `sudo ./svc.sh install && sudo ./svc.sh start`.
   Add the label **`jetson`** (the workflow asks for `[self-hosted, linux, ARM64, jetson]`).

2. **Let the runner pin clocks without a password** (and nothing else):

   ```bash
   echo "$USER ALL=(root) NOPASSWD: /usr/bin/jetson_clocks" | sudo tee /etc/sudoers.d/edgelens-ci
   sudo chmod 0440 /etc/sudoers.d/edgelens-ci
   ```

   The power mode is NOT switched by CI (changing nvpmodel can need a reboot):
   set it once (`sudo nvpmodel -m 0` for MAXN) — the contract checks it on every run.

3. **Power sensor** (Jetson Nano only; AGX Orin is readable by default): make
   the `chmod` from `docs/validation-v0.2.md` permanent with a boot service:

   ```bash
   sudo tee /etc/systemd/system/edgelens-power.service >/dev/null <<'EOF'
   [Unit]
   Description=Let EdgeLens read the INA3221 power rails
   [Service]
   Type=oneshot
   ExecStart=/bin/sh -c 'chmod o+r /sys/bus/i2c/drivers/ina3221x/*/iio:device*/rail_name_* /sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input'
   [Install]
   WantedBy=multi-user.target
   EOF
   sudo systemctl enable --now edgelens-power
   ```

4. **No desktop on the board**: `sudo systemctl set-default multi-user.target`
   (the GUI is background load; the contract's `max_background_cpu_pct` catches it).

5. **Python environment**: by default the workflow creates `~/.edgelens-ci`
   (with system site-packages, so JetPack's onnxruntime-gpu and TensorRT stay
   visible). On a Jetson Nano point it at your Python 3.8 venv instead:
   repository → Settings → Variables → `EDGELENS_VENV=/home/you/edgelens-venv`.
   Until v0.2 is on PyPI, install EdgeLens into that venv from your checkout
   (`pip install -e ~/edgelens`); the workflow keeps an installed 0.2.

6. **Make it required**: Settings → Branches → protect `main` → *Require status
   checks to pass* → `edgelens / performance gate`.

Don't add a folder named `edgelens/` at the top of your repository: `python -m
edgelens` would import it instead of the installed package.

**Security:** a self-hosted runner runs whatever code a job contains. Use it on
a private repository, or require approval for workflows from fork pull requests.

## Adapt it

- **Your pipeline:** replace `pipeline.py`, or measure a model directly:
  `EDGELENS_TARGET: --model models/lstm.onnx --provider TensorrtExecutionProvider`
  (then add `provider: TensorrtExecutionProvider` to `assumes` — a silent fallback
  to CPU becomes INCONCLUSIVE instead of a misleading PASS).
- **Your requirements:** edit `perf/anomaly.contract.yaml` — guarantees use the
  same keys as `edgelens validate` flags; requirement IDs go into `id:`.
  `edgelens contract check` tells you how long a run it needs.
- **Long runs** (drift, memory growth, throttling): add soak guarantees
  (`max_latency_drift_pct`, `max_mem_growth_mb_per_h`, `max_throttled_windows`) —
  `contract run` then runs `edgelens soak` for `min_duration`. Do that nightly
  (`on: schedule`), not on every pull request.
- **A C++ application** (your inference runtime, an EDDL program): record it with markers and validate the
  recording against the same contract:

  ```bash
  edgelens record --period-ms 1000 --deadline-ms 3000 --duration 300 --save edgelens-out/run.json -- ./my_app --config app.yaml
  edgelens validate edgelens-out/run.json --contract perf/anomaly.contract.yaml --evidence edgelens-out/evidence.md
  ```
