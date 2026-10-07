# Usage: three ways to measure a pipeline

EdgeLens works the same way whichever of these you pick: the result is the
same JSON document (`schema_version: 1`, see [architecture.md](architecture.md)),
so `diagnose`, `report`, `compare` and `validate` all work identically
regardless of how the run was produced.

## 1. Just a model: `--model`

```bash
edgelens benchmark --model model.onnx --deadline-ms 33
edgelens benchmark --model sensor_model.onnx --input-shape vib:1x8x2048   # dynamic/multi-input
```

Every model input is fed with its declared dtype. Dynamic dimensions of
non-image inputs are never guessed: EdgeLens asks for `--input-shape`
rather than benchmarking a workload that doesn't exist.

## 2. Your own pipeline, EdgeLens drives the loop (harness mode)

```python
import edgelens as el

pipe = el.Pipeline("bearing-monitor", pack="timeseries",
                   sample_rate_hz=10_000, window=1024, hop=512)
pipe.set_source(el.WindowSource("vibration.npy", window=1024, hop=512))

@pipe.stage
def filter(window): ...            # one argument: receives the previous stage's output

@pipe.stage
def fft(x): ...

@pipe.stage(role="inference", device="gpu")
def classify(features): ...

result = pipe.run(iterations=2000)      # deadline defaults to the 51.2 ms hop period
print(result["latency"]["p99"], result["deadline"]["miss_ratio"],
      result["pack_metrics"]["real_time_factor"])
```

The same pipeline from the CLI: put it in a script with a
`build_pipeline()` function and run
`edgelens benchmark --pipeline my_pipeline.py`. See
[`examples/timeseries_vibration.py`](../examples/timeseries_vibration.py).
Scripts with the classic `build_stage_fns()` returning a dict still work,
and since v0.1.0 the dict may use **any stage names**.

`pace=True` (CLI `--pace` with `--period-ms`) releases iterations on a fixed
schedule and measures true response time, including queueing behind a slow
iteration.

## 3. Keep your loop, add three lines (observer mode)

```python
with el.trace("webcam-detector", pack="vision", target_fps=30, save="run.json") as t:
    while running:
        with t.iteration():
            with t.stage("capture"):    frame = cam.read()
            with t.stage("inference"):  dets = model(frame)
            with t.stage("postprocess"): out = nms(dets)
```

Then `edgelens diagnose run.json` and `edgelens report run.json` work exactly
as for a benchmark. See
[`examples/observer_vision_loop.py`](../examples/observer_vision_loop.py).

## Scenario packs

One generic engine; packs add a stage template, scenario metrics and
diagnosis rules in that scenario's language. Stages carry a **role**
(`input`, `preprocess`, `transfer`, `inference`, `postprocess`, `decision`,
`other`), inferred from the name or set explicitly, so the generic
diagnosis rules work for every pipeline.

| Pack         | For                                 | Adds                                                                                                                                                                                              |
| ------------ | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vision`     | camera / video / image               | classic 6-stage template; `target_fps` → frame budget as deadline; `BELOW_TARGET_FPS`                                                                                                           |
| `timeseries` | sensor, vibration, audio, IMU, CAN   | `sample_rate_hz`/`window`/`hop` → hop period as deadline; real-time factor, headroom, max sustainable rate; `CANNOT_KEEP_UP`, `TAIL_OVERRUNS_HOP`, `HIGH_OVERLAP_COST`; `WindowSource` replay |
| `custom`     | anything else                        | generic metrics and rules                                                                                                                                                                       |

Third-party packs: subclass `edgelens.packs.Pack` and call
`el.register_pack(MyPack())`.
