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

### Real inputs, accuracy and threads

Synthetic inputs time a model; they say nothing about what it answers, and
an accuracy measured on a workstation says nothing about the FP16 /
TensorRT / quantised build on the device. Give it data you already prepared,
and its labels:

```bash
# image classifier: 500 preprocessed images, class index per image
edgelens benchmark --model mobilenet.onnx --inputs val_images.npy --labels val_classes.npy \
    --iterations 500 --save mobilenet.json
edgelens validate mobilenet.json --p99-ms 20 --min-accuracy 0.70

# time-series model: sensor windows in a raw float32 file, a target per window
edgelens benchmark --model lstm.onnx --inputs windows.bin --labels targets.bin \
    --threads 4 --inputs-order spread --iterations 300 --save lstm.json
edgelens validate lstm.json --max-rmse 0.2
```

| Option | Meaning |
| --- | --- |
| `--inputs` | `.npy` (memory-mapped), `.npz` (one array per input name) or raw little-endian `.bin/.raw/.f32` in the model's input dtype. One sample per iteration, first dimension = sample; `[N, 3, 224, 224]` feeds a `[1, 3, 224, 224]` input. Fed **as given**: resize and normalise the data the way the model was trained. The file's SHA-256 is in the result, so `compare` can tell a data change from a setup change |
| `--labels` | ground truth, one row per sample (`.npy`, raw float32 `.bin`, `.csv/.txt`), scored against the model's **first** output (see below) |
| `--inputs-order` | `sequential` (file order: soaks, replaying a recording) or `spread` (a fixed stride that visits every sample once per pass, so a short run already covers the whole data set) |
| `--threads` | onnxruntime intra-op threads (default: one per core, all busy). Recorded in the result; fewer threads often keep the latency and free CPU for the rest of the system |

What the labels give, decided from their shape and the model's first output:

| Labels | Model output | Result | Requirement |
| --- | --- | --- | --- |
| `[N]` values | 1 value | RMSE, MAE, worst error; RMSE per label level when there are few distinct labels | `max_rmse` |
| `[N]` of 0 / 1 | 1 score | the above plus accuracy, precision, recall, false alarms and misses at > 0.5 | `max_rmse`, `min_accuracy` |
| `[N, k]` values | k values | RMSE overall and per output | `max_rmse` |
| `[N]` class indices | C class scores | top-1, top-5 (C > 5) and per-class accuracy | `min_accuracy` (top-1) |

Labels that can't fit the model (3 values per sample for a 1-value output,
fractional "classes") are refused **before** the run. The result says whether
the run made a **full pass** over the data; an accuracy over 200 of 2000
samples is printed with a warning. Detection and segmentation metrics (mAP,
IoU) are not computed: use `--pipeline` with your own post-processing.

The same options work for `periodic`, `soak` and `contract run`; `sweep`
takes `--labels` and `--threads` and scores every backend against the labels.

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
