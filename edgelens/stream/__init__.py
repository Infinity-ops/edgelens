"""
edgelens.stream
-----------------
Windowed time-series inference, measured end to end (v0.2 Layer 4).

A sensor model does not see a sample: it sees a WINDOW of preprocessed
history, and its answer is smoothed and thresholded before anyone acts on
it. The model can answer in 2 ms while the decision arrives minutes after the
event. This package describes such a pipeline once (a *stream spec*) and
measures all of it:

    edgelens stream init | check | fit-scaler     the spec
    edgelens replay   spec.yaml data/*.csv        run recorded data through it
    edgelens detect   replay results              decision latency, alarm quality
    edgelens envelope fit | check                 the data the model was trained on
    edgelens perturb  spec.yaml data.csv          data faults -> robustness curves
    edgelens parity   spec.yaml data.csv          train/serve feature parity

Nothing here knows about a particular robot: column names, rates, window
sizes, the scaler, the model and the alarm rule all come from the spec.
docs/stream.md walks through a complete example.
"""

from .spec import SpecError, load_spec  # noqa: F401
