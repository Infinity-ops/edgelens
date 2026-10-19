"""Stand-in model for toy.stream.yaml: health from the vibration spread of the
last 10 steps (the window is [1, sequence, 5 stats x 3 columns], stat-major:
index 3 = vib_std). Replace it with your ONNX model (model: {onnx: ...})."""

import numpy as np


def predict(window):
    vib_std = float(np.mean(window[0, -10:, 3]))
    return float(np.clip(1.33 - 0.53 * vib_std, 0.0, 1.0))
