"""Time the plate reader on the Raspberry Pi (or any machine).

    pip install onnxruntime numpy
    python3 ocr/tools/pi_speed.py                 # times every plate_*.onnx in ocr/models

Prints milliseconds per plate for 1 and 4 threads, and for reading
4 plates in one call (all plates found in a frame), which is cheaper per plate.
"""
import glob
import os
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = sorted(glob.glob(os.path.join(HERE, '..', 'models', 'plate_*.onnx')))
x1 = np.random.default_rng(0).integers(0, 255, (1, 64, 128, 3), dtype=np.uint8)
x4 = np.repeat(x1, 4, axis=0)


def timed(sess, x, n=30):
    name = sess.get_inputs()[0].name
    for _ in range(3):
        sess.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(n):
        sess.run(None, {name: x})
    return (time.perf_counter() - t) / n * 1000


for path in MODELS:
    row = []
    for threads in (1, 4):
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(path, o, providers=['CPUExecutionProvider'])
        row.append(f'{threads} thread: {timed(s, x1):6.1f} ms')
        if threads == 4:
            try:
                row.append(f'4 plates/call: {timed(s, x4) / 4:6.1f} ms/plate')
            except Exception:
                row.append('4 plates/call: n/a (fixed batch 1)')
    print(f'{os.path.basename(path):24} ' + '   '.join(row))
