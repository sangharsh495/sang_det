# Plate OCR (Indian plates, day / night / IR)

Reader: `fast-plate-ocr` cct-xs (fits the Pi 3B+ budget). Training runs on Kaggle.

## Where accuracy stood (300-plate benchmark, per condition)

| Reader | Labels | Decoding | Day | Ambient | IR |
|---|---|---|---|---|---|
| R1 (deployed) | original | argmax | 77.3% | 69.0% | 72.3% |
| R1 | fixed | argmax | 83.7% | 73.6% | 77.3% |
| R1 | fixed | **Indian format** | **85.4%** | **76.3%** | **81.0%** |

* **Fixed labels**: 20 benchmark labels were wrong (e.g. `UP72BTO82O` for
  `UP72BT0820`, `MP09S6003` for `MP09SS6003`). 5 unreadable crops were removed.
  The night sets are transforms of the same 300 crops, so they carried the same errors.
* **Indian format decoding** (`plate_format.decode`): picks the most probable
  string that is a valid registration instead of a per-slot argmax. No
  retraining, about 0.1 ms per plate.

Night benchmarks are transformed daylight photos, so they are an upper bound until real night crops are collected.

## Files

* `plate_format.py`: `decode()` (format-constrained reading) and `PlateVoter`
  (combines several frames of one tracked vehicle before reading). Needs only numpy.
* `kaggle/train_v6.py`: v6 training on cleaned labels, all three conditions,
  stronger low-clarity augmentation, and a warm start from R1. `kaggle/build.sh v6 --push`.
* `kaggle/train_v7.py`: pre-train on ~175k synthetic crops (train-datasets), then fine-tune on real.
* `tools/pseudo_label.py`: label unlabeled video crops by agreement across frames.
* `tools/label_audit.py`: flags and fixes labels that cannot be Indian registrations.

## Using it on the Pi

```python
from plate_format import decode, PlateVoter
prob = session.run(None, {'input': crop_128x64_uint8[None]})[0][0]   # (10, 37)
text, conf = decode(prob)

# Per tracked vehicle, across frames:
voter = PlateVoter(min_frames=3)
voter.add(prob, quality=sharpness)      # each frame
if voter.ready(): text, conf = voter.result()
```

Use the raw probability output of the ONNX model, not `LicensePlateRecognizer.run()`,
which returns already-decoded text.
