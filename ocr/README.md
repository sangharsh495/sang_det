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

### v8 (R1 -> 150k synthetic -> real day/ambient/IR), Indian-format decoding

| Reader | Day | Day 160px | Day 70px | Amb | Amb 160 | Amb 70 | IR | IR 160 | IR 70 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 85.4 | 86.1 | 84.7 | 76.3 | 76.9 | 75.3 | 81.0 | 80.7 | 74.6 | 80.1 |
| **v8** | 84.4 | 86.4 | 86.8 | 76.9 | 78.0 | 79.0 | 83.1 | 82.7 | 81.7 | **82.1** |

Model: `models/plate_v8.onnx` (3 MB, cct-xs, same input as R1).

### New labels for v9 (no paid API)

27,757 unlabeled street-video crops were read by Qwen2.5-VL-7B (Kaggle GPU,
87.5% exact on the day benchmark), Nemotron (5.6k crops) and v8. A crop is kept
when two readers agree, no reader disagrees and the read is a valid 9-10
character plate: 7,814 crops / 4,023 plates (benchmark check of the rule: 224/225,
the one miss was a typo in the benchmark answer). Each is added as day,
ambient and IR versions -> `sangharshgedekar/indian-plate-extra-v9`.
`kaggle/train_v9.py` fine-tunes v8 on real + these.

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
