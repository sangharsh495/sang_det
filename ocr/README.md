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
| v8 | 84.4 | 86.4 | 86.8 | 76.9 | 78.0 | 79.0 | 83.1 | 82.7 | 81.7 | 82.1 |
| **v9** (v8 + 7.7k agreed street crops) | 86.8 | 87.5 | 86.4 | 79.0 | 79.7 | 79.3 | 82.0 | 81.7 | 80.3 | **82.5** |

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

## Real-world test: YouTube riding footage (v8, `kaggle/train_yttest.py`)

7 Indian traffic / moto-vlog videos (Mumbai, Bangalore, Delhi), first 8 min each,
1 frame/s, sang_det's plate detector, v8 reader. 1,248 detections, checked by eye on
random samples:

* About 40% of detections are not plates (the vlogger's phone screen, signs, tail
  lights) and another ~35% are too blurry or small for a person to read.
  v8 gives all of these confidence <= 0.1.
* Every correct read had confidence >= 0.26. Keep reads with **confidence >= 0.2**:
  168 of 1,248 tracks pass, which is nearly every plate a person can read.
* Of 48 random confident reads, **41 were right (85%)**. 5 of the 7 misses were plates
  cut off at the frame edge; excluding those: 41/43 (95%). Dropping reads shorter
  than 9 characters removes most cut-off plates.
* At 1 frame/s most vehicles are seen once, so voting rarely applies; on the Pi
  (more frames per vehicle) PlateVoter has more to work with.

## Faster reading on the Pi

* `models/plate_v9_int8.onnx`: v9 with int8 matrix multiplies (static calibration on 600
  day/ambient/IR crops). 2.0 MB instead of 3.1 MB. Benchmark (built size): day 86.1%,
  ambient 80.3%, IR 83.1% (fp32 v9: 86.8 / 79.0 / 82.0), so no accuracy loss.
  On x86 it is ~11% faster; int8 usually gains more on ARM. Measure on the Pi:
  `python3 ocr/tools/pi_speed.py`.
* Full int8 (Conv too) was 1.8x faster but lost 3-4 points, so it is not used.
* Cheaper than a faster model: read all plates of a frame in one call, read each
  vehicle only until PlateVoter is confident, and skip crops that fail the blur check.

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
