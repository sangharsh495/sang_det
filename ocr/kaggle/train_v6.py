"""v6: one cct-xs reader for day / ambient night / 850 nm IR, aimed at 90%.

Starting point (plate-nv1-3cond R1, the deployed reader), re-scored on the
300-plate benchmark with corrected labels:

                 as labelled   fixed labels   fixed + format decoding
    day             77.3%         83.7%            85.4%
    ambient         69.0%         73.6%            76.3%
    ir              72.3%         77.3%            81.0%

What v6 changes and why:

* Labels. 3% of crops carried registrations no Indian plate can have
  (HH47.. for MH47.., XA41.. for KA41.., KAO3.. for KA03..). 974 labels were
  fixed where a confusable-character swap agreed with the reader, 1,377 crops
  dropped. 20 benchmark labels were fixed by eye and 5 unreadable crops
  removed (benchmark/labels_<cond>_original.json keeps the old set).
* IR in training. R1 trained on day + ambient only; v6 trains all three.
* Underfitting. R1 (lr 1e-4) and R4 (lr 5e-5, stopped after 16 min) never
  got near their own training data. v6 uses a higher lr and longer schedule.
* Low clarity. The library default augmentation barely blurs (sigma <= 0.5,
  downscale >= 0.75). v6 blurs, downscales to 30%, adds sensor noise and
  heavy JPEG, the failure cases seen in the benchmark errors. Night imaging
  itself is already in the data (ambient / ir), so only mild extra darkening.
* Decoding. Scores are reported raw and with Indian-format decoding
  (plate_format.decode), which is what the Pi should run.

Runs, back to back:
    A  warm start from R1,           lr 3e-4, <= 45 epochs
    B  public cct_xs_v2_global start, lr 1e-3, <= 70 epochs

The night benchmarks are TRANSFORMED DAYLIGHT PHOTOGRAPHS, not real night
photographs: an upper bound on field accuracy until real night crops exist.

`kaggle kernels push` uploads only this file, so ocr/kaggle/build.sh pastes
plate_format.py above it.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

T0 = time.time()
IN = Path('/kaggle/input')
WORK = Path('/kaggle/working')
BASE = 'https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates'
RESULTS = {}


def log(msg):
    line = f'[{(time.time() - T0) / 60:6.1f} min] {msg}'
    print(line, flush=True)
    with open(WORK / 'progress.txt', 'a') as fh:
        fh.write(line + '\n')


def sh(cmd):
    print(f'\n$ {cmd}', flush=True)
    return subprocess.call(cmd, shell=True)


smi = subprocess.run('nvidia-smi -L', shell=True, capture_output=True, text=True)
log(f'GPU: {smi.stdout.strip() if smi.returncode == 0 else "NONE"}')
if smi.returncode != 0:
    raise SystemExit('no GPU -- refusing to burn 12 h on CPU')

sh(f'{sys.executable} -m pip install -q "fast-plate-ocr[train,onnx]"')
import albumentations as A  # noqa: E402
import cv2  # noqa: E402
import onnxruntime as ort  # noqa: E402
import yaml  # noqa: E402

DATA = next(p.parent for p in IN.rglob('train_all.csv')).parent
log(f'data root {DATA}')


def locate(name):
    """Kaggle may unpack a/ to a/ or a/a/; return the folder holding images."""
    d = DATA / name
    inner = d / name
    return inner if inner.is_dir() else d


STAGE = WORK / 'stage'
STAGE.mkdir(exist_ok=True)
for cond in ('day', 'ambient', 'ir'):
    if not (STAGE / cond).exists():
        os.symlink(locate(cond), STAGE / cond, target_is_directory=True)
for f in ('train_all.csv', 'val_all.csv'):
    shutil.copy(DATA / 'manifests' / f, STAGE / f)
BENCH = DATA / 'benchmark'
R1 = DATA / 'r1'
log('staged')

# ---------------------------------------------------------------- augmentation
BLACK = (0, 0, 0)
aug = A.Compose([
    A.Affine(translate_percent=(-0.03, 0.03), scale=(0.8, 1.1), rotate=(-12, 12),
             shear=(-8, 8), border_mode=cv2.BORDER_CONSTANT, fill=BLACK, p=0.7),
    A.Perspective(scale=(0.02, 0.06), p=0.2),
    A.OneOf([
        A.GaussianBlur(sigma_limit=(0.5, 1.5), p=1),
        A.MotionBlur(blur_limit=(3, 9), p=1),
        A.Defocus(radius=(1, 3), p=1),
    ], p=0.35),
    A.Downscale(scale_range=(0.3, 0.9),
                interpolation_pair={'downscale': cv2.INTER_AREA, 'upscale': cv2.INTER_LINEAR},
                p=0.4),
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.2), contrast_limit=(-0.3, 0.2), p=0.4),
    A.RandomGamma(gamma_limit=(80, 180), p=0.15),
    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=1),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
        A.CLAHE(clip_limit=(1, 3), p=1),
    ], p=0.25),
    A.ToGray(p=0.05),
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.08),
    A.OneOf([
        A.GaussNoise(std_range=(0.02, 0.08), p=1),
        A.ISONoise(intensity=(0.05, 0.3), p=1),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1),
    ], p=0.3),
    A.ImageCompression(quality_range=(25, 85), p=0.3),
    A.OneOf([
        A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(2, 8),
                        hole_width_range=(2, 8), p=1),
        A.PixelDropout(dropout_prob=0.02, p=1),
    ], p=0.2),
])
AUG = WORK / 'aug_v6.yaml'
A.save(aug, str(AUG), data_format='yaml')
A.load(str(AUG), data_format='yaml')
log('augmentation saved')

# ------------------------------------------------------------------- configs
ARCH = WORK / 'model_config.yaml'
PCFG = WORK / 'plate_config.yaml'
shutil.copy(R1 / 'model_config.yaml', ARCH)
shutil.copy(R1 / 'plate_config.yaml', PCFG)
PRE = WORK / 'cct_xs_v2_global.keras'
urllib.request.urlretrieve(f'{BASE}/cct_xs_v2_global.keras', PRE)

RUNS = [  # tag, weights, lr, epochs, patience, stop by (h)
    ('A', R1 / 'best.keras', 3e-4, 45, 8, 4.5),
    ('B', PRE, 1e-3, 70, 10, 10.6),
]


# ----------------------------------------------------------------- benchmark
def benchmark(tag, onnx_path):
    sess = ort.InferenceSession(str(onnx_path))
    inp = sess.get_inputs()[0].name
    tmp = Path(tempfile.mkdtemp())
    res = {}
    for cond in ('day', 'ambient', 'ir'):
        for labset in ('fixed', 'original'):
            suffix = '' if labset == 'fixed' else '_original'
            lab = json.load(open(BENCH / f'labels_{cond}{suffix}.json'))
            for w in (0, 160, 70):
                raw = fmt = n = 0
                for name, truth in lab:
                    im = cv2.imread(str(BENCH / cond / name))
                    if im is None:
                        continue
                    if w and im.shape[1] > w:
                        im = cv2.resize(im, (w, max(8, round(im.shape[0] * w / im.shape[1]))),
                                        interpolation=cv2.INTER_AREA)
                    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (128, 64),
                                    interpolation=cv2.INTER_LINEAR)
                    prob = sess.run(None, {inp: im[None]})[0][0]
                    n += 1
                    raw += argmax_text(prob) == truth
                    fmt += decode(prob)[0] == truth
                key = f'{cond}@{w or "built"}/{labset}'
                res[key] = {'raw': raw / max(n, 1), 'format': fmt / max(n, 1), 'n': n}
                if labset == 'fixed' or w == 0:
                    log(f'{tag:3} {key:24} raw {raw / max(n, 1):6.1%}   '
                        f'format {fmt / max(n, 1):6.1%}')
    shutil.rmtree(tmp, ignore_errors=True)
    RESULTS[tag] = res
    json.dump(RESULTS, open(WORK / 'v6_results.json', 'w'), indent=2)
    return res



def run_budgeted(cmd, until_h):
    """Run cmd, but stop it at until_h hours after start so Kaggle's 12 h
    limit never discards the run. The best checkpoint saved so far is kept."""
    import signal
    print(f'\n$ {cmd}', flush=True)
    budget = until_h * 3600 - (time.time() - T0)
    proc = subprocess.Popen(cmd, shell=True, start_new_session=True)
    try:
        return proc.wait(timeout=max(300, budget))
    except subprocess.TimeoutExpired:
        log(f'time budget reached ({until_h} h) -- stopping, keeping best checkpoint')
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=120)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
        return -1

def train(tag, weights, lr, epochs, patience, until_h):
    out = WORK / f'run_{tag}'
    code = run_budgeted(
        f'{sys.executable} -m fast_plate_ocr.cli.train '
        f'--model-config-file {ARCH} --plate-config-file {PCFG} '
        f'--annotations {STAGE}/train_all.csv --val-annotations {STAGE}/val_all.csv '
        f'--augmentation-path {AUG} --weights-path "{weights}" '
        f'--epochs {epochs} --batch-size 128 --lr {lr} --warmup-fraction 0.03 '
        f'--label-smoothing 0.05 --use-ema --workers 8 --use-multiprocessing '
        f'--max-queue-size 32 --early-stopping-metric val_plate_acc '
        f'--early-stopping-patience {patience} --output-dir {out}', until_h)
    log(f'{tag} training exit {code}')
    ck = sorted(out.rglob('*.keras'))
    return next((p for p in ck if 'best' in p.name), ck[-1] if ck else None)


def export(tag, ckpt):
    dst = WORK / f'export_{tag}'
    dst.mkdir(exist_ok=True)
    sh(f'{sys.executable} -m fast_plate_ocr.cli.export --model "{ckpt}" '
       f'--plate-config-file {PCFG} --format onnx --onnx-input-dtype uint8 --save-dir {dst}')
    got = sorted(dst.glob('*.onnx'))
    return got[0] if got else None


r1_onnx = export('R1', R1 / 'best.keras')
if r1_onnx:
    benchmark('R1', r1_onnx)

for tag, weights, lr, epochs, patience, until_h in RUNS:
    log(f'=== run {tag}: lr {lr}, <= {epochs} epochs, from {Path(weights).name}')
    ckpt = train(tag, weights, lr, epochs, patience, until_h)
    onnx = export(tag, ckpt) if ckpt else None
    if onnx:
        benchmark(tag, onnx)

cells = [f'{c}@{w}/fixed' for c in ('day', 'ambient', 'ir') for w in ('built', '160', '70')]
for tag, res in RESULTS.items():
    log(f'{tag:3} mean over 9 cells (fixed labels): raw '
        f'{sum(res[k]["raw"] for k in cells) / 9:.1%}  format '
        f'{sum(res[k]["format"] for k in cells) / 9:.1%}')
shutil.rmtree(STAGE, ignore_errors=True)
log('done')
