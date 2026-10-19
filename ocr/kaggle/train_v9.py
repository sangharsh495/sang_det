"""v9: start from v8 and fine-tune on real crops plus ~7.8k newly labelled street crops.

New labels (indian-plate-extra-v9): unlabeled sang_det video crops read by
Qwen2.5-VL-7B, Nemotron and v8; a crop is kept only when two readers agree,
no reader disagrees, and the read is a valid 9-10 character Indian plate.
On the 295-plate benchmark this agreement rule was right 224/225 times (the
one miss was a typo in the benchmark answer). Each crop is added as day,
ambient-night and IR versions (night_transform.py). Plates in the benchmark
or validation set are removed.

v8 stage 2 had already plateaued by epoch ~22, so v9 is a single fine-tune
stage at a lower learning rate. Time budget 5.2 h.

plate_format.py is pasted above this file by ocr/kaggle/build.sh.
"""
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

T0 = time.time()
IN = Path('/kaggle/input')
WORK = Path('/kaggle/working')
BASE = 'https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates'
RESULTS = {}
SYN_REPLAY = 15000


def log(msg):
    line = f'[{(time.time() - T0) / 60:6.1f} min] {msg}'
    print(line, flush=True)
    with open(WORK / 'progress.txt', 'a') as fh:
        fh.write(line + '\n')


def sh(cmd):
    print(f'\n$ {cmd}', flush=True)
    return subprocess.call(cmd, shell=True)


def shallow_find(name, depth=4):
    """Find a file by name without walking 400k images."""
    frontier = [(IN, 0)]
    while frontier:
        d, k = frontier.pop(0)
        hit = d / name
        if hit.exists():
            return hit
        if k < depth:
            try:
                frontier += [(c, k + 1) for c in d.iterdir()
                             if c.is_dir() and c.name not in {'images', 'detection', 'day',
                                                              'ambient', 'ir', 'train', 'val',
                                                              'test'}]
            except OSError:
                pass
    raise SystemExit(f'cannot find {name} under /kaggle/input')


smi = subprocess.run('nvidia-smi -L', shell=True, capture_output=True, text=True)
log(f'GPU: {smi.stdout.strip() if smi.returncode == 0 else "NONE"}')
if smi.returncode != 0:
    raise SystemExit('no GPU')

sh(f'{sys.executable} -m pip install -q "fast-plate-ocr[train,onnx]"')
import albumentations as A  # noqa: E402
import cv2  # noqa: E402
import onnxruntime as ort  # noqa: E402
import yaml  # noqa: E402

# ------------------------------------------------------------------ locate
REAL = shallow_find('labels_day_original.json').parent.parent  # indian-plate-3cond-v6
SYN = shallow_find('ocr_labels_v3.csv').parent   # train-datasets
EXTRA = shallow_find('extra_train.csv').parent    # indian-plate-extra-v9
V8 = sorted(p for pat in ('*/run_S2/*/best.keras', '*/*/run_S2/*/best.keras', '*/*/*/run_S2/*/best.keras')
            for p in IN.glob(pat))
if not V8:
    raise SystemExit('v8 best.keras not found (add kernel source plate-ocr-v8)')
V8 = V8[0]
log(f'extra {EXTRA}\nv8 {V8}')
log(f'real {REAL}\nsyn  {SYN}')
SYN_SETS = [  # label csv, folder holding ocr/
    ('ocr_labels.csv', 'ocr_crops'),
    ('ocr_labels_part2.csv', '11_ocr_plate_crops_part2_36349'),
    ('ocr_labels_part3.csv', '15_ocr_plate_crops_part3_1850'),
    ('ocr_labels_v3.csv', '21_ocr_plate_crops_v3_89355'),
]


def locate(base, name):
    d = base / name
    return d / name if (d / name).is_dir() else d


STAGE = WORK / 'stage'
STAGE.mkdir(exist_ok=True)
for cond in ('day', 'ambient', 'ir'):
    if not (STAGE / cond).exists():
        os.symlink(locate(REAL, cond), STAGE / cond, target_is_directory=True)
    if not (STAGE / f'x_{cond}').exists():
        os.symlink(locate(EXTRA, cond), STAGE / f'x_{cond}', target_is_directory=True)
syn_rows = {'train': [], 'val': []}
for i, (lab, folder) in enumerate(SYN_SETS):
    root = SYN / folder / 'ocr'
    if not root.is_dir():
        log(f'MISSING {root} -- skipped')
        continue
    link = STAGE / f'syn{i}'
    if not link.exists():
        os.symlink(root, link, target_is_directory=True)
    kept = 0
    for r in csv.DictReader(open(SYN / lab, encoding='utf-8')):
        text = r['plate_text'].strip().upper()
        if r['split'] not in syn_rows or len(text) > 10 or not text.isalnum():
            continue
        syn_rows[r['split']].append((f'syn{i}/{r["image_path"]}', text))
        kept += 1
    log(f'{folder}: {kept:,} rows')

# Drop rows whose image is missing (some archives were partial uploads).
for split in syn_rows:
    before = len(syn_rows[split])
    syn_rows[split] = [r for r in syn_rows[split] if (STAGE / r[0]).exists()]
    log(f'synthetic {split}: {len(syn_rows[split]):,} of {before:,} images present')

real_train = list(csv.reader(open(REAL / 'manifests' / 'train_all.csv')))[1:]
real_val = list(csv.reader(open(REAL / 'manifests' / 'val_all.csv')))[1:]
extra = [('x_' + p, t) for p, t in list(csv.reader(open(EXTRA / 'extra_train.csv')))[1:]]
extra = [r for r in extra if (STAGE / r[0]).exists()]
log(f'real train {len(real_train):,}  extra {len(extra):,}')
rng = random.Random(7)
syn_val = rng.sample(syn_rows['val'], min(4000, len(syn_rows['val'])))


def write(name, rows):
    with open(STAGE / name, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['image_path', 'plate_text'])
        w.writerows(rows)
    log(f'{name}: {len(rows):,}')


write('s2_train.csv', real_train + extra + rng.sample(syn_rows['train'],
                                               min(SYN_REPLAY, len(syn_rows['train']))))
write('s2_val.csv', real_val)

# ---------------------------------------------------------------- configs
R1 = REAL / 'r1'
ARCH = WORK / 'model_config.yaml'
PCFG = WORK / 'plate_config.yaml'
shutil.copy(R1 / 'model_config.yaml', ARCH)
shutil.copy(R1 / 'plate_config.yaml', PCFG)

BLACK = (0, 0, 0)
aug = A.Compose([
    A.Affine(translate_percent=(-0.03, 0.03), scale=(0.8, 1.1), rotate=(-12, 12),
             shear=(-8, 8), border_mode=cv2.BORDER_CONSTANT, fill=BLACK, p=0.7),
    A.Perspective(scale=(0.02, 0.06), p=0.1),
    A.OneOf([
        A.GaussianBlur(sigma_limit=(0.5, 1.5), p=1),
        A.MotionBlur(blur_limit=(3, 9), p=1),
        ], p=0.3),
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
    A.ToGray(p=0.1),
    A.OneOf([
        A.GaussNoise(std_range=(0.02, 0.08), p=1),
        A.ISONoise(intensity=(0.05, 0.3), p=1),
    ], p=0.3),
    A.ImageCompression(quality_range=(25, 85), p=0.3),
    A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(2, 8),
                    hole_width_range=(2, 8), p=0.1),
])
AUG = WORK / 'aug_v8.yaml'
A.save(aug, str(AUG), data_format='yaml')
LIGHT = WORK / 'aug_light.yaml'  # synthetic crops are already degraded
A.save(A.Compose([
    A.Affine(translate_percent=(-0.02, 0.02), scale=(0.85, 1.05), rotate=(-8, 8),
             border_mode=cv2.BORDER_CONSTANT, fill=BLACK, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
    A.ToGray(p=0.15),
]), str(LIGHT), data_format='yaml')


# -------------------------------------------------------------- helpers

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

def train(tag, weights, train_csv, val_csv, aug_path, lr, epochs, patience, until_h):
    out = WORK / f'run_{tag}'
    code = run_budgeted(
        f'{sys.executable} -m fast_plate_ocr.cli.train '
        f'--model-config-file {ARCH} --plate-config-file {PCFG} '
        f'--annotations {STAGE}/{train_csv} --val-annotations {STAGE}/{val_csv} '
        f'--augmentation-path {aug_path} --weights-path "{weights}" '
        f'--epochs {epochs} --batch-size 128 --lr {lr} --warmup-fraction 0.03 '
        f'--label-smoothing 0.05 --use-ema --workers 4 --use-multiprocessing '
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


def benchmark(tag, onnx_path):
    sess = ort.InferenceSession(str(onnx_path))
    inp = sess.get_inputs()[0].name
    bench = REAL / 'benchmark'
    res = {}
    for cond in ('day', 'ambient', 'ir'):
        lab = json.load(open(bench / f'labels_{cond}.json'))
        for w in (0, 160, 70):
            raw = fmt = n = 0
            for name, truth in lab:
                im = cv2.imread(str(bench / cond / name))
                if im is None:
                    continue
                if w and im.shape[1] > w:
                    im = cv2.resize(im, (w, max(8, round(im.shape[0] * w / im.shape[1]))),
                                    interpolation=cv2.INTER_AREA)
                im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (128, 64))
                prob = sess.run(None, {inp: im[None]})[0][0]
                n += 1
                raw += argmax_text(prob) == truth
                fmt += decode(prob)[0] == truth
            key = f'{cond}@{w or "built"}'
            res[key] = {'raw': raw / max(n, 1), 'format': fmt / max(n, 1), 'n': n}
            log(f'{tag:4} {key:14} raw {raw / max(n, 1):6.1%}   format {fmt / max(n, 1):6.1%}')
    RESULTS[tag] = res
    json.dump(RESULTS, open(WORK / 'v9_results.json', 'w'), indent=2)


# ------------------------------------------------------------------ run
log('=== v8 reference')
benchmark('v8', export('v8', V8))
log('=== v9: fine-tune from v8 on real + new labels (+ synthetic replay)')
s2 = train('v9', V8, 's2_train.csv', 's2_val.csv', AUG, 2e-4, 25, 5, 5.2)
if s2:
    benchmark('v9', export('v9', s2))

cells = [f'{c}@{w}' for c in ('day', 'ambient', 'ir') for w in ('built', '160', '70')]
for tag, res in RESULTS.items():
    log(f'{tag:4} mean over 9 cells: raw {sum(res[k]["raw"] for k in cells) / 9:.1%}  '
        f'format {sum(res[k]["format"] for k in cells) / 9:.1%}')
shutil.rmtree(STAGE, ignore_errors=True)
log('done')
