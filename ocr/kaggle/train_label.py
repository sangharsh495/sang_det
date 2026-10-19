"""Label unlabeled street-video crops with NVIDIA Nemotron + Llama-3.2-90B (CPU notebook).

Keys are NOT in this code. Add them in the notebook: Add-ons -> Secrets ->
name NVIDIA_API_KEY, value = one or more nvapi-... keys separated by commas.

Inputs  : sang-det-zips (45,831 crops + plate_labels*.csv),
          indian-plate-nv-3cond (manifest of crops already used in training).
Output  : /kaggle/working/nemo_reads.csv, llama_reads.csv  (path, text). Written as it goes, so
          a run cut off at the time limit still keeps everything read so far.
"""
import base64
import csv
import glob
import itertools
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

T0 = time.time()
BUDGET_S = 11.3 * 3600
IN = Path('/kaggle/input')
URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
MODELS = {'nemo': 'nvidia/nemotron-3.5-super-vl-120b-a12b',
          'llama': 'meta/llama-3.2-90b-vision-instruct'}
PER, WORKERS = 4, 32
CELL_W, CELL_H = 320, 120
PROMPT = (
    'This image is a grid of {n} Indian vehicle number-plate crops, each labelled '
    'with a red index number in its top-left corner. Read the registration on '
    'each plate exactly as printed, letters and digits only, no spaces. Two-row '
    'plates: read top row then bottom row. If a plate is unreadable, give "". '
    'Reply with JSON only, mapping index to text, e.g. {{"1": "MH12AB1234", "2": ""}}.'
)

from kaggle_secrets import UserSecretsClient  # noqa: E402
KEYS = [k.strip() for k in UserSecretsClient().get_secret('NVIDIA_API_KEY').split(',') if k.strip()]
print(f'{len(KEYS)} keys loaded from Kaggle Secrets', flush=True)

# ------------------------------------------------------------ build crop list
sdz = next(p.parent for p in IN.rglob('plate_labels*.csv'))
man = next(IN.rglob('full_manifest.csv'))
crops = {}
for d in sdz.iterdir():
    inner = d / d.name
    if inner.is_dir():
        for f in inner.iterdir():
            crops[f.name] = f
used = {r['src_key'].split('|', 1)[1].split('/')[-1]
        for r in csv.DictReader(open(man)) if r['condition'] == 'day' and r['source'] == 'video'}
labels = {}
for f in glob.glob(str(sdz / 'plate_labels*.csv')):
    for r in csv.DictReader(open(f, encoding='utf-8')):
        labels[r['filename']] = r
todo = []
for name, p in sorted(crops.items()):
    if name in used:
        continue
    r = labels.get(name)
    text = re.sub(r'[^A-Z0-9]', '', ((r or {}).get('plate_text') or '').upper())
    if r is None or (r['status'] == 'labeled' and not is_valid(text)):
        todo.append(p)
print(f'{len(crops):,} crops, {len(todo):,} to read', flush=True)


# ------------------------------------------------------------ reading
def tile(paths):
    rows = (len(paths) + 1) // 2
    canvas = np.full((rows * CELL_H, 2 * CELL_W, 3), 255, np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        s = min((CELL_W - 8) / w, (CELL_H - 8) / h)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
        r, c = divmod(i, 2)
        y, x = r * CELL_H + 4, c * CELL_W + 4
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(canvas, str(i + 1), (c * CELL_W + 2, r * CELL_H + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


key_cycle = itertools.cycle(KEYS)
key_lock = threading.Lock()


def ask(model, paths):
    img = base64.b64encode(tile(paths)).decode()
    sysmsg = [{'role': 'system', 'content': '/no_think'}] if 'nemotron' in model else []
    body = {'model': model, 'max_tokens': 4000, 'temperature': 0.0, 'top_p': 1.0,
            'messages': sysmsg + [{'role': 'user', 'content': [
                             {'type': 'text', 'text': PROMPT.format(n=len(paths))},
                             {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img}'}}]}]}
    delay = 5
    for _ in range(6):
        with key_lock:
            key = next(key_cycle)
        try:
            r = requests.post(URL, json=body, timeout=180,
                              headers={'Authorization': f'Bearer {key}', 'Accept': 'application/json'})
        except requests.RequestException:
            time.sleep(delay); delay = min(delay * 2, 120)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay); delay = min(delay * 2, 120)
            continue
        if r.status_code != 200:
            print('HTTP', r.status_code, r.text[:150], flush=True)
            return {}
        m = re.search(r'\{.*\}', r.json()['choices'][0]['message'].get('content') or '', re.S)
        if not m:
            return {}
        try:
            raw = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
        out = {}
        for k, v in raw.items():
            try:
                out[int(k)] = re.sub(r'[^A-Z0-9]', '', str(v).upper())
            except ValueError:
                pass
        return out
    return {}


batches = [todo[i:i + PER] for i in range(0, len(todo), PER)]
files = {k: open(f'/kaggle/working/{k}_reads.csv', 'w', newline='') for k in MODELS}
writers = {k: csv.writer(f) for k, f in files.items()}
for w in writers.values():
    w.writerow(['path', 'text'])
lock = threading.Lock()
done = [0]
jobs = [(k, b) for b in batches for k in MODELS]      # both models interleaved


def run(job):
    k, batch = job
    if time.time() - T0 > BUDGET_S:
        return
    res = ask(MODELS[k], batch)
    with lock:
        for i, p in enumerate(batch, 1):
            if i in res:
                writers[k].writerow([str(p.relative_to(sdz)), res[i]])
        files[k].flush()
        done[0] += 1
        if done[0] % 200 == 0:
            print(f'{done[0]:,}/{len(jobs):,} requests  {(time.time() - T0) / 60:.0f} min', flush=True)


with ThreadPoolExecutor(WORKERS) as ex:
    list(ex.map(run, jobs))
for f in files.values():
    f.close()
print('finished', done[0], 'requests', flush=True)
