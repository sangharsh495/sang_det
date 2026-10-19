"""Read plate crops with NVIDIA-hosted vision models, many crops per request.

Crops are tiled into a numbered grid (default 8 per image), so one API call
reads 8 plates. Keys are read one per line from a file (never from the repo)
and used round-robin across parallel workers; on HTTP 429 a worker backs off.

    python nv_label.py --keys KEYFILE --model nvidia/nemotron-3.5-super-vl-120b-a12b \
        --out reads.csv CROP [CROP ...]          # or --list file_of_paths.txt

Output CSV: path, text (A-Z0-9 only, '' when the model says unreadable).
Already-read paths in --out are skipped, so the run can be resumed.
"""
from __future__ import annotations

import argparse
import base64
import csv
import itertools
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
CELL_W, CELL_H = 320, 120
NO_THINK = False
PROMPT = (
    'This image is a grid of {n} Indian vehicle number-plate crops, each labelled '
    'with a red index number in its top-left corner. Read the registration on '
    'each plate exactly as printed, letters and digits only, no spaces. Two-row '
    'plates: read top row then bottom row. If a plate is unreadable, give "". '
    'Reply with JSON only, mapping index to text, e.g. {{"1": "MH12AB1234", "2": ""}}.'
)


def tile(paths: list[Path], cols: int = 2) -> bytes:
    rows = (len(paths) + cols - 1) // cols
    canvas = np.full((rows * CELL_H, cols * CELL_W, 3), 255, np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        s = min((CELL_W - 8) / w, (CELL_H - 8) / h)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
        r, c = divmod(i, cols)
        y, x = r * CELL_H + 4, c * CELL_W + 4
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(canvas, str(i + 1), (c * CELL_W + 2, r * CELL_H + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    ok, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


class Keys:
    def __init__(self, keys):
        self._it = itertools.cycle(keys)
        self._lock = threading.Lock()

    def next(self):
        with self._lock:
            return next(self._it)


def ask(keys: Keys, model: str, paths: list[Path], tries: int = 6) -> dict[int, str]:
    img = base64.b64encode(tile(paths)).decode()
    body = {
        'model': model,
        'messages': ([{'role': 'system', 'content': '/no_think'}] if NO_THINK else []) + [{'role': 'user', 'content': [
            {'type': 'text', 'text': PROMPT.format(n=len(paths))},
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img}'}},
        ]}],
        'max_tokens': 4000, 'temperature': 0.0, 'top_p': 1.0,
    }
    delay = 5
    for _ in range(tries):
        try:
            r = requests.post(URL, json=body, timeout=180,
                              headers={'Authorization': f'Bearer {keys.next()}',
                                       'Accept': 'application/json'})
        except requests.RequestException:
            time.sleep(delay); delay *= 2
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay); delay = min(delay * 2, 120)
            continue
        if r.status_code != 200:
            print(f'HTTP {r.status_code}: {r.text[:200]}', file=sys.stderr)
            return {}
        msg = r.json()['choices'][0]['message']
        text = msg.get('content') or ''
        m = re.search(r'\{.*\}', text, re.S)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('crops', nargs='*')
    ap.add_argument('--list')
    ap.add_argument('--keys', required=True)
    ap.add_argument('--model', default='nvidia/nemotron-3.5-super-vl-120b-a12b')
    ap.add_argument('--per', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-think', action='store_true', help='ask reasoning models to answer directly')
    a = ap.parse_args()
    global NO_THINK
    NO_THINK = a.no_think

    paths = [Path(p) for p in a.crops]
    if a.list:
        paths += [Path(l.strip()) for l in open(a.list) if l.strip()]
    done = set()
    if Path(a.out).exists():
        done = {r['path'] for r in csv.DictReader(open(a.out))}
    paths = [p for p in paths if str(p) not in done]
    keys = Keys([k.strip() for k in open(a.keys) if k.strip()])
    batches = [paths[i:i + a.per] for i in range(0, len(paths), a.per)]
    print(f'{len(paths):,} crops in {len(batches):,} requests', flush=True)

    new = not Path(a.out).exists()
    fh = open(a.out, 'a', newline='')
    w = csv.writer(fh)
    if new:
        w.writerow(['path', 'text'])
    lock = threading.Lock()
    t0, n = time.time(), [0]

    def run(batch):
        res = ask(keys, a.model, batch)
        if not res:
            return
        with lock:
            for i, p in enumerate(batch, 1):
                if i in res:
                    w.writerow([str(p), res[i]])
            fh.flush()
            n[0] += 1
            if n[0] % 25 == 0:
                print(f'{n[0]}/{len(batches)} requests, {time.time() - t0:.0f}s', flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, batches))
    fh.close()


if __name__ == '__main__':
    main()
