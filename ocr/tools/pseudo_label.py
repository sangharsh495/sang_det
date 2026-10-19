"""Label unlabeled video crops by agreement across frames of the same vehicle.

sang_det crop names look like  v0072_<slug>_t000142000_f0000142_01.jpg
(video id, timestamp in ms, frame, plate index). A vehicle stays in view for
several sampled frames, so:

1. read every crop with the current reader (format-decoded);
2. within one video, chain crops whose reads are within edit distance 2 and
   whose timestamps are <= GAP_MS apart -- one chain ~ one vehicle;
3. combine each chain with PlateVoter and accept it only if it has enough
   frames, high combined confidence, and most single-frame reads agree.

Every crop in an accepted chain gets the voted text. Rejected chains stay
unlabeled. Run with --eval to measure precision on crops that already have
human/AI labels before trusting it on the rest.

Usage:
    python pseudo_label.py MODEL.onnx CROP_DIR [CROP_DIR ...] --out labels.csv
        [--labels plate_labels.csv ...] [--eval]
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plate_format import PlateVoter, decode, is_valid  # noqa: E402

NAME_RE = re.compile(r'^(v\d+)_.*_t(\d+)_f(\d+)_(\d+)\.jpg$')
GAP_MS = 3000
MIN_FRAMES = 3
MIN_CONF = 0.5
MIN_AGREE = 0.6


def lev(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def read_all(model: str, paths: list[Path], batch: int = 256):
    sess = ort.InferenceSession(model)
    inp = sess.get_inputs()[0].name
    out = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = []
        for p in chunk:
            im = cv2.imread(str(p))
            im = np.zeros((64, 128, 3), np.uint8) if im is None else cv2.resize(
                cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (128, 64))
            imgs.append(im)
        for p, prob in zip(chunk, sess.run(None, {inp: np.stack(imgs)})[0]):
            out[p] = prob
    return out


def build_chains(paths, probs):
    by_video = collections.defaultdict(list)
    for p in paths:
        m = NAME_RE.match(p.name)
        if m:
            by_video[m.group(1)].append((int(m.group(2)), p, decode(probs[p])[0]))
    result = []
    for items in by_video.values():
        items.sort(key=lambda x: x[0])
        live: list[dict] = []
        for t, p, text in items:
            live_now = [c for c in live if t - c['t'] <= GAP_MS]
            for c in live:
                if c not in live_now:
                    result.append(c)
            live = live_now
            match = next((c for c in live if lev(text, c['text']) <= 2), None)
            if match is None:
                match = {'members': [], 't': t, 'text': text}
                live.append(match)
            match['members'].append((p, text))
            match['t'] = t
        result.extend(live)
    return result


def vote(chain, probs):
    voter = PlateVoter(min_frames=MIN_FRAMES)
    for p, _ in chain['members']:
        voter.add(probs[p])
    text, conf = voter.result()
    agree = sum(t == text for _, t in chain['members']) / len(chain['members'])
    ok = (voter.ready() and conf >= MIN_CONF and agree >= MIN_AGREE and is_valid(text))
    return text, conf, agree, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--labels', nargs='*', default=[])
    ap.add_argument('--eval', action='store_true')
    a = ap.parse_args()

    paths = sorted(p for d in a.dirs for p in Path(d).glob('*.jpg'))
    print(f'{len(paths):,} crops', flush=True)
    probs = read_all(a.model, paths)
    ch = build_chains(paths, probs)
    print(f'{len(ch):,} chains', flush=True)

    known = {}
    for f in a.labels:
        for r in csv.DictReader(open(f, encoding='utf-8')):
            t = re.sub(r'[^A-Z0-9]', '', (r.get('plate_text') or '').upper())
            if r.get('status') == 'labeled' and is_valid(t):
                known[r['filename']] = t

    rows, hit, miss = [], 0, 0
    for c in ch:
        text, conf, agree, ok = vote(c, probs)
        if not ok:
            continue
        for p, _ in c['members']:
            rows.append((p.name, text, f'{conf:.3f}', len(c['members'])))
            if p.name in known:
                hit += known[p.name] == text
                miss += known[p.name] != text
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['filename', 'plate_text', 'vote_conf', 'chain_len'])
        w.writerows(rows)
    print(f'accepted {len(rows):,} crops in '
          f'{sum(1 for c in ch if vote(c, probs)[3]):,} chains -> {a.out}')
    if a.eval and hit + miss:
        print(f'agreement with existing labels: {hit}/{hit + miss} = {hit / (hit + miss):.1%}')


if __name__ == '__main__':
    main()
