"""Real-world test of the v8 reader on YouTube moto-vlog / traffic footage.

CPU only. For each video: sample 1 frame/s, detect plates with the same
detector sang_det uses, read each crop with v8 (Indian-format decoding), and
group frames of one vehicle into a track. Per track we keep
  single : v8 reading of the sharpest/largest crop
  voted  : PlateVoter over every crop of the track
and render contact sheets so the readings can be checked by eye.

Outputs in /kaggle/working: tracks.csv, crops/, sheets/sheet_XX.jpg
"""
import csv
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

T0 = time.time()
WORK = Path('/kaggle/working')
QUERIES = [
    'ytsearch3:mumbai bike ride traffic 4k moto vlog',
    'ytsearch2:bangalore traffic ride 4k pov',
    'ytsearch2:delhi traffic drive 4k dashcam',
]
CLIP_SECONDS = 480
MODEL_URL = ('https://raw.githubusercontent.com/sangharsh495/sang_det/'
             'claude/affectionate-faraday-0y3l6v/ocr/models/plate_v8.onnx')


def log(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


subprocess.call(f'{sys.executable} -m pip install -q -U yt-dlp ultralytics onnxruntime', shell=True)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# ------------------------------------------------------------- models
urllib.request.urlretrieve(MODEL_URL, WORK / 'plate_v8.onnx')
reader = ort.InferenceSession(str(WORK / 'plate_v8.onnx'))
det_path = None
for repo in ('morsetechlab/yolov11-license-plate-detection', 'keremberke/yolov8n-license-plate'):
    try:
        pts = sorted(f for f in list_repo_files(repo) if f.endswith('.pt'))
        pick = next((f for f in pts if 'n' in Path(f).stem.lower()), pts[0])
        det_path = hf_hub_download(repo, pick)
        log(f'detector {repo}/{pick}')
        break
    except Exception as e:
        log(f'detector {repo} failed: {e}')
detector = YOLO(det_path)

# ------------------------------------------------------------- videos
vdir = WORK / 'videos'
vdir.mkdir(exist_ok=True)
for q in QUERIES:
    subprocess.call(
        f'yt-dlp -q --no-warnings -f "bv*[height<=1080][ext=mp4]/b[height<=1080]" '
        f'--download-sections "*0-{CLIP_SECONDS}" --max-filesize 600M '
        f'-o "{vdir}/%(id)s.%(ext)s" "{q}"', shell=True)
videos = sorted(vdir.glob('*'))
log(f'{len(videos)} videos downloaded: {[v.name for v in videos]}')
if not videos:
    raise SystemExit('no videos could be downloaded (YouTube may be blocking this machine)')


def read(crop):
    im = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (128, 64))
    return reader.run(None, {reader.get_inputs()[0].name: im[None]})[0][0]


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


# ------------------------------------------------------------- detect + read
cdir = WORK / 'crops'
cdir.mkdir(exist_ok=True)
tracks = []
for v in videos:
    cap = cv2.VideoCapture(str(v))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(round(fps))                      # 1 frame per second
    idx = 0
    live = []
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                t = idx / fps
                res = detector.predict(frame, imgsz=960, conf=0.35, verbose=False)[0]
                for b in res.boxes.xyxy.cpu().numpy().astype(int):
                    x1, y1, x2, y2 = b
                    w, h = x2 - x1, y2 - y1
                    if w < 40 or h < 12:
                        continue
                    px, py = int(w * 0.08), int(h * 0.15)
                    crop = frame[max(0, y1 - py):y2 + py, max(0, x1 - px):x2 + px]
                    prob = read(crop)
                    text, conf = decode(prob)
                    live = [tr for tr in live if t - tr['t_last'] <= 2.0]
                    tr = next((tr for tr in live if lev(text, tr['last_text']) <= 2), None)
                    if tr is None:
                        tr = {'id': len(tracks), 'video': v.stem, 't0': t, 'crops': []}
                        tracks.append(tr)
                        live.append(tr)
                    name = f'{tr["id"]:05d}_{len(tr["crops"]):02d}.jpg'
                    cv2.imwrite(str(cdir / name), crop)
                    tr['crops'].append((name, prob, w * h, text, conf))
                    tr['t_last'], tr['last_text'] = t, text
        idx += 1
    log(f'{v.name}: {len(tracks)} tracks so far')

# ------------------------------------------------------------- per-track readings
rows = []
for tr in tracks:
    best = max(tr['crops'], key=lambda c: c[2])
    voter = PlateVoter(min_frames=1)
    for _, prob, area, _, _ in tr['crops']:
        voter.add(prob, quality=min(1.0, area / 6000))
    voted, vconf = voter.result()
    rows.append([tr['id'], tr['video'], f'{tr["t0"]:.0f}', len(tr['crops']), best[0],
                 best[3], f'{best[4]:.3f}', voted, f'{vconf:.3f}'])
with open(WORK / 'tracks.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['track', 'video', 't', 'frames', 'best_crop', 'single', 'single_conf', 'voted', 'voted_conf'])
    w.writerows(rows)
log(f'{len(rows)} tracks, {sum(len(t["crops"]) for t in tracks)} crops')

# ------------------------------------------------------------- contact sheets
sdir = WORK / 'sheets'
sdir.mkdir(exist_ok=True)
tiles = []
for r in rows:
    im = cv2.resize(cv2.imread(str(cdir / r[4])), (300, 100))
    im = cv2.copyMakeBorder(im, 0, 22, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(im, f'#{r[0]} n{r[3]} S:{r[5]} V:{r[7]}', (2, 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
    tiles.append(im)
for s in range(0, len(tiles), 24):
    page = tiles[s:s + 24]
    while len(page) % 4:
        page.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(sdir / f'sheet_{s // 24:02d}.jpg'),
                np.vstack([np.hstack(page[i:i + 4]) for i in range(0, len(page), 4)]))
for v in videos:
    v.unlink()
log('done')
