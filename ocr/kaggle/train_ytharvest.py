"""Harvest real plate crops from YouTube riding footage with sang_det's settings.

CPU only. One shard of queries per kernel (SHARD is set by build_harvest.sh).
For each video: sample 2 frames/s (sang_det default), detect plates with the
same detector and filters sang_det uses, crop with sang_det's 3 px padding,
read with v8 (Indian-format decoding), and save every crop that passes.

Labels are NOT final here. A later GPU pass reads the same crops with
Qwen2.5-VL; a crop is labelled only where Qwen and v8 agree.

Outputs in /kaggle/working:
    crops/<video>_<t>_<k>.jpg
    crops.csv   file, video, query, t, track, w, h, edge, det_conf, v8, v8_conf
Hard time budget so Kaggle's 12 h limit never discards the run.
"""
import csv
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SHARD = 'day1'
VIDEO_IDS = []          # set by build_channels.py: harvest these videos instead of searching
T0 = time.time()
BUDGET_S = 10.8 * 3600
WORK = Path('/kaggle/working')
QUERIES = {
    'day1': ['mumbai bike ride 4k', 'pune traffic ride 4k pov', 'bangalore traffic ride 4k',
             'hyderabad traffic drive 4k', 'chennai bike ride 4k pov', 'kolkata traffic drive 4k',
             'nagpur city ride 4k', 'indore city drive 4k'],
    'day2': ['delhi traffic drive 4k dashcam', 'jaipur city ride 4k', 'ahmedabad traffic ride 4k',
             'lucknow city drive 4k', 'kerala city ride 4k pov', 'goa scooter ride 4k',
             'chandigarh city drive 4k', 'surat city ride 4k'],
    'night': ['mumbai night ride 4k', 'bangalore night drive 4k', 'delhi night drive 4k',
              'pune night ride 4k pov', 'hyderabad night drive 4k', 'chennai night ride 4k',
              'india highway night drive dashcam', 'kolkata night drive 4k'],
    # Street-walking vlogs: slow camera, parked vehicles, sharp plates.
    'walk1': ['mumbai street walk 4k', 'delhi street walk 4k', 'old delhi walking tour 4k',
              'bangalore walking tour 4k', 'pune walking tour 4k', 'hyderabad street walk 4k',
              'chennai street walk 4k', 'kolkata street walk 4k'],
    'walk2': ['jaipur walking tour 4k', 'varanasi walking tour 4k', 'lucknow street walk 4k',
              'ahmedabad walking tour 4k', 'indore street walk 4k', 'goa walking tour 4k',
              'kochi walking tour 4k', 'chandigarh walking tour 4k'],
}.get(SHARD, [])
PER_QUERY = 6 if SHARD.startswith('walk') else 8
MAX_SECONDS = 2400 if SHARD.startswith(('walk', 'ch')) else 1200   # walks are long, keep 40 min
FPS = 2.0                              # sang_det sampling.fps
MAX_W = 1920                           # sang_det sampling.max_frame_width
IMGSZ, CONF, IOU = 960, 0.35, 0.45     # sang_det detector.*
MIN_W, MIN_H, AR_LO, AR_HI, MAX_AREA = 60, 20, 1.2, 8.0, 0.25   # sang_det filters.*
PAD = 3                                # sang_det filters.crop_padding_px
MIN_LAP, MIN_STD = 50.0, 18.0          # sang_det quality.*
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

urllib.request.urlretrieve(MODEL_URL, WORK / 'plate_v8.onnx')
reader = ort.InferenceSession(str(WORK / 'plate_v8.onnx'))
RIN = reader.get_inputs()[0].name

# Same weight choice as sang_det app/detector.py
HINTS = ('plate', 'license', 'licence', 'number_plate', 'numberplate', 'lp')
det_path = None
for repo in ('morsetechlab/yolov11-license-plate-detection', 'keremberke/yolov8n-license-plate'):
    try:
        files = [f for f in list_repo_files(repo) if f.endswith('.pt')]
        files.sort(key=lambda f: (0 if any(h in f.lower() for h in HINTS) else 1,
                                  0 if 'best' in f.lower() else 1,
                                  0 if 'n.' in f.lower() or 'nano' in f.lower() else 1, len(f)))
        det_path = hf_hub_download(repo, files[0])
        log(f'detector {repo}/{files[0]}')
        break
    except Exception as e:
        log(f'detector {repo} failed: {e}')
detector = YOLO(det_path)


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def good_quality(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var() >= MIN_LAP and g.std() >= MIN_STD


# ------------------------------------------------------------- video list
ids = [(v, SHARD) for v in VIDEO_IDS]
for q in QUERIES:
    out = subprocess.run(
        f'yt-dlp --flat-playlist --print "%(id)s %(duration)s" "ytsearch{PER_QUERY}:{q}"',
        shell=True, capture_output=True, text=True).stdout.split('\n')
    for line in out:
        parts = line.split()
        if len(parts) == 2 and parts[0] not in {i for i, _ in ids}:
            try:
                dur = float(parts[1])
            except ValueError:
                continue
            if dur >= 120:                 # skip Shorts
                ids.append((parts[0], q))
log(f'{len(ids)} videos queued')

cdir = WORK / 'crops'
cdir.mkdir(exist_ok=True)
vdir = WORK / 'videos'
vdir.mkdir(exist_ok=True)
fh = open(WORK / 'crops.csv', 'w', newline='')
wr = csv.writer(fh)
wr.writerow(['file', 'video', 'query', 't', 'track', 'w', 'h', 'edge', 'det_conf', 'v8', 'v8_conf'])
n_track = 0
n_crop = 0
for vid, q in ids:
    if time.time() - T0 > BUDGET_S:
        log('time budget reached')
        break
    subprocess.call(
        f'yt-dlp -q --no-warnings -f "bv*[height<=1080][ext=mp4]/b[height<=1080]" '
        f'--download-sections "*0-{MAX_SECONDS}" --max-filesize 900M '
        f'-o "{vdir}/{vid}.%(ext)s" "https://www.youtube.com/watch?v={vid}"', shell=True)
    files = list(vdir.glob(f'{vid}.*'))
    if not files:
        log(f'{vid}: download failed')
        continue
    cap = cv2.VideoCapture(str(files[0]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps / FPS)))
    idx = 0
    live = []
    before = n_crop
    while time.time() - T0 < BUDGET_S:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                H, W = frame.shape[:2]
                if W > MAX_W:
                    frame = cv2.resize(frame, (MAX_W, round(H * MAX_W / W)), interpolation=cv2.INTER_AREA)
                    H, W = frame.shape[:2]
                t = idx / fps
                res = detector.predict(frame, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False,
                                       max_det=20)[0]
                for k, (b, dc) in enumerate(zip(res.boxes.xyxy.cpu().numpy().astype(int),
                                                res.boxes.conf.cpu().numpy())):
                    x1, y1, x2, y2 = b
                    w, h = x2 - x1, y2 - y1
                    if w < MIN_W or h < MIN_H or not AR_LO <= w / max(h, 1) <= AR_HI \
                            or w * h > MAX_AREA * W * H:
                        continue
                    crop = frame[max(0, y1 - PAD):y2 + PAD, max(0, x1 - PAD):x2 + PAD]
                    if crop.size == 0 or not good_quality(crop):
                        continue
                    edge = int(x1 <= 2 or y1 <= 2 or x2 >= W - 3 or y2 >= H - 3)
                    im = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (128, 64))
                    text, conf = decode(reader.run(None, {RIN: im[None]})[0][0])
                    live = [tr for tr in live if t - tr[1] <= 2.0]
                    tr = next((tr for tr in live if lev(text, tr[2]) <= 2), None)
                    if tr is None:
                        tr = [n_track, t, text]
                        n_track += 1
                        live.append(tr)
                    tr[1], tr[2] = t, text
                    name = f'{vid}_{int(t * 10):06d}_{k}.jpg'
                    cv2.imwrite(str(cdir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    wr.writerow([name, vid, q, f'{t:.1f}', tr[0], w, h, edge, f'{dc:.3f}',
                                 text, f'{conf:.4f}'])
                    n_crop += 1
        idx += 1
    cap.release()
    for f in files:
        f.unlink()
    fh.flush()
    log(f'{vid}: {n_crop - before} crops (total {n_crop:,}, {n_track:,} tracks)')
fh.close()
log('done')
