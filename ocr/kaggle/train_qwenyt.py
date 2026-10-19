"""Label YouTube harvest crops (plate-harvest-*) with Qwen2.5-VL-7B on Kaggle GPUs.

1. Score the reader on the 295-plate day benchmark (known answers) so we know
   how far to trust it.
2. Read up to 3 crops per track from the plate-harvest-* kernel outputs.

Output (written as it goes, so a cut-off run keeps its work):
    /kaggle/working/qwen_bench.csv   benchmark reads
    /kaggle/working/qwen_reads.csv   path, text
Hard time budget: stops starting new batches after 4.8 h.
"""
import csv
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
BUDGET_S = 5.5 * 3600
IN = Path('/kaggle/input')
WORK = Path('/kaggle/working')
MID = 'Qwen/Qwen2.5-VL-7B-Instruct'
BATCH = 16
PROMPT = ('Read the Indian vehicle registration number on this number plate. '
          'Reply with only the letters and digits, no spaces or punctuation. '
          'For a two-row plate give the top row then the bottom row. '
          'Ignore "IND" and any other words. If it cannot be read, reply NONE.')


def log(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


subprocess.call(f'{sys.executable} -m pip install -q -U "transformers>=4.51" accelerate', shell=True)
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

log(f'GPUs: {torch.cuda.device_count()}')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MID, torch_dtype=torch.float16, device_map='auto', attn_implementation='sdpa')
proc = AutoProcessor.from_pretrained(MID, min_pixels=28 * 28 * 16, max_pixels=28 * 28 * 160)
proc.tokenizer.padding_side = 'left'
TEMPLATE = proc.apply_chat_template(
    [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}],
    tokenize=False, add_generation_prompt=True)
log('model loaded')


def load(p):
    im = Image.open(p).convert('RGB')
    w, h = im.size
    s = 336 / max(w, 1)                      # small plates -> enlarge to ~336 px wide
    return im.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)


@torch.inference_mode()
def read(paths):
    imgs = [load(p) for p in paths]
    inputs = proc(text=[TEMPLATE] * len(imgs), images=imgs, return_tensors='pt',
                  padding=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    texts = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return [re.sub(r'[^A-Z0-9]', '', t.upper().replace('NONE', '')) for t in texts]


# ------------------------------------------------------------ 1. benchmark
bench = next(IN.rglob('labels_day_original.json')).parent   # fixed labels live beside it
lab = json.load(open(bench / 'labels_day.json'))
ok = n = 0
with open(WORK / 'qwen_bench.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['name', 'truth', 'qwen', 'qwen_fmt'])
    for i in range(0, len(lab), BATCH):
        chunk = lab[i:i + BATCH]
        for (name, truth), text in zip(chunk, read([bench / 'day' / nm for nm, _ in chunk])):
            fmt = text if is_valid(text) else ''
            w.writerow([name, truth, text, fmt])
            n += 1
            ok += text == truth
log(f'BENCHMARK day: {ok}/{n} = {ok / n:.1%} exact')

# ------------------------------------------------------------ 2. crop list
# Harvest crops: skip plates touching the frame edge (often cut off) and read
# at most the 3 largest crops of each track, so the GPU time covers more vehicles.
todo = []
for man in sorted(IN.rglob('crops.csv')):
    rows = [r for r in csv.DictReader(open(man)) if r['edge'] == '0']
    shard = man.parent.name
    by_track = {}
    for r in rows:
        by_track.setdefault(r['track'], []).append(r)
    for rs in by_track.values():
        for r in sorted(rs, key=lambda r: -int(r['w']) * int(r['h']))[:3]:
            p = man.parent / 'crops' / r['file']
            if p.exists():
                todo.append(p)
    log(f'{shard}: {len(rows):,} crops, {len(by_track):,} tracks')
todo.sort(key=lambda p: p.name)
log(f'{len(todo):,} crops to read')

# ------------------------------------------------------------ 3. read
with open(WORK / 'qwen_reads.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['path', 'text'])
    for i in range(0, len(todo), BATCH):
        if time.time() - T0 > BUDGET_S:
            log('time budget reached')
            break
        chunk = todo[i:i + BATCH]
        try:
            texts = read(chunk)
        except Exception as e:                      # one bad image must not stop the run
            log(f'batch {i} failed: {e}')
            continue
        for p, t in zip(chunk, texts):
            w.writerow([p.name, t])
        fh.flush()
        if (i // BATCH) % 100 == 0:
            log(f'{i + len(chunk):,}/{len(todo):,}')
log('done')
