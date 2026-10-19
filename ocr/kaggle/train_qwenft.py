"""Fine-tune Qwen2.5-VL-3B (LoRA) to read Indian plates, then use it as the teacher.

Why: zero-shot Qwen-7B reads 87.5% of the day benchmark, which caps how good its
labels can be. A small VLM trained on our 73k labelled real crops should read
plates far better and label the crops the small Pi reader cannot.

1. Zero-shot 3B score on the day benchmark (reference).
2. LoRA fine-tune on ~24k real crops (day / ambient / IR, benchmark plates
   removed) + the agreed extra-v9 day crops. fp16 on one T4, budget 2.6 h.
3. Score on the day / ambient / IR benchmarks -> qwenft_bench.csv.
4. Read the 27.7k unlabeled sang-det-zips crops -> qwenft_reads.csv.
The LoRA adapter is saved in /kaggle/working/lora for later labeling runs.

plate_format.py is pasted above this file by ocr/kaggle/build.sh.
"""
import csv
import glob
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
TRAIN_UNTIL_H = 2.6
BUDGET_H = 5.0
IN = Path('/kaggle/input')
WORK = Path('/kaggle/working')
MID = 'Qwen/Qwen2.5-VL-3B-Instruct'
N_TRAIN = 24000
BATCH, LR = 8, 1e-4
PROMPT = ('Read the Indian vehicle registration number on this number plate. '
          'Reply with only the letters and digits, no spaces or punctuation. '
          'For a two-row plate give the top row then the bottom row. '
          'Ignore "IND" and any other words. If it cannot be read, reply NONE.')


def log(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


subprocess.call(f'{sys.executable} -m pip install -q -U "transformers>=4.51" accelerate peft', shell=True)
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MID, torch_dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa')
proc = AutoProcessor.from_pretrained(MID, min_pixels=28 * 28 * 16, max_pixels=28 * 28 * 160)
tok = proc.tokenizer
USER = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
TEMPLATE = proc.apply_chat_template(USER, tokenize=False, add_generation_prompt=True)
log('model loaded')


def load(p):
    im = Image.open(p).convert('RGB')
    w, h = im.size
    s = 336 / max(w, 1)
    return im.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)


@torch.inference_mode()
def read(paths):
    tok.padding_side = 'left'
    imgs = [load(p) for p in paths]
    x = proc(text=[TEMPLATE] * len(imgs), images=imgs, return_tensors='pt', padding=True).to(0)
    out = model.generate(**x, max_new_tokens=16, do_sample=False)
    texts = proc.batch_decode(out[:, x['input_ids'].shape[1]:], skip_special_tokens=True)
    return [re.sub(r'[^A-Z0-9]', '', t.upper().replace('NONE', '')) for t in texts]


# ------------------------------------------------------------ data
bench = next(IN.rglob('labels_day_original.json')).parent
REAL = bench.parent


def locate(base, name):
    d = base / name
    return d / name if (d / name).is_dir() else d


def score(tag):
    model.eval()
    res = {}
    with open(WORK / f'{tag}_bench.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['cond', 'name', 'truth', 'read'])
        for cond in ('day', 'ambient', 'ir'):
            lab = json.load(open(bench / f'labels_{cond}.json'))
            ok = 0
            for i in range(0, len(lab), 16):
                chunk = lab[i:i + 16]
                for (name, truth), t in zip(chunk, read([bench / cond / n for n, _ in chunk])):
                    w.writerow([cond, name, truth, t])
                    ok += t == truth
            res[cond] = ok / len(lab)
            log(f'{tag} {cond}: {ok}/{len(lab)} = {ok / len(lab):.1%}')
            if tag == 'zeroshot':
                break                       # day is enough for the reference
    return res


results = {'zeroshot': score('zeroshot')}

bench_plates = {t for c in ('day', 'ambient', 'ir') for _, t in json.load(open(bench / f'labels_{c}.json'))}
rows = [(locate(REAL, p.split('/')[0]) / p.split('/', 1)[1], t)
        for p, t in list(csv.reader(open(REAL / 'manifests' / 'train_all.csv')))[1:]]
rng = random.Random(11)
rows = [r for r in rows if r[1] not in bench_plates and len(r[1]) <= 10]
rows = rng.sample(rows, min(N_TRAIN, len(rows)))
extra = list(IN.rglob('extra_train.csv'))
if extra:
    eroot = extra[0].parent
    rows += [(locate(eroot, 'day') / p.split('/', 1)[1], t)
             for p, t in list(csv.reader(open(extra[0])))[1:]
             if p.startswith('day/') and t not in bench_plates and len(t) <= 10]
rows = [r for r in rows if r[0].exists()]
rng.shuffle(rows)
log(f'{len(rows):,} training crops')

# ------------------------------------------------------------ LoRA training
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
for p in model.parameters():                 # trainable LoRA weights in fp32
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
scaler = torch.amp.GradScaler('cuda')
steps = len(rows) // BATCH
warm = 50


def lr_at(s):
    if s < warm:
        return LR * (s + 1) / warm
    return LR * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))


model.train()
tok.padding_side = 'right'
done_steps = 0
run = 0.0
for s in range(steps):
    if time.time() - T0 > TRAIN_UNTIL_H * 3600:
        log('training time budget reached')
        break
    chunk = rows[s * BATCH:(s + 1) * BATCH]
    imgs = [load(p) for p, _ in chunk]
    texts = [proc.apply_chat_template(
        USER + [{'role': 'assistant', 'content': [{'type': 'text', 'text': t}]}],
        tokenize=False, add_generation_prompt=False) for _, t in chunk]
    x = proc(text=texts, images=imgs, return_tensors='pt', padding=True).to(0)
    labels = torch.full_like(x['input_ids'], -100)
    for i, (_, t) in enumerate(chunk):
        n = int(x['attention_mask'][i].sum())
        a = len(tok(t + '<|im_end|>\n', add_special_tokens=False)['input_ids'])
        labels[i, n - a:n] = x['input_ids'][i, n - a:n]
    for g in opt.param_groups:
        g['lr'] = lr_at(s)
    with torch.autocast('cuda', dtype=torch.float16):
        loss = model(**x, labels=labels).loss
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    run = 0.98 * run + 0.02 * loss.item() if s else loss.item()
    done_steps = s + 1
    if s % 100 == 0:
        log(f'step {s}/{steps} loss {run:.4f}')
log(f'trained {done_steps} steps')
model.save_pretrained(WORK / 'lora')
model.gradient_checkpointing_disable()
model.config.use_cache = True

results['finetuned'] = score('qwenft')
json.dump(results, open(WORK / 'qwenft_results.json', 'w'), indent=2)

# ------------------------------------------------------------ relabel street crops
sdz = next((p.parent for p in IN.rglob('plate_labels*.csv')), None)
man = next(IN.rglob('full_manifest.csv'), None)
if sdz and man:
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
    log(f'{len(todo):,} street crops to read')
    with open(WORK / 'qwenft_reads.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['path', 'text'])
        for i in range(0, len(todo), 16):
            if time.time() - T0 > BUDGET_H * 3600:
                log('time budget reached')
                break
            chunk = todo[i:i + 16]
            try:
                texts = read(chunk)
            except Exception as e:
                log(f'batch {i} failed: {e}')
                continue
            for p, t in zip(chunk, texts):
                w.writerow([str(p.relative_to(sdz)), t])
            fh.flush()
            if (i // 16) % 200 == 0:
                log(f'{i + len(chunk):,}/{len(todo):,}')
log('done')
