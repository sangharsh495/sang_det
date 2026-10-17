"""Fix or drop training labels that are not valid Indian registrations.

Input: audit.csv (split,path,label,raw,fmt,conf) from running the current reader
over every crop with plate_format.decode. Output: cleaned CSVs in ds/clean/.
"""
import csv, itertools, re, sys, collections
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from plate_format import is_valid as valid

CONF = {'O': '0D', '0': 'OD', 'I': '1', '1': 'I', 'H': 'MW', 'X': 'K', 'N': 'MH', 'Z': '2', '2': 'Z',
        'S': '5', '5': 'S', 'B': '8', '8': 'B', 'G': '6C', '6': 'G', 'D': '0', 'Q': '0', 'M': 'H',
        'K': 'H', 'T': '1', 'Y': 'V', 'C': 'G', 'R': 'P', 'U': 'P'}


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def variants(lab):
    """All valid strings reachable by swapping up to two confusable characters."""
    out = set()
    idx = [i for i, c in enumerate(lab) if c in CONF]
    for k in (1, 2):
        for pos in itertools.combinations(idx, k):
            for subs in itertools.product(*[CONF[lab[i]] for i in pos]):
                s = list(lab)
                for i, c in zip(pos, subs):
                    s[i] = c
                s = ''.join(s)
                if valid(s):
                    out.add(s)
    return out


stats = collections.Counter()
rows = {'train_real': [], 'val_real': []}
for r in csv.DictReader(open('audit.csv')):
    lab = re.sub(r'[^A-Z0-9]', '', r['label'].upper())
    pred, conf = r['fmt'], float(r['conf'])
    if valid(lab):
        new, why = lab, 'ok'
    else:
        vs = variants(lab)
        if pred in vs:
            new, why = pred, 'swap_agrees'
        elif len(vs) == 1 and lev(lab, pred) <= 2:
            new, why = vs.pop(), 'swap_unique'
        elif lev(lab, pred) <= 1 and conf > 0.3:
            new, why = pred, 'model_close'
        else:
            new, why = None, 'drop'
    stats[(r['split'], why)] += 1
    if new:
        rows[r['split']].append((r['path'], new))

for split, rs in rows.items():
    with open(f'ds/clean/{split}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['image_path', 'plate_text']); w.writerows(rs)
for k, v in sorted(stats.items()):
    print(k, v)
