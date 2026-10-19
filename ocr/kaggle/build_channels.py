"""Build (and optionally push) one round of channel-harvest CPU kernels.

    python3 ocr/kaggle/build_channels.py ROUND [--push] [--state STATE.json]

Each of the 5 channels in ocr/kaggle/channels/chN.txt gets a kernel
`plate-ch{N}-r{ROUND}` holding the next PER_ROUND not-yet-done video IDs. A
kernel stops at its time budget, so after a round finishes run
`--mark-done KERNEL_LOG` to record which videos were actually processed;
the rest go into the next round. Every round is a separate kernel, so no
round's crops are overwritten.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PER_ROUND = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('round', type=int, nargs='?')
    ap.add_argument('--push', action='store_true')
    ap.add_argument('--state', required=True)
    ap.add_argument('--mark-done', nargs='*', default=[], help='kernel log files of a finished round')
    a = ap.parse_args()
    st = Path(a.state)
    state = json.loads(st.read_text()) if st.exists() else {'done': []}
    done = set(state['done'])
    for log in a.mark_done:
        text = Path(log).read_text()
        got = set(re.findall(r'\] ([A-Za-z0-9_-]{11}): (?:\d+ crops|download failed)', text))
        done |= got
        print(f'{log}: {len(got)} videos done')
    state['done'] = sorted(done)
    st.write_text(json.dumps(state))
    if a.round is None:
        return
    base = (HERE / 'train_ytharvest.py').read_text()
    fmt = (HERE.parent / 'plate_format.py').read_text().replace('from __future__ import annotations', '')
    for ch in sorted(HERE.glob('channels/ch*.txt')):
        name = ch.stem
        ids = [v for v in ch.read_text().split() if v not in done][:PER_ROUND]
        if not ids:
            print(f'{name}: all videos done')
            continue
        slug = f'plate-{name}-r{a.round}'
        out = HERE / 'build' / slug
        out.mkdir(parents=True, exist_ok=True)
        code = base.replace("SHARD = 'day1'\n", f"SHARD = '{name}'\n", 1)
        code = code.replace('VIDEO_IDS = []', f'VIDEO_IDS = {ids!r}', 1)
        (out / 'train_ytharvest.py').write_text(fmt + '\n' + code)
        subprocess.run(['python3', '-m', 'py_compile', str(out / 'train_ytharvest.py')], check=True)
        (out / 'kernel-metadata.json').write_text(json.dumps({
            'id': f'sangharshgedekar/{slug}', 'title': slug, 'code_file': 'train_ytharvest.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': False, 'enable_tpu': False, 'enable_internet': True,
            'dataset_sources': [], 'kernel_sources': [], 'competition_sources': [],
            'model_sources': []}, indent=2))
        print(f'{slug}: {len(ids)} videos')
        if a.push:
            subprocess.run(['kaggle', 'kernels', 'push', '-p', str(out)], check=False)


if __name__ == '__main__':
    main()
