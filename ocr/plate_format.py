"""Indian plate format decoding and multi-frame voting for fast-plate-ocr models.

The OCR model outputs a (slots, 37) probability grid: one row per character
slot, one column per symbol (0-9, A-Z, '_' pad). Taking the argmax of each row
can produce strings no Indian plate can carry ("TR99AZ795", "KAO3...").
`decode()` instead returns the most probable string that IS a valid
registration:

    SS NN L{0,3} N{1,4}     state code, district, series, number  (MH12AB1234)
    YY BH NNNN L{1,2}       Bharat series                           (22BH1234AB)

`PlateVoter` combines several frames of the same vehicle. A 24/7 camera sees
each vehicle in many frames, and summing log-probabilities across frames
before decoding beats reading the single best frame.

Only numpy is required, so this runs unchanged on a Raspberry Pi.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
STATES = tuple(
    'AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN '
    'MP MZ NL OD OR PB PY RJ SK TN TR TS UK UA UP WB'.split())
PLATE_RE = re.compile(r'^([A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}|\d{2}BH\d{4}[A-Z]{1,2})$')

_DIG = np.array([ALPHABET.index(c) for c in '0123456789'])
_LET = np.array([ALPHABET.index(c) for c in ALPHABET[10:36]])
_PAD = ALPHABET.index('_')
_STATE_IDX = np.array([[ALPHABET.index(s[0]), ALPHABET.index(s[1])] for s in STATES])


def is_valid(text: str) -> bool:
    """True if `text` is a well-formed Indian registration."""
    return bool(PLATE_RE.match(text)) and (text[:2] in STATES or text[2:4] == 'BH')


def _templates(slots: int):
    for d in (1, 2):
        for letters in range(4):
            for n in range(1, 5):
                t = 'SS' + 'D' * d + 'L' * letters + 'D' * n
                if len(t) <= slots:
                    yield t
    for letters in (1, 2):
        t = 'DDBH' + 'DDDD' + 'L' * letters
        if len(t) <= slots:
            yield t


_TEMPLATE_CACHE: dict[int, list[str]] = {}


def decode(prob: np.ndarray) -> tuple[str, float]:
    """Best valid plate for one (slots, 37) probability or log-prob grid.

    Returns (text, confidence) where confidence is the joint probability of
    the chosen string under the model, in [0, 1].
    """
    lp = prob if prob.min() < 0 else np.log(np.clip(prob, 1e-9, 1.0))
    slots = lp.shape[0]
    templates = _TEMPLATE_CACHE.setdefault(slots, list(_templates(slots)))
    pad_tail = np.concatenate([np.cumsum(lp[::-1, _PAD])[::-1], [0.0]])

    state_scores = lp[0, _STATE_IDX[:, 0]] + lp[1, _STATE_IDX[:, 1]]
    best_state = int(state_scores.argmax())
    dig_best = lp[:, _DIG].argmax(1)
    let_best = lp[:, _LET].argmax(1)

    best_text, best_score = '', -np.inf
    for t in templates:
        score, chars, i = 0.0, [], 0
        while i < len(t):
            c = t[i]
            if c == 'S':
                score += state_scores[best_state]
                chars.append(STATES[best_state])
                i += 2
                continue
            if c == 'D':
                k = _DIG[dig_best[i]]
            elif c == 'L':
                k = _LET[let_best[i]]
            else:
                k = ALPHABET.index(c)
            score += lp[i, k]
            chars.append(ALPHABET[k])
            i += 1
        score += pad_tail[len(t)]
        if score > best_score:
            best_score, best_text = score, ''.join(chars)
    return best_text, float(np.exp(best_score))


def argmax_text(prob: np.ndarray) -> str:
    """Plain per-slot argmax, for comparison."""
    return ''.join(ALPHABET[i] for i in prob.argmax(1)).replace('_', '')


@dataclass
class PlateVoter:
    """Accumulate frames of one tracked vehicle, then read the plate once.

    Each frame adds its log-probabilities, weighted by an optional quality
    score (e.g. crop sharpness or detector confidence). Frames that are
    nearly unreadable therefore add little, and the plate is decoded from
    the evidence of all frames together.
    """
    min_frames: int = 3
    _sum: np.ndarray | None = field(default=None, repr=False)
    _weight: float = 0.0
    frames: int = 0

    def add(self, prob: np.ndarray, quality: float = 1.0) -> None:
        lp = np.log(np.clip(prob, 1e-9, 1.0)) * quality
        self._sum = lp if self._sum is None else self._sum + lp
        self._weight += quality
        self.frames += 1

    def ready(self) -> bool:
        return self.frames >= self.min_frames

    def result(self) -> tuple[str, float]:
        if self._sum is None:
            return '', 0.0
        return decode(self._sum / max(self._weight, 1e-6))
