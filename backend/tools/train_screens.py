"""Train the screen classifier (src/stream_log/screens.npz) from hand-labelled VODs.

Labels are a JSON list, one entry per VOD:
    {"embeddings": "path/to/frames.npy",       # stream_log.games.frame_embeddings, 1 per 10 s
     "game": [[0, 28.5], ...],                  # minutes: games in progress
     "loading": [[0, 0.9], ...],                # loading screens
     "lobby": [[29, 29.9], ...]}                # post-game lobby / home screen
Everything else is "other". Frames within 30 s of a labelled boundary are left out of training.

    python tools/train_screens.py labels.json            # leave-one-VOD-out check, then train
    python tools/train_screens.py labels.json --embed P1.mp4=out1.npy ...   # make embeddings first
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stream_log import games

SCALE = 10.0
MARGIN = 0.5  # minutes


def frame_labels(entry: dict, n: int, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    labels, keep = np.full(n, games.OTHER), np.ones(n, dtype=bool)
    order = (("loading", games.LOADING), ("lobby", games.LOBBY), ("game", games.GAME))
    for i in range(n):
        minute = (i + 0.5) * games.STEP / 60
        for key, state in order:
            span = next(((lo, hi) for lo, hi in entry.get(key, []) if lo <= minute < hi), None)
            if span:
                labels[i] = state
                if margin and (minute - span[0] < margin or span[1] - minute < margin):
                    keep[i] = False
                break
    return labels, keep


def train(x: np.ndarray, y: np.ndarray, epochs: int = 400, lr: float = 0.5, l2: float = 1e-3):
    k = len(games.STATES)
    w, b = np.zeros((x.shape[1], k)), np.zeros(k)
    onehot = np.eye(k)[y]
    weight = (len(y) / (k * (onehot.sum(0) + 1)))[y]  # rare screens count more
    for _ in range(epochs):
        z = x @ w + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        g = (p - onehot) * weight[:, None] / len(y)
        w -= lr * (x.T @ g + l2 * w)
        b -= lr * g.sum(0)
    return w, b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--version", default="1")
    parser.add_argument("--out", type=Path, default=games.CLASSIFIER)
    args = parser.parse_args()
    entries = json.loads(args.labels.read_text(encoding="utf-8"))
    base = args.labels.parent
    data = []
    for entry in entries:
        emb = np.load(base / entry["embeddings"])
        data.append((entry, emb, *frame_labels(entry, len(emb), MARGIN)))

    for held, (entry, emb, _, _) in enumerate(data):
        rest = [d for i, d in enumerate(data) if i != held]
        w, b = train(np.concatenate([d[1][d[3]] for d in rest]) * SCALE,
                     np.concatenate([d[2][d[3]] for d in rest]))
        clf = {"w": w, "b": b, "scale": SCALE, "version": "check"}
        path = games.viterbi(games.screen_probs(emb, clf))
        truth, _ = frame_labels(entry, len(emb))
        found = [(round(g["start"] / 60, 1), round(g["end"] / 60, 1))
                 for g in games.games_from_path(path)]
        print(f"held out {entry['embeddings']}: frames {np.mean(path == truth):.3f}")
        print(f"  found  {found}")
        print(f"  truth  {[tuple(s) for s in entry['game']]}")

    w, b = train(np.concatenate([d[1][d[3]] for d in data]) * SCALE,
                 np.concatenate([d[2][d[3]] for d in data]))
    np.savez(args.out, w=w.astype(np.float32), b=b.astype(np.float32), scale=SCALE,
             version=args.version, states=np.array(games.STATES))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
