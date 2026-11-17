"""Small ONNX models shared by several stages: text embeddings, text emotions, SigLIP.

All run on ONNX Runtime's CPU provider (signed by Microsoft, so Smart App Control allows it).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2) - 2)
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _tokenizer(path: Path, max_length: int, pad_to: int | None = None):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.enable_truncation(max_length)
    if pad_to:
        tokenizer.enable_padding(length=pad_to, pad_id=tokenizer.token_to_id("</s>") or 1)
    else:
        tokenizer.enable_padding()
    return tokenizer


def _feeds(session_, encodings) -> dict[str, np.ndarray]:
    ids = np.array([e.ids for e in encodings], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    available = {i.name for i in session_.get_inputs()}
    feeds = {"input_ids": ids}
    if "attention_mask" in available:
        feeds["attention_mask"] = mask
    if "token_type_ids" in available:
        feeds["token_type_ids"] = np.zeros_like(ids)
    return feeds


class TextEmbedder:
    """bge-small-en-v1.5: CLS pooling, unit vectors (384-d)."""

    def __init__(self, model_dir: Path, batch_size: int = 64) -> None:
        self.session = session(model_dir / "onnx" / "model.onnx")
        self.tokenizer = _tokenizer(model_dir / "tokenizer.json", 128)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer.encode_batch(texts[start : start + self.batch_size])
            hidden = self.session.run(None, _feeds(self.session, batch))[0]
            cls = hidden[:, 0]
            out.append(cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-9))
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


class TextEmotions:
    """roberta-base go_emotions: 28 multi-label emotions (sigmoid)."""

    def __init__(self, model_dir: Path, batch_size: int = 64) -> None:
        onnx_dir = model_dir / "onnx"
        self.session = session(onnx_dir / "model_quantized.onnx")
        self.tokenizer = _tokenizer(onnx_dir / "tokenizer.json", 64)
        config = json.loads((onnx_dir / "config.json").read_text(encoding="utf-8"))
        self.labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
        self.batch_size = batch_size

    def scores(self, texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer.encode_batch(texts[start : start + self.batch_size])
            logits = self.session.run(None, _feeds(self.session, batch))[0]
            out.append(1 / (1 + np.exp(-logits)))
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), dtype=np.float32)


class Siglip:
    """SigLIP base 224: zero-shot image/text matching for stream modes and visual events."""

    SIZE = 224

    def __init__(self, model_dir: Path) -> None:
        onnx_dir = model_dir / "onnx"
        self.vision = session(onnx_dir / "vision_model.onnx")
        self.text = session(onnx_dir / "text_model.onnx")
        # SigLIP was trained with text padded to 64 tokens; shorter padding changes the scores.
        self.tokenizer = _tokenizer(model_dir / "tokenizer.json", 64, pad_to=64)
        # The exported towers don't carry the learned temperature/bias, so scores are only used
        # through a softmax over competing prompts, where the bias cancels.
        self.temperature = 100.0

    def _output(self, sess, feeds: dict) -> np.ndarray:
        names = [o.name for o in sess.get_outputs()]
        name = "pooler_output" if "pooler_output" in names else names[-1]
        (out,) = sess.run([name], feeds)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch([t.lower() for t in texts])
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        return self._output(self.text, {"input_ids": ids})

    def embed_images(self, frames: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """`frames`: uint8 (N, 224, 224, 3) RGB."""
        out = []
        for start in range(0, len(frames), batch_size):
            pixels = frames[start : start + batch_size].astype(np.float32) / 255.0
            pixels = ((pixels - 0.5) / 0.5).transpose(0, 3, 1, 2)
            out.append(self._output(self.vision, {"pixel_values": pixels}))
        return np.concatenate(out) if out else np.zeros((0, 768), dtype=np.float32)

    def prompt_probs(self, image_embs: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
        """Softmax over prompts for each image: (N images, M prompts)."""
        logits = image_embs @ text_embs.T * self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)
