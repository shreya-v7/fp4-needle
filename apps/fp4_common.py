"""Shared helpers for the experiment runners: named parameters, checkpoints, safetensors."""
import json
import os
import struct

import numpy as np

import needle as ndl
import needle.nn as nn


def named_parameters(module, prefix=""):
    """(name, Parameter) pairs in a stable order, named by attribute path."""
    out = []

    def walk(value, name):
        if isinstance(value, nn.Parameter):
            out.append((name, value))
        elif isinstance(value, nn.Module):
            for k, v in value.__dict__.items():
                walk(v, f"{name}.{k}" if name else k)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(v, f"{name}.{i}")
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{name}.{k}")

    walk(module, prefix)
    return out


def save_checkpoint(model, path, meta):
    arrays = {name: p.numpy() for name, p in named_parameters(model)}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, __meta__=np.array(json.dumps(meta)), **arrays)


def load_checkpoint(path):
    data = np.load(path)
    meta = json.loads(str(data["__meta__"]))
    return meta, {k: data[k] for k in data.files if k != "__meta__"}


def assign_weights(model, arrays, device):
    for name, p in named_parameters(model):
        p.data = ndl.Tensor(arrays[name].astype(np.float32), device=device)


def is_matmul_weight(name, array):
    """2-D weights used as the right operand of x @ W (reduction axis 0); not embeddings."""
    return array.ndim == 2 and "embedding" not in name and min(array.shape) >= 2


_ST_DTYPES = {"F32": np.float32, "F16": np.float16}


def read_safetensors(path, keep=lambda name, shape: True):
    """Minimal safetensors reader (no PyTorch): 8-byte header length, JSON header, raw data."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in header.items():
            if name == "__metadata__" or not keep(name, info["shape"]):
                continue
            start, end = info["data_offsets"]
            f.seek(base + start)
            dtype = _ST_DTYPES[info["dtype"]]
            out[name] = np.frombuffer(f.read(end - start), dtype=dtype).reshape(info["shape"]).astype(np.float32)
    return out
