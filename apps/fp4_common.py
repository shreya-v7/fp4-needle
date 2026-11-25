# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Checkpoint helpers for the experiment runners (save, load, and load back into a model)."""
import json
import os

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
