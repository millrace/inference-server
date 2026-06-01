"""Generate safetensors-loader fixtures from the real Qwen2.5-0.5B checkpoint.

Writes tests/fixtures/loader/ (GITIGNORED — references a machine-specific path,
and the checkpoint itself is ~1 GB and lives in the HF cache):

  meta.txt      line 1: absolute path to model.safetensors
                line 2: tensor count N
                next N: "<name> <ndim> <dim0> <dim1> ..."  (expected shape)
  expected.bin  N * 8 float32: the first 8 flat elements of each tensor, in the
                same order, decoded bf16 -> f32 by torch (the ground truth the
                Mojo loader must reproduce).

Run via `pixi run loader-capture`.
"""

import glob
import os

import numpy as np
from safetensors import safe_open

NAMES = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.bias",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.23.mlp.down_proj.weight",
]
FIX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "loader"))


def main():
    snaps = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*"
        )
    )
    if not snaps:
        raise SystemExit("Qwen2.5-0.5B-Instruct not found in HF cache — pull it first")
    path = os.path.realpath(os.path.join(snaps[0], "model.safetensors"))

    os.makedirs(FIX, exist_ok=True)
    firsts = []
    lines = [path, str(len(NAMES))]
    with safe_open(path, framework="pt") as f:
        for name in NAMES:
            t = f.get_tensor(name)
            shape = list(t.shape)
            flat = t.float().numpy().reshape(-1)[:8].astype(np.float32)
            firsts.append(flat)
            lines.append(name + " " + str(len(shape)) + " " + " ".join(str(d) for d in shape))
            print(f"{name}: shape={shape} dtype={t.dtype} first0={float(flat[0]):.6g}")

    with open(os.path.join(FIX, "meta.txt"), "w") as fh:
        fh.write("\n".join(lines))
    np.concatenate(firsts).astype(np.float32).tofile(os.path.join(FIX, "expected.bin"))
    print(f"OK: wrote {len(NAMES)} tensor specs + expected.bin to {FIX}")
    print(f"checkpoint: {path}")


if __name__ == "__main__":
    main()
