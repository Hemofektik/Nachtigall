"""Export a trained checkpoint to ONNX and verify parity with PyTorch.

Usage:
    python -m n8igall_mvp.export_onnx --checkpoint runs/v1/best.pt --out runs/v1/model.onnx

The exported graph takes ``float32 [1, C_in, 900, 900]`` (log1p radar history
+ static channels) and returns ``float32 [1, 1, 900, 900]`` (log1p rate at
+lead minutes). Run inference in C++ via ONNX Runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import UNet


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    leads = ckpt.get("leads")
    out_channels = len(leads) if leads else 1
    model = UNet(ckpt["in_channels"], base=ckpt["base"],
                 out_channels=out_channels)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, ckpt["in_channels"], args.height, args.width)
    torch.onnx.export(
        model, dummy, str(args.out),
        input_names=["input"], output_names=["output"],
        opset_version=args.opset, dynamo=False,
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
    )

    # Parity check against onnxruntime
    import onnxruntime as ort
    with torch.no_grad():
        ref = model(dummy).numpy()
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})[0]
    max_err = float(np.abs(ref - out).max())
    print(f"exported {args.out} | max |pytorch - onnx| = {max_err:.2e}")
    if max_err > 1e-4:
        raise SystemExit("parity check failed")
    print("parity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())