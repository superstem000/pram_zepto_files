#!/usr/bin/env python3
"""ONNX export of a trained LOCA-PRAM checkpoint (input for the Hailo-8 DFC).

Exports only (p, mu_xy). The other three heads (mu_a, sigma, bg) are
training-only (GMMLoss inputs). tanh_scale is applied host-side and stored
in postprocess.json, not in the graph.

Usage:
    python export_onnx.py --checkpoint ../training/runs/<run>/LOCA_PRAM_BEST.pth

Outputs:
    <onnx>              the exported graph, static [1, 1, H, W]
    <postproc.json>     host-side scalars (tanh_scale, window_size)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from model import GaussianMixtureModel, WINDOW_SIZE

try:
    import onnx
    import onnxruntime as ort
except ImportError as e:
    sys.exit(f"error: install onnx + onnxruntime first ({e})")


class _ExportWrapper(nn.Module):
    """torch.onnx.export traces model.forward; route through forward_export()
    so only (p, mu_xy) are exported."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_export(x)


def main():
    ap = argparse.ArgumentParser(
        description="Export a LOCA-PRAM checkpoint to ONNX for the Hailo-8 DFC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the trained .pth checkpoint "
                         "(e.g. ../training/runs/<run>/LOCA_PRAM_BEST.pth).")
    ap.add_argument("--onnx", default="loca_pram_export.onnx",
                    help="Output ONNX path.")
    ap.add_argument("--postproc", default="postprocess.json",
                    help="Sidecar JSON with host-side scalars (tanh_scale, "
                         "window_size).")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--allow-reshape-transpose", action="store_true",
                    help="Silence the Reshape/Transpose presence warning.")
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"error: checkpoint not found: {args.checkpoint}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"loading checkpoint: {args.checkpoint}")
    model = GaussianMixtureModel(num_channels=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    print(f"tanh_scale = {model.tanh_scale} (host-side scalar)")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    _H, _W = WINDOW_SIZE
    dummy = torch.zeros(1, 1, _H, _W)
    print(f"input shape: (1, 1, {_H}, {_W})   opset={args.opset}")

    exporter = _ExportWrapper(model).eval()
    torch.onnx.export(
        exporter,
        (dummy,),
        args.onnx,
        export_params=True,
        training=torch.onnx.TrainingMode.PRESERVE,
        do_constant_folding=False,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["p", "mu_xy"],
    )
    print(f"wrote {args.onnx}")

    # ------------------------------------------------------------------
    # Sidecar postprocess.json (host-side scalars only)
    # ------------------------------------------------------------------
    with open(args.postproc, "w") as f:
        json.dump({
            "tanh_scale":  model.tanh_scale,
            "window_size": [int(_H), int(_W)],
        }, f, indent=2)
    print(f"wrote {args.postproc}")

    # ------------------------------------------------------------------
    # Op-type audit
    # ------------------------------------------------------------------
    graph = onnx.load(args.onnx)
    op_types = sorted({node.op_type for node in graph.graph.node})
    print(f"\nUnique ONNX ops ({len(op_types)}):")
    for t in op_types:
        print(f"  {t}")

    # Ops not supported by the Hailo DFC.
    HARD_FAIL = {"ScatterND", "GroupNormalization", "InstanceNormalization"}
    found_bad = HARD_FAIL & set(op_types)
    if found_bad:
        sys.exit(f"HARD FAIL: unsupported op(s) present in graph: {found_bad}")

    reshape_transpose = {"Reshape", "Transpose"} & set(op_types)
    if reshape_transpose and not args.allow_reshape_transpose:
        print(f"\n  WARN: {reshape_transpose} present. Where from:")
        for node in graph.graph.node:
            if node.op_type in reshape_transpose:
                print(f"    op={node.op_type}  name={node.name}  "
                      f"inputs={list(node.input)}")

    # ------------------------------------------------------------------
    # I/O + node histogram
    # ------------------------------------------------------------------
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    print("\nGraph I/O:")
    for io in sess.get_inputs():
        print(f"  input   name={io.name}  shape={io.shape}  dtype={io.type}")
    for io in sess.get_outputs():
        print(f"  output  name={io.name}  shape={io.shape}  dtype={io.type}")

    print("\nNode-type histogram:")
    for op, c in sorted(Counter(n.op_type for n in graph.graph.node).items(),
                        key=lambda kv: -kv[1]):
        print(f"  {c:4d}  {op}")

    # ------------------------------------------------------------------
    # Parity check: PyTorch vs onnxruntime on random input
    # ------------------------------------------------------------------
    np.random.seed(0)
    x_np = np.random.randn(1, 1, _H, _W).astype(np.float32)

    with torch.no_grad():
        p_pt, mu_pt = exporter(torch.from_numpy(x_np))
    p_pt  = p_pt.cpu().numpy()
    mu_pt = mu_pt.cpu().numpy()

    p_ort, mu_ort = sess.run(["p", "mu_xy"], {"input": x_np})

    p_diff  = float(np.max(np.abs(p_pt  - p_ort)))
    mu_diff = float(np.max(np.abs(mu_pt - mu_ort)))
    print("\nParity check (PyTorch vs onnxruntime, max abs diff):")
    print(f"  p     : {p_diff:.3e}")
    print(f"  mu_xy : {mu_diff:.3e}")
    if max(p_diff, mu_diff) > 1e-4:
        print("  !! parity drift > 1e-4 — inspect before feeding into DFC.")
    else:
        print("  ok.")


if __name__ == "__main__":
    main()
