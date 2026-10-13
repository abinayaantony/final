"""
One-time helper: convert the MiniFASNetV2 PyTorch weights from the
Silent-Face-Anti-Spoofing project into an ONNX file that
liveness_detection.py can run with onnxruntime (no PyTorch needed at runtime).

You only need PyTorch for THIS script, not for the live system.

Usage (Windows, from the folder that contains webcam_test.py):

    git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
    pip install torch onnx onnxruntime
    python convert_minifasnet_to_onnx.py --repo Silent-Face-Anti-Spoofing

Output: models/MiniFASNetV2.onnx   (input 1x3x80x80 BGR float, output 1x3 probabilities)
Class order in the output is [spoof, REAL, spoof]; index 1 = real face.
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="path to the cloned Silent-Face-Anti-Spoofing folder")
    ap.add_argument("--weights", default=None,
                    help="optional path to the .pth file (default: the V2 file inside the repo)")
    ap.add_argument("--out", default=os.path.join("models", "MiniFASNetV2.onnx"))
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    from src.model_lib.MiniFASNet import MiniFASNetV2   # from the cloned repo
    from src.utility import get_kernel

    weights = args.weights or os.path.join(
        repo, "resources", "anti_spoof_models", "2.7_80x80_MiniFASNetV2.pth")
    if not os.path.isfile(weights):
        sys.exit(f"Weights not found: {weights}")

    # The file name "2.7_80x80_MiniFASNetV2.pth" means: crop scale 2.7, 80x80 input.
    model = MiniFASNetV2(conv6_kernel=get_kernel(80, 80))
    state = torch.load(weights, map_location="cpu")
    # Weights were saved from DataParallel -> strip the "module." prefix
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    class WithSoftmax(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.softmax(self.m(x), dim=1)

    wrapped = WithSoftmax(model).eval()
    dummy = torch.rand(1, 3, 80, 80) * 255.0   # upstream feeds raw 0-255 BGR (no /255)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    kwargs = dict(input_names=["input"], output_names=["probs"],
                  opset_version=13, dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}})
    try:                       # newer PyTorch: use the classic exporter
        torch.onnx.export(wrapped, dummy, args.out, dynamo=False, **kwargs)
    except TypeError:          # older PyTorch has no 'dynamo' argument
        torch.onnx.export(wrapped, dummy, args.out, **kwargs)

    # Check that ONNX gives the same answer as PyTorch
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = wrapped(dummy).numpy()
    out = sess.run(None, {"input": dummy.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    print(f"Saved {args.out}")
    print(f"Max PyTorch-vs-ONNX difference: {diff:.2e}")
    if diff > 1e-3:
        sys.exit("Mismatch is larger than expected - do not use this file.")
    print("OK")


if __name__ == "__main__":
    main()
