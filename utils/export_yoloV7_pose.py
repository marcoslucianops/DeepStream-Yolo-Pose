import os
import onnx
import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x[0]
        boxes = x[:, :, :4]
        objectness = x[:, :, 4:5]
        scores = x[:, :, 5:6]
        scores *= objectness
        kpts = x[:, :, 6:]
        return torch.cat([boxes, scores, kpts], dim=-1)


def yolov7_pose_export(weights, device):
    model = attempt_load(weights, map_location=device)
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()
        if isinstance(m, models.common.Conv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model.model[-1].export = False
    model.model[-1].concat = True
    model.eval()
    return model


def suppress_warnings():
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)


def main(args):
    suppress_warnings()

    print(f"\nStarting: {args.weights}")

    print("Opening YOLOv7-Pose model")

    device = torch.device("cpu")
    model = yolov7_pose_export(args.weights, device)

    if hasattr(model, "names") and len(model.names) > 0:
        print("Creating labels.txt file")
        with open("labels.txt", "w", encoding="utf-8") as f:
            for name in model.names:
                f.write(f"{name}\n")

    model = nn.Sequential(model, DeepStreamOutput())

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    if img_size == [640, 640] and args.p6:
        img_size = [1280] * 2

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = args.weights.rsplit(".", 1)[0] + ".onnx"

    dynamic_axes = {
        "input": {
            0: "batch"
        },
        "output": {
            0: "batch"
        }
    }

    print("Exporting the model to ONNX")
    torch.onnx.export(
        model, onnx_input_im, onnx_output_file, verbose=False, opset_version=args.opset, do_constant_folding=True,
        input_names=["input"], output_names=["output"], dynamic_axes=dynamic_axes if args.dynamic else None
    )

    if args.simplify:
        print("Simplifying the ONNX model")
        import onnxslim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f"Done: {onnx_output_file}\n")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="DeepStream YOLOv7-Pose conversion")
    parser.add_argument("-w", "--weights", required=True, type=str, help="Input weights (.pt) file path (required)")
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[640], help="Inference size [H,W] (default [640])")
    parser.add_argument("--p6", action="store_true", help="P6 model")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify model")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch-size")
    parser.add_argument("--batch", type=int, default=1, help="Static batch-size")
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit("Invalid weights file")
    if args.dynamic and args.batch > 1:
        raise SystemExit("Cannot set dynamic batch-size and static batch-size at same time")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
