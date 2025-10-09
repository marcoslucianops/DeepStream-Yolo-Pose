import os
import onnx
import torch
import torch.nn as nn

from super_gradients.training import models


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x[0]
        scores = x[1]
        b, c = boxes.shape[:2]
        kpts = torch.cat([x[2], x[3].unsqueeze(-1)], dim=-1).view(b, c, -1)
        return torch.cat([boxes, scores, kpts], dim=-1)


def yolonas_pose_export(model_name, weights, size):
    img_size = size * 2 if len(size) == 1 else size
    model = models.get(model_name, num_classes=17, checkpoint_path=weights)
    model.eval()
    model.prep_model_for_conversion(input_size=[1, 3, *img_size])
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

    print("Opening YOLO-NAS-Pose model")

    device = torch.device("cpu")
    model = yolonas_pose_export(args.model, args.weights, args.size)

    model = nn.Sequential(model, DeepStreamOutput())

    img_size = args.size * 2 if len(args.size) == 1 else args.size

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
    parser = argparse.ArgumentParser(description="DeepStream YOLO-NAS-Pose conversion")
    parser.add_argument("-m", "--model", required=True, type=str, help="Model name (required)")
    parser.add_argument("-w", "--weights", required=True, type=str, help="Input weights (.pth) file path (required)")
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[640], help="Inference size [H,W] (default [640])")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify model")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch-size")
    parser.add_argument("--batch", type=int, default=1, help="Static batch-size")
    args = parser.parse_args()
    if args.model == "":
        raise SystemExit("Invalid model name")
    if not os.path.isfile(args.weights):
        raise SystemExit("Invalid weights file")
    if args.dynamic and args.batch > 1:
        raise SystemExit("Cannot set dynamic batch-size and static batch-size at same time")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
