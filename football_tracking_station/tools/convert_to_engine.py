"""
tools/convert_to_engine.py

Convert YOLO and/or DeepBall .pt checkpoints to TensorRT .engine.

Recommended usage:
    python tools/convert_to_engine.py --target yolo --yolo weights/yolo_best.pt
    python tools/convert_to_engine.py --target deepball --deepball weights/deepball_best.pt
    python tools/convert_to_engine.py --target all --yolo weights/yolo_best.pt --deepball weights/deepball_best.pt

Notes:
- Run this on a machine with NVIDIA GPU, CUDA, TensorRT, and trtexec available.
- YOLO conversion uses Ultralytics export(format="engine").
- DeepBall conversion exports ONNX first, then builds TensorRT engine with trtexec.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ConvertResult:
    model_type: str
    input_path: Path
    engine_path: Path


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT engine conversion should be run on an NVIDIA CUDA GPU machine. "
            "torch.cuda.is_available() returned False."
        )


def _require_file(path: str | Path, suffix: str = ".pt") -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    if suffix and p.suffix.lower() != suffix:
        raise ValueError(f"Expected a {suffix} file, got: {p}")
    return p


def _require_trtexec() -> str:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError(
            "Cannot find trtexec in PATH. Install TensorRT and make sure trtexec is available."
        )
    return trtexec


class YoloEngineConverter:
    """Convert Ultralytics YOLO .pt to TensorRT .engine."""

    def __init__(
        self,
        pt_path: str | Path,
        imgsz: int = 640,
        batch: int = 1,
        fp16: bool = True,
        dynamic: bool = False,
        device: int = 0,
    ):
        self.pt_path = _require_file(pt_path, ".pt")
        self.imgsz = imgsz
        self.batch = batch
        self.fp16 = fp16
        self.dynamic = dynamic
        self.device = device

    def convert(self) -> ConvertResult:
        _require_cuda()

        from ultralytics import YOLO

        print(f"[YOLO] Loading: {self.pt_path}")
        model = YOLO(str(self.pt_path))

        print(
            "[YOLO] Exporting TensorRT engine "
            f"(imgsz={self.imgsz}, batch={self.batch}, fp16={self.fp16}, dynamic={self.dynamic})"
        )

        exported = model.export(
            format="engine",
            imgsz=self.imgsz,
            batch=self.batch,
            half=self.fp16,
            dynamic=self.dynamic,
            device=self.device,
        )

        engine_path = Path(exported)
        if not engine_path.is_file():
            candidate = self.pt_path.with_suffix(".engine")
            if candidate.is_file():
                engine_path = candidate
            else:
                raise RuntimeError(f"YOLO export finished but engine file was not found: {exported}")

        print(f"[YOLO] Done: {engine_path}")
        return ConvertResult("yolo", self.pt_path, engine_path)


class DeepBallEngineConverter:
    """Convert DeepBall .pt to ONNX, then TensorRT .engine."""

    def __init__(
        self,
        pt_path: str | Path,
        fp16: bool = True,
        workspace_mb: int = 1024,
        opset: int = 11,
        keep_onnx: bool = False,
    ):
        self.pt_path = _require_file(pt_path, ".pt")
        self.fp16 = fp16
        self.workspace_mb = workspace_mb
        self.opset = opset
        self.keep_onnx = keep_onnx

    def convert(self) -> ConvertResult:
        _require_cuda()
        trtexec = _require_trtexec()

        from football_tracking_station.core.deepball_architecture import DeepBall

        onnx_path = self.pt_path.with_suffix(".onnx")
        engine_path = self.pt_path.with_suffix(".engine")

        device = torch.device("cuda:0")

        print(f"[DeepBall] Loading: {self.pt_path}")
        model = DeepBall().to(device)

        ckpt = torch.load(str(self.pt_path), map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()

        print(f"[DeepBall] Exporting ONNX: {onnx_path}")
        dummy = torch.randn(1, 9, 256, 256, device=device)

        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=self.opset,
            do_constant_folding=True,
        )

        print(f"[DeepBall] Building TensorRT engine: {engine_path}")
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--workspace={self.workspace_mb}",
        ]

        if self.fp16:
            cmd.append("--fp16")

        print("[DeepBall] Running:", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,
        )

        if result.returncode != 0:
            print("[DeepBall] trtexec stdout tail:")
            print(result.stdout[-3000:])
            print("[DeepBall] trtexec stderr tail:")
            print(result.stderr[-3000:])
            raise RuntimeError("DeepBall TensorRT conversion failed.")

        if not engine_path.is_file():
            raise RuntimeError(f"trtexec finished but engine file was not found: {engine_path}")

        if not self.keep_onnx:
            onnx_path.unlink(missing_ok=True)

        print(f"[DeepBall] Done: {engine_path}")
        return ConvertResult("deepball", self.pt_path, engine_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO and/or DeepBall .pt models to TensorRT .engine.")

    parser.add_argument(
        "--target",
        choices=["yolo", "deepball", "all"],
        required=True,
        help="Which model to convert.",
    )

    parser.add_argument("--yolo", type=str, default=None, help="Path to YOLO .pt checkpoint.")
    parser.add_argument("--deepball", type=str, default=None, help="Path to DeepBall .pt checkpoint.")

    parser.add_argument("--imgsz", type=int, default=640, help="YOLO export image size.")
    parser.add_argument("--batch", type=int, default=1, help="YOLO export batch size.")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic shape for YOLO export.")

    parser.add_argument("--workspace", type=int, default=1024, help="TensorRT workspace size in MB for DeepBall.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset for DeepBall export.")
    parser.add_argument("--keep-onnx", action="store_true", help="Keep intermediate DeepBall ONNX file.")

    parser.add_argument("--fp32", action="store_true", help="Disable FP16 and export/build FP32 engine.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index for YOLO export.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fp16 = not args.fp32

    results: list[ConvertResult] = []

    if args.target in ("yolo", "all"):
        if not args.yolo:
            raise ValueError("--yolo is required when --target is yolo or all")

        results.append(
            YoloEngineConverter(
                pt_path=args.yolo,
                imgsz=args.imgsz,
                batch=args.batch,
                fp16=fp16,
                dynamic=args.dynamic,
                device=args.device,
            ).convert()
        )

    if args.target in ("deepball", "all"):
        if not args.deepball:
            raise ValueError("--deepball is required when --target is deepball or all")

        results.append(
            DeepBallEngineConverter(
                pt_path=args.deepball,
                fp16=fp16,
                workspace_mb=args.workspace,
                opset=args.opset,
                keep_onnx=args.keep_onnx,
            ).convert()
        )

    print("\n=== Conversion summary ===")
    for r in results:
        print(f"{r.model_type}: {r.input_path} -> {r.engine_path}")


if __name__ == "__main__":
    main()
