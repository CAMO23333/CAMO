"""Evaluate CAMO outputs with all image- and video-quality metrics from the paper."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Mapping

import cv2
import imageio.v3 as iio
import numpy as np
import pyiqa
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from video_quality_metrics import evaluate_dover, evaluate_fastervqa


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PAPER_METRICS = ("psnr", "ssim", "lpips", "dists", "musiq", "brisque", "dover", "fastervqa")
FULL_REFERENCE_METRICS = {"psnr", "ssim", "lpips", "dists"}
Y_CHANNEL_METRICS = {"psnr", "ssim"}
VIDEO_QUALITY_METRICS = {"dover", "fastervqa"}
METRIC_ALIASES = {"fastvqa": "fastervqa"}


def _supported_entry(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS


def _sequence_map(root: Path) -> Dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Sequence directory does not exist: {root}")
    return {
        item.stem: item
        for item in sorted(root.iterdir())
        if _supported_entry(item)
    }


def _read_video(path: Path) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames could be decoded from {path}")
    return torch.stack(frames)


def _read_image_folder(path: Path) -> torch.Tensor:
    files = [item for item in sorted(path.iterdir()) if item.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise ValueError(f"No images found in {path}")
    return torch.stack([to_tensor(Image.open(item).convert("RGB")) for item in files])


def load_sequence(path: Path) -> torch.Tensor:
    if path.is_dir():
        return _read_image_folder(path)
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return _read_video(path)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)
    raise ValueError(f"Unsupported input: {path}")


def _crop_frames(frames: torch.Tensor, height: int, width: int, center: bool) -> torch.Tensor:
    _, _, frame_height, frame_width = frames.shape
    top = max((frame_height - height) // 2, 0) if center else 0
    left = max((frame_width - width) // 2, 0) if center else 0
    return frames[:, :, top : top + height, left : left + width]


def align_sequences(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    center_crop: bool,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_gt_frames = ground_truth.shape[0]
    original_pred_frames = prediction.shape[0]
    frame_count = min(original_gt_frames, original_pred_frames)
    if frame_count == 0:
        raise ValueError(f"{name}: an input sequence is empty")

    ground_truth = ground_truth[:frame_count]
    prediction = prediction[:frame_count]
    target_height = min(ground_truth.shape[-2], prediction.shape[-2])
    target_width = min(ground_truth.shape[-1], prediction.shape[-1])
    if (
        ground_truth.shape[-2:] != prediction.shape[-2:]
        or original_gt_frames != original_pred_frames
    ):
        print(
            f"[{name}] aligned to {frame_count} frames at "
            f"{target_width}x{target_height} before evaluation"
        )

    return (
        _crop_frames(ground_truth, target_height, target_width, center_crop),
        _crop_frames(prediction, target_height, target_width, center_crop),
    )


def _crop_border(frames: torch.Tensor, border: int) -> torch.Tensor:
    if border == 0:
        return frames
    if border < 0 or border * 2 >= min(frames.shape[-2:]):
        raise ValueError(
            f"Invalid crop border {border} for resolution "
            f"{frames.shape[-1]}x{frames.shape[-2]}"
        )
    return frames[:, :, border:-border, border:-border]


def _rgb_to_y(frames: torch.Tensor) -> torch.Tensor:
    red, green, blue = frames[:, 0:1], frames[:, 1:2], frames[:, 2:3]
    return 0.257 * red + 0.504 * green + 0.098 * blue + 0.0625


def _mean_metric(
    model: torch.nn.Module,
    prediction: torch.Tensor,
    ground_truth: torch.Tensor | None,
    batch_mode: bool,
) -> float:
    if batch_mode:
        values = model(prediction, ground_truth) if ground_truth is not None else model(prediction)
        return float(values.mean().item())

    values = []
    for frame_index in range(prediction.shape[0]):
        pred_frame = prediction[frame_index : frame_index + 1]
        if ground_truth is None:
            value = model(pred_frame)
        else:
            value = model(pred_frame, ground_truth[frame_index : frame_index + 1])
        values.append(float(value.mean().item()))
    return float(np.mean(values))


def evaluate_image_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor | None,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    batch_mode: bool,
    crop_border: int,
    test_y_channel: bool,
) -> Dict[str, float]:
    prediction = prediction.to(device)
    ground_truth = ground_truth.to(device) if ground_truth is not None else None
    scores: Dict[str, float] = {}

    with torch.inference_mode():
        for name, model in models.items():
            if name in FULL_REFERENCE_METRICS:
                if ground_truth is None:
                    continue
                pred_eval = _crop_border(prediction, crop_border)
                gt_eval = _crop_border(ground_truth, crop_border)
                if test_y_channel and name in Y_CHANNEL_METRICS:
                    pred_eval = _rgb_to_y(pred_eval)
                    gt_eval = _rgb_to_y(gt_eval)
                score = _mean_metric(model, pred_eval, gt_eval, batch_mode)
            else:
                score = _mean_metric(model, prediction, None, batch_mode)
            scores[name] = round(score, 4)

    return scores


def _parse_metrics(value: str) -> list[str]:
    metrics = []
    for part in value.split(","):
        name = METRIC_ALIASES.get(part.strip().lower(), part.strip().lower())
        if name and name not in metrics:
            metrics.append(name)
    if not metrics:
        raise ValueError("At least one metric must be requested")
    return metrics


def _write_metric_video(frames: torch.Tensor, output_path: Path, fps: int) -> None:
    array = (
        frames.permute(0, 2, 3, 1)
        .mul(255.0)
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    iio.imwrite(
        output_path,
        array,
        fps=fps,
        codec="libx264rgb",
        pixelformat="rgb24",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18"],
    )


def _prepare_video_inputs(
    predictions: Mapping[str, Path],
    temporary_root: Path,
    fps: int,
) -> Dict[str, Path]:
    prepared: Dict[str, Path] = {}
    for name, source in tqdm(predictions.items(), desc="Preparing video metrics"):
        destination = temporary_root / f"{name}.mp4"
        if source.is_file() and source.suffix.lower() == ".mp4":
            try:
                destination.symlink_to(source.resolve())
            except OSError:
                shutil.copy2(source, destination)
        else:
            _write_metric_video(load_sequence(source), destination, fps)
        prepared[name] = destination
    return prepared


def _merge_metric(
    per_sample: Dict[str, Dict[str, float]],
    metric: str,
    values: Mapping[str, float],
) -> None:
    missing = sorted(set(per_sample) - set(values))
    if missing:
        raise RuntimeError(f"{metric} did not return scores for: {', '.join(missing)}")
    for name in per_sample:
        per_sample[name][metric] = round(float(values[name]), 4)


def run(args: argparse.Namespace) -> Path:
    prediction_root = Path(args.pred)
    ground_truth_root = Path(args.gt) if args.gt else None
    output_root = Path(args.out) if args.out else prediction_root
    metrics = _parse_metrics(args.metrics)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    device = torch.device(args.device)

    if ground_truth_root is None and any(name in FULL_REFERENCE_METRICS for name in metrics):
        missing = sorted(name for name in metrics if name in FULL_REFERENCE_METRICS)
        raise ValueError(f"--gt is required for full-reference metrics: {', '.join(missing)}")

    image_metrics = [name for name in metrics if name not in VIDEO_QUALITY_METRICS]
    print(f"Initializing image metrics on {device}: {', '.join(image_metrics) or 'none'}")
    models = {
        name: pyiqa.create_metric(name, device=device).to(device).eval()
        for name in image_metrics
    }
    predictions = _sequence_map(prediction_root)
    ground_truths = _sequence_map(ground_truth_root) if ground_truth_root else {}
    if not predictions:
        raise ValueError(f"No supported predictions found in {prediction_root}")

    per_sample: Dict[str, Dict[str, float]] = {}
    evaluated_predictions: Dict[str, Path] = {}
    for name, prediction_path in tqdm(predictions.items(), desc="Image metrics"):
        if ground_truth_root is not None and name not in ground_truths:
            print(f"[{name}] skipped: matching ground truth was not found")
            continue

        prediction = load_sequence(prediction_path)
        ground_truth = load_sequence(ground_truths[name]) if ground_truth_root else None
        if ground_truth is not None:
            ground_truth, prediction = align_sequences(
                ground_truth,
                prediction,
                center_crop=args.center_crop,
                name=name,
            )
        per_sample[name] = evaluate_image_metrics(
            prediction,
            ground_truth,
            models,
            device,
            batch_mode=args.batch_mode,
            crop_border=args.crop_border,
            test_y_channel=args.test_y_channel,
        )
        evaluated_predictions[name] = prediction_path

    if not per_sample:
        raise RuntimeError("No matched samples were evaluated")

    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    requested_video_metrics = VIDEO_QUALITY_METRICS.intersection(metrics)
    if requested_video_metrics:
        with tempfile.TemporaryDirectory(prefix="camo-metrics-") as temporary_dir:
            video_paths = _prepare_video_inputs(
                evaluated_predictions,
                Path(temporary_dir),
                args.video_fps,
            )
            if "dover" in requested_video_metrics:
                dover_repo = Path(args.dover_repo)
                dover_weights = (
                    Path(args.dover_weights)
                    if args.dover_weights
                    else dover_repo / "pretrained_weights" / "DOVER.pth"
                )
                _merge_metric(
                    per_sample,
                    "dover",
                    evaluate_dover(
                        Path(temporary_dir),
                        dover_repo,
                        dover_weights,
                        device,
                        num_workers=args.video_metric_workers,
                    ),
                )
            if "fastervqa" in requested_video_metrics:
                fastervqa_repo = Path(args.fastervqa_repo)
                fastervqa_weights = (
                    Path(args.fastervqa_weights)
                    if args.fastervqa_weights
                    else fastervqa_repo / "pretrained_weights" / "FAST_VQA_3D_1_1.pth"
                )
                _merge_metric(
                    per_sample,
                    "fastervqa",
                    evaluate_fastervqa(
                        video_paths,
                        fastervqa_repo,
                        fastervqa_weights,
                        device,
                    ),
                )

    incomplete = {
        name: [metric for metric in metrics if metric not in scores]
        for name, scores in per_sample.items()
        if any(metric not in scores for metric in metrics)
    }
    if incomplete:
        raise RuntimeError(f"Some requested metrics were not computed: {incomplete}")

    average = {
        metric: round(float(np.mean([scores[metric] for scores in per_sample.values()])), 4)
        for metric in metrics
    }
    result = {
        "metrics": metrics,
        "per_sample": per_sample,
        "average": average,
        "count": len(per_sample),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / ("metrics_" + "_".join(metrics) + ".json")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)

    print(f"Evaluated {len(per_sample)} samples")
    for metric, value in average.items():
        print(f"{metric.upper()}: {value:.4f}")
    print(f"Results saved to {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CAMO restoration results")
    parser.add_argument("--pred", required=True, help="Prediction directory")
    parser.add_argument("--gt", default="", help="Ground-truth directory")
    parser.add_argument("--out", default="", help="Output directory (defaults to --pred)")
    parser.add_argument(
        "--metrics",
        default=",".join(PAPER_METRICS),
        help="Comma-separated metrics; defaults to every metric reported in the paper",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda, cuda:1, or cpu",
    )
    parser.add_argument("--batch_mode", action="store_true", help="Evaluate all frames as one batch")
    parser.add_argument("--crop_border", type=int, default=0, help="Border cropped for FR metrics")
    parser.add_argument("--test_y_channel", action="store_true", help="Use Y for PSNR and SSIM")
    parser.add_argument(
        "--center_crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center-crop mismatched resolutions",
    )
    parser.add_argument("--video_fps", type=int, default=25, help="FPS for image-sequence conversion")
    parser.add_argument("--video_metric_workers", type=int, default=4)
    parser.add_argument("--dover_repo", default=os.environ.get("DOVER_REPO", "metrics/DOVER"))
    parser.add_argument("--dover_weights", default=os.environ.get("DOVER_WEIGHTS", ""))
    parser.add_argument(
        "--fastervqa_repo",
        default=os.environ.get("FASTERVQA_REPO", "metrics/FAST-VQA-and-FasterVQA"),
    )
    parser.add_argument("--fastervqa_weights", default=os.environ.get("FASTERVQA_WEIGHTS", ""))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
