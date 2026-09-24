"""Adapters for the DOVER and FasterVQA paper metrics."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import yaml
from tqdm import tqdm


def _require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Run `bash setup_metrics.sh` or pass the corresponding repository/weight option."
        )
    return path


def _require_repo(path: Path, package_dir: str, description: str) -> Path:
    path = path.expanduser().resolve()
    if not (path / package_dir).is_dir():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Run `bash setup_metrics.sh` or pass the corresponding repository option."
        )
    return path


def _load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _prepend_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def evaluate_dover(
    video_dir: Path,
    repo: Path,
    weights: Path,
    device: torch.device,
    num_workers: int = 4,
) -> Dict[str, float]:
    """Evaluate every MP4 in ``video_dir`` with the official DOVER model."""

    repo = _require_repo(repo, "dover", "DOVER repository")
    weights = _require_file(weights, "DOVER checkpoint")
    config_path = _require_file(repo / "dover.yml", "DOVER configuration")
    _prepend_import_path(repo)

    try:
        from dover.datasets import ViewDecompositionDataset
        from dover.models import DOVER
    except ImportError as error:
        raise RuntimeError(
            "DOVER dependencies could not be imported. Install requirements.txt and rerun "
            "setup_metrics.sh."
        ) from error

    with config_path.open("r", encoding="utf-8") as handle:
        options = yaml.safe_load(handle)

    model = DOVER(**options["model"]["args"]).to(device)
    checkpoint = _load_checkpoint(weights, device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    data_options = copy.deepcopy(options["data"]["val-l1080p"]["args"])
    data_options["anno_file"] = None
    data_options["data_prefix"] = str(video_dir.resolve())
    # The released samples contain 30 frames. This avoids excessive wrapping in
    # the original 96-frame technical-view sampling setup.
    data_options["sample_types"]["technical"]["num_clips"] = 1
    data_options["sample_types"]["technical"]["frame_interval"] = 1

    dataset = ViewDecompositionDataset(data_options)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    scores: Dict[str, float] = {}
    sample_types = ("aesthetic", "technical")
    try:
        for data in tqdm(loader, desc="DOVER"):
            if len(data) == 1:
                continue

            views = {}
            for key in sample_types:
                if key not in data:
                    continue
                view = data[key].to(device)
                batch, channels, frames, height, width = view.shape
                clips = int(data["num_clips"][key])
                views[key] = (
                    view.reshape(batch, channels, clips, frames // clips, height, width)
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(batch * clips, channels, frames // clips, height, width)
                )

            with torch.inference_mode():
                raw = model(views, reduce_scores=False)
                raw = [float(value.float().mean().cpu()) for value in raw]

            technical = (raw[1] - 0.1107) / 0.07355
            aesthetic = (raw[0] + 0.08285) / 0.03774
            fused = technical * 0.6104 + aesthetic * 0.3896
            name = Path(data["name"][0]).stem
            scores[name] = float(1.0 / (1.0 + np.exp(-fused)))
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return scores


def evaluate_fastervqa(
    videos: Mapping[str, Path],
    repo: Path,
    weights: Path,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate videos with the FasterVQA configuration used in the paper."""

    repo = _require_repo(repo, "fastvqa", "FAST-VQA/FasterVQA repository")
    weights = _require_file(weights, "FasterVQA checkpoint")
    config_path = _require_file(
        repo / "options" / "fast" / "f3dvqa-b.yml",
        "FasterVQA configuration",
    )
    _prepend_import_path(repo)

    try:
        import decord
        from fastvqa.datasets import FragmentSampleFrames, SampleFrames, get_spatial_fragments
        from fastvqa.models import DiViDeAddEvaluator
    except ImportError as error:
        raise RuntimeError(
            "FasterVQA dependencies could not be imported. Install requirements.txt and rerun "
            "setup_metrics.sh."
        ) from error

    decord.bridge.set_bridge("torch")
    with config_path.open("r", encoding="utf-8") as handle:
        options = yaml.safe_load(handle)

    model = DiViDeAddEvaluator(**options["model"]["args"]).to(device)
    checkpoint = _load_checkpoint(weights, device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    data_options = options["data"]["val-kv1k"]["args"]
    sample_options = data_options["sample_types"]
    mean = torch.tensor([123.675, 116.28, 103.53])
    std = torch.tensor([58.395, 57.12, 57.375])
    score_mean, score_std = 0.14759505, 0.03613452
    scores: Dict[str, float] = {}

    try:
        for name, video_path in tqdm(videos.items(), desc="FasterVQA"):
            reader = decord.VideoReader(str(video_path))
            if len(reader) == 0:
                raise ValueError(f"No frames could be decoded from {video_path}")

            samples = {}
            for sample_type, sample_args in sample_options.items():
                temporal_fragments = data_options.get("t_frag", 1)
                if temporal_fragments > 1:
                    sampler = FragmentSampleFrames(
                        fsize_t=sample_args["clip_len"] // temporal_fragments,
                        fragments_t=temporal_fragments,
                        num_clips=sample_args.get("num_clips", 1),
                    )
                else:
                    sampler = SampleFrames(
                        clip_len=sample_args["clip_len"],
                        num_clips=sample_args.get("num_clips", 1),
                    )

                num_clips = sample_args.get("num_clips", 1)
                frame_indices = sampler(len(reader))
                frame_cache = {frame: reader[int(frame)] for frame in np.unique(frame_indices)}
                video = torch.stack([frame_cache[frame] for frame in frame_indices], dim=0)
                video = video.permute(3, 0, 1, 2)
                sampled = get_spatial_fragments(video, **sample_args)
                sampled = ((sampled.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
                sampled = sampled.reshape(
                    sampled.shape[0], num_clips, -1, *sampled.shape[2:]
                ).transpose(0, 1)
                samples[sample_type] = sampled.to(device)

            with torch.inference_mode():
                raw_score = float(model(samples).float().mean().cpu())
            normalized = (raw_score - score_mean) / score_std
            scores[name] = float(1.0 / (1.0 + np.exp(-normalized)))
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return scores
