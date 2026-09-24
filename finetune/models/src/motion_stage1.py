from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from .loss import MotionDecouplingLoss
from .motion_decoupler import MotionDecoupler
from .motion_enhancement import ObjectMotionGuidedLatentPropagation
from .motion_uncertainty import MotionUncertaintyEstimator
from .reliability_dit_injection import ReliabilityDiTInjector
from .stocp import STOCPPriorEstimator
from .turb_rectifier import TurbulenceAwareLatentRectifier


MOTION_STATE_PREFIX = "camo_motion."


def default_motion_config() -> Dict[str, Any]:
    return {
        "motion_pipeline": {
            "enable_decoupling": True,
            "enable_turbulence_rectifier": True,
            "enable_object_motion_enhancement": True,
            "enable_reliability_injection": True,
            "enable_reliability_velocity_scaling": True,
            "motion_loss_weight": 0.5,
            "upsample_mode": "bilinear",
            "estimator": {
                "embed_dims": [32, 64, 128, 256],
                "motion_dims": [0, 0, 64, 64],
                "num_heads": [8, 16],
                "depths": [2, 2, 6, 2],
            },
            "loss": {
                "lambda_obj": 1.0,
                "lambda_zero": 1.0,
                "lambda_ratio": 1.0,
                "ratio_eps": 1e-6,
            },
            "adapter": {
                "hidden_channels": 32,
                "temporal_kernel": 5,
                "num_heads": 4,
                "window_size": 8,
                "num_refine_blocks": 2,
                "align_corners": False,
            },
            "uncertainty": {
                "hidden_channels": 16,
                "normalize_cues": True,
                "init_uncertainty": 0.5,
            },
            "stocp": {
                "enable": True,
                "uncertainty_weight": 0.5,
                "patch_size": 32,
                "stride": 8,
                "max_shift": 16,
                "radiance_gamma": 0.6,
                "norm_mode": "fixed",
                "phi_scale": 1.39,
                "rad_scale": 0.185,
                "edge_scale": 0.258,
                "temporal_downsample": 4,
            },
            "reliability_injection": {
                "hidden_channels": 32,
            },
            "decoupler_layers": 8,
            "rectifier_layers": 8,
            "enhancer_layers": 8,
        }
    }


def _recursive_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _recursive_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_motion_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = cfg["motion_pipeline"]
    if (
        p["enable_turbulence_rectifier"]
        or p["enable_object_motion_enhancement"]
        or p["enable_reliability_injection"]
        or p["enable_reliability_velocity_scaling"]
    ) and not p["enable_decoupling"]:
        raise ValueError(
            "Invalid motion config: enable_turbulence_rectifier/enable_object_motion_enhancement/"
            "enable_reliability_injection/enable_reliability_velocity_scaling "
            "requires enable_decoupling=True"
        )

    if p["enable_decoupling"] and int(p.get("decoupler_layers", 1)) < 1:
        raise ValueError("decoupler_layers must be >= 1 when enable_decoupling is True")
    if p["enable_turbulence_rectifier"] and int(p.get("rectifier_layers", 1)) < 1:
        raise ValueError("rectifier_layers must be >= 1 when enable_turbulence_rectifier is True")
    if p["enable_object_motion_enhancement"] and int(p.get("enhancer_layers", 1)) < 1:
        raise ValueError("enhancer_layers must be >= 1 when enable_object_motion_enhancement is True")

    stocp = p.get("stocp", {})
    if stocp.get("norm_mode", "fixed") not in {"fixed", "robust"}:
        raise ValueError("motion_stocp_norm_mode must be 'fixed' or 'robust'")
    if min(
        float(stocp.get("phi_scale", 1.39)),
        float(stocp.get("rad_scale", 0.185)),
        float(stocp.get("edge_scale", 0.258)),
    ) <= 0.0:
        raise ValueError("motion_stocp_phi_scale/rad_scale/edge_scale must be positive")
    if int(stocp.get("temporal_downsample", 4)) < 1:
        raise ValueError("motion_stocp_temporal_downsample must be >= 1")

    return cfg


def build_motion_config_from_args(args: Any) -> Dict[str, Any]:
    cfg = default_motion_config()
    p = cfg["motion_pipeline"]

    p["enable_decoupling"] = bool(
        getattr(args, "enable_motion_decoupling", p["enable_decoupling"])
    )
    p["enable_turbulence_rectifier"] = bool(
        getattr(args, "enable_motion_turbulence_rectifier", p["enable_turbulence_rectifier"])
    )
    p["enable_object_motion_enhancement"] = bool(
        getattr(args, "enable_motion_object_enhancement", p["enable_object_motion_enhancement"])
    )
    p["enable_reliability_injection"] = bool(
        getattr(args, "enable_motion_reliability_injection", p["enable_reliability_injection"])
    )
    p["enable_reliability_velocity_scaling"] = bool(
        getattr(
            args,
            "enable_motion_reliability_velocity_scaling",
            p["enable_reliability_velocity_scaling"],
        )
    )
    p["motion_loss_weight"] = float(getattr(args, "motion_loss_weight", p["motion_loss_weight"]))
    p["upsample_mode"] = str(getattr(args, "motion_upsample_mode", p["upsample_mode"]))

    p["estimator"]["embed_dims"] = list(
        getattr(args, "motion_estimator_embed_dims", p["estimator"]["embed_dims"])
    )
    p["estimator"]["motion_dims"] = list(
        getattr(args, "motion_estimator_motion_dims", p["estimator"]["motion_dims"])
    )
    p["estimator"]["num_heads"] = list(
        getattr(args, "motion_estimator_num_heads", p["estimator"]["num_heads"])
    )
    p["estimator"]["depths"] = list(
        getattr(args, "motion_estimator_depths", p["estimator"]["depths"])
    )

    p["loss"]["lambda_obj"] = float(
        getattr(args, "motion_loss_lambda_obj", p["loss"]["lambda_obj"])
    )
    p["loss"]["lambda_zero"] = float(
        getattr(args, "motion_loss_lambda_zero", p["loss"]["lambda_zero"])
    )
    p["loss"]["lambda_ratio"] = float(
        getattr(args, "motion_loss_lambda_ratio", p["loss"]["lambda_ratio"])
    )
    p["loss"]["ratio_eps"] = float(
        getattr(args, "motion_loss_ratio_eps", p["loss"]["ratio_eps"])
    )

    p["adapter"]["hidden_channels"] = int(
        getattr(args, "motion_adapter_hidden_channels", p["adapter"]["hidden_channels"])
    )
    p["adapter"]["temporal_kernel"] = int(
        getattr(args, "motion_adapter_temporal_kernel", p["adapter"]["temporal_kernel"])
    )
    p["adapter"]["num_heads"] = int(
        getattr(args, "motion_adapter_num_heads", p["adapter"]["num_heads"])
    )
    p["adapter"]["window_size"] = int(
        getattr(args, "motion_adapter_window_size", p["adapter"]["window_size"])
    )
    p["adapter"]["num_refine_blocks"] = int(
        getattr(args, "motion_adapter_num_refine_blocks", p["adapter"]["num_refine_blocks"])
    )
    p["adapter"]["align_corners"] = bool(
        getattr(args, "motion_adapter_align_corners", p["adapter"]["align_corners"])
    )

    p["uncertainty"]["hidden_channels"] = int(
        getattr(args, "motion_uncertainty_hidden_channels", p["uncertainty"]["hidden_channels"])
    )
    p["uncertainty"]["normalize_cues"] = bool(
        getattr(args, "motion_uncertainty_normalize_cues", p["uncertainty"]["normalize_cues"])
    )
    p["uncertainty"]["init_uncertainty"] = float(
        getattr(args, "motion_uncertainty_init", p["uncertainty"]["init_uncertainty"])
    )

    p["stocp"]["enable"] = bool(
        getattr(args, "enable_motion_stocp_uncertainty", p["stocp"]["enable"])
    )
    p["stocp"]["uncertainty_weight"] = float(
        getattr(args, "motion_stocp_uncertainty_weight", p["stocp"]["uncertainty_weight"])
    )
    p["stocp"]["patch_size"] = int(
        getattr(args, "motion_stocp_patch_size", p["stocp"]["patch_size"])
    )
    p["stocp"]["stride"] = int(
        getattr(args, "motion_stocp_stride", p["stocp"]["stride"])
    )
    p["stocp"]["max_shift"] = int(
        getattr(args, "motion_stocp_max_shift", p["stocp"]["max_shift"])
    )
    p["stocp"]["radiance_gamma"] = float(
        getattr(args, "motion_stocp_radiance_gamma", p["stocp"]["radiance_gamma"])
    )
    p["stocp"]["norm_mode"] = str(
        getattr(args, "motion_stocp_norm_mode", p["stocp"]["norm_mode"])
    )
    p["stocp"]["phi_scale"] = float(
        getattr(args, "motion_stocp_phi_scale", p["stocp"]["phi_scale"])
    )
    p["stocp"]["rad_scale"] = float(
        getattr(args, "motion_stocp_rad_scale", p["stocp"]["rad_scale"])
    )
    p["stocp"]["edge_scale"] = float(
        getattr(args, "motion_stocp_edge_scale", p["stocp"]["edge_scale"])
    )
    p["stocp"]["temporal_downsample"] = int(
        getattr(args, "motion_stocp_temporal_downsample", p["stocp"]["temporal_downsample"])
    )

    p["reliability_injection"]["hidden_channels"] = int(
        getattr(
            args,
            "motion_reliability_injection_hidden_channels",
            p["reliability_injection"]["hidden_channels"],
        )
    )

    p["decoupler_layers"] = int(
        getattr(args, "motion_decoupler_layers", p.get("decoupler_layers", 1))
    )
    p["rectifier_layers"] = int(
        getattr(args, "motion_rectifier_layers", p.get("rectifier_layers", 1))
    )
    p["enhancer_layers"] = int(
        getattr(args, "motion_enhancer_layers", p.get("enhancer_layers", 1))
    )

    _validate_motion_config(cfg)
    return cfg


def is_motion_enabled(config: Dict[str, Any]) -> bool:
    p = config["motion_pipeline"]
    return bool(
        p["enable_decoupling"]
        or p["enable_turbulence_rectifier"]
        or p["enable_object_motion_enhancement"]
        or p["enable_reliability_injection"]
        or p["enable_reliability_velocity_scaling"]
    )


class Stage1MotionPipeline(nn.Module):
    def __init__(self, config: Dict[str, Any], latent_channels: int) -> None:
        super().__init__()
        p = config["motion_pipeline"]

        self.enable_decoupling = bool(p["enable_decoupling"])
        self.enable_turbulence_rectifier = bool(p["enable_turbulence_rectifier"])
        self.enable_object_motion_enhancement = bool(p["enable_object_motion_enhancement"])
        self.enable_reliability_injection = bool(p["enable_reliability_injection"])
        self.enable_reliability_velocity_scaling = bool(p["enable_reliability_velocity_scaling"])
        self.motion_loss_weight = float(p["motion_loss_weight"])
        self.upsample_mode = str(p["upsample_mode"])

        self.latent_channels = latent_channels
        motion_channels = latent_channels
        self.decoupler_layers = int(p.get("decoupler_layers", 1))
        self.motion_decouplers = nn.ModuleList(
            [MotionDecoupler(channels=motion_channels) for _ in range(self.decoupler_layers)]
        )

        loss_cfg = p["loss"]
        self.motion_loss = MotionDecouplingLoss(
            lambda_obj=float(loss_cfg["lambda_obj"]),
            lambda_zero=float(loss_cfg["lambda_zero"]),
            lambda_ratio=float(loss_cfg.get("lambda_ratio", 0.0)),
            ratio_eps=float(loss_cfg.get("ratio_eps", 1e-6)),
            reduction="mean",
        )

        adapter_cfg = p["adapter"]
        hidden_channels = int(adapter_cfg["hidden_channels"])
        temporal_kernel = int(adapter_cfg["temporal_kernel"])
        num_heads = int(adapter_cfg.get("num_heads", 4))
        window_size = adapter_cfg.get("window_size", 8)
        num_refine_blocks = int(adapter_cfg["num_refine_blocks"])
        align_corners = bool(adapter_cfg["align_corners"])

        self.rectifier_layers = int(p.get("rectifier_layers", 1))
        self.turb_rectifiers = nn.ModuleList(
            [
                TurbulenceAwareLatentRectifier(
                    z_channels=latent_channels,
                    motion_channels=motion_channels,
                    hidden_channels=hidden_channels,
                    proj_channels=hidden_channels,
                    temporal_kernel=temporal_kernel,
                    align_corners=align_corners,
                )
                for _ in range(self.rectifier_layers)
            ]
            if self.enable_turbulence_rectifier
            else []
        )

        self.enhancer_layers = int(p.get("enhancer_layers", 1))
        self.motion_enhancers = nn.ModuleList(
            [
                ObjectMotionGuidedLatentPropagation(
                    latent_channels=latent_channels,
                    motion_channels=motion_channels,
                    hidden_channels=hidden_channels,
                    temporal_kernel=temporal_kernel,
                    num_heads=num_heads,
                    window_size=window_size,
                    num_refine_blocks=num_refine_blocks,
                    align_corners=align_corners,
                )
                for _ in range(self.enhancer_layers)
            ]
            if self.enable_object_motion_enhancement
            else []
        )

        uncertainty_cfg = p["uncertainty"]
        self.motion_uncertainty = (
            MotionUncertaintyEstimator(
                channels=motion_channels,
                hidden_channels=int(uncertainty_cfg["hidden_channels"]),
                normalize_cues=bool(uncertainty_cfg["normalize_cues"]),
                init_uncertainty=float(uncertainty_cfg["init_uncertainty"]),
            )
            if self.enable_reliability_injection or self.enable_reliability_velocity_scaling
            else None
        )
        self.reliability_injection_hidden_channels = int(
            p["reliability_injection"]["hidden_channels"]
        )
        self.dit_reliability_injector: Optional[ReliabilityDiTInjector] = None

        stocp_cfg = p.get("stocp", {})
        self.enable_stocp_uncertainty = self.enable_decoupling and bool(stocp_cfg.get("enable", True))
        self.stocp_uncertainty_weight = float(stocp_cfg.get("uncertainty_weight", 0.5))
        if not (0.0 <= self.stocp_uncertainty_weight <= 1.0):
            raise ValueError("motion_stocp_uncertainty_weight must be in [0, 1]")
        self.stocp_temporal_downsample = int(stocp_cfg.get("temporal_downsample", 4))
        self.stocp_estimator = (
            STOCPPriorEstimator(
                patch_size=int(stocp_cfg.get("patch_size", 32)),
                stride=int(stocp_cfg.get("stride", 8)),
                max_shift=int(stocp_cfg.get("max_shift", 16)),
                radiance_gamma=float(stocp_cfg.get("radiance_gamma", 0.6)),
                norm_mode=str(stocp_cfg.get("norm_mode", "fixed")),
                phi_scale=float(stocp_cfg.get("phi_scale", 1.39)),
                rad_scale=float(stocp_cfg.get("rad_scale", 0.185)),
                edge_scale=float(stocp_cfg.get("edge_scale", 0.258)),
            )
            if self.enable_stocp_uncertainty
            else None
        )
        if self.stocp_estimator is not None:
            self.stocp_estimator.requires_grad_(False)
            self.stocp_estimator.eval()

    @property
    def enabled(self) -> bool:
        return (
            self.enable_decoupling
            or self.enable_turbulence_rectifier
            or self.enable_object_motion_enhancement
            or self.enable_reliability_injection
            or self.enable_reliability_velocity_scaling
        )

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        for p in module.parameters():
            return p.dtype
        return torch.float32

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        for p in module.parameters():
            return p.device
        return torch.device("cpu")

    def _resize_spatial(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        h, w = target_hw
        if x.shape[-2:] == (h, w):
            return x

        b, c, f, _, _ = x.shape
        x_4d = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, x.shape[-2], x.shape[-1])

        if self.upsample_mode in {"bilinear", "bicubic"}:
            x_4d = F.interpolate(x_4d, size=(h, w), mode=self.upsample_mode, align_corners=False)
        else:
            x_4d = F.interpolate(x_4d, size=(h, w), mode=self.upsample_mode)

        return x_4d.view(b, f, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def _to_motion_dtype(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self._module_dtype(self.motion_decouplers)
        device = self._module_device(self.motion_decouplers)
        return x.to(device=device, dtype=dtype)

    @staticmethod
    def _stocp_compute_dtype(dtype: torch.dtype) -> torch.dtype:
        if dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return dtype

    @staticmethod
    def _as_b1fhw(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"{name} must be 5D [B,1,F,H,W] or [B,F,1,H,W], got {tuple(x.shape)}")
        if x.shape[1] != 1 and x.shape[2] == 1:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        if x.shape[1] != 1:
            raise ValueError(f"{name} channel dimension must be 1, got {tuple(x.shape)}")
        return x

    @classmethod
    def _align_uncertainty_to_bcfhw(
        cls,
        uncertainty: torch.Tensor,
        target_shape_bcfhw,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, _channels, num_frames, height, width = [int(v) for v in target_shape_bcfhw]
        uncertainty = cls._as_b1fhw(uncertainty, "uncertainty")
        uncertainty = uncertainty.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        if uncertainty.shape[0] != batch_size:
            raise ValueError(
                f"uncertainty batch size {uncertainty.shape[0]} does not match target batch {batch_size}"
            )
        if uncertainty.shape[2:] != (num_frames, height, width):
            uncertainty = F.interpolate(
                uncertainty,
                size=(num_frames, height, width),
                mode="trilinear",
                align_corners=False,
            )
        return uncertainty.clamp(0.0, 1.0)

    @staticmethod
    def _prepare_stocp_input(
        lq_video: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if lq_video.dim() != 5:
            raise ValueError(f"lq_video must be [B,C,F,H,W], got {tuple(lq_video.shape)}")
        if not lq_video.is_floating_point():
            raise TypeError(f"lq_video must be floating point, got {lq_video.dtype}")

        # Dataset videos are normalized to [-1, 1]. STOCP consumes [0, 1]
        # in [B, F, C, H, W] order.
        video_01 = lq_video.detach().to(device=device, dtype=dtype)
        video_01 = video_01.mul(0.5).add(0.5).clamp(0.0, 1.0)
        return video_01.permute(0, 2, 1, 3, 4).contiguous()

    def estimate_stocp_uncertainty(
        self,
        lq_video: torch.Tensor,
        latent_shape_bcfhw,
    ) -> Optional[torch.Tensor]:
        if not self.enable_stocp_uncertainty or self.stocp_estimator is None:
            return None

        module_dtype = self._module_dtype(self.motion_decouplers)
        module_device = self._module_device(self.motion_decouplers)
        compute_dtype = self._stocp_compute_dtype(module_dtype)
        _batch_size, _channels, _num_frames, latent_h, latent_w = [int(v) for v in latent_shape_bcfhw]

        with torch.no_grad():
            self.stocp_estimator.to(device=module_device, dtype=compute_dtype)
            video_01 = self._prepare_stocp_input(
                lq_video,
                device=module_device,
                dtype=compute_dtype,
            )
            stocp_out = self.stocp_estimator(
                video_01,
                latent_hw=(latent_h, latent_w),
                latent_temporal_downsample=self.stocp_temporal_downsample,
            )
            q_t_l = stocp_out["Q_latent"].permute(0, 2, 1, 3, 4).contiguous()
            q_t_l = self._align_uncertainty_to_bcfhw(
                q_t_l,
                latent_shape_bcfhw,
                device=module_device,
                dtype=module_dtype,
            )
        return q_t_l.detach()

    @staticmethod
    def latent_temporal_residual(z: torch.Tensor) -> torch.Tensor:
        """
        Fixed latent-space temporal residual used as the implicit motion/degradation carrier.

        Args:
            z: [B, C, F, H, W]

        Returns:
            dz: [B, C, F, H, W], where the final frame is padded with zeros.
        """
        if z.dim() != 5:
            raise ValueError(
                f"latent_temporal_residual expects [B, C, F, H, W], got {tuple(z.shape)}"
            )
        if z.shape[2] < 2:
            return torch.zeros_like(z)
        dz = z[:, :, 1:] - z[:, :, :-1]
        pad = torch.zeros_like(z[:, :, :1])
        return torch.cat([dz, pad], dim=2)

    def _extract_motion_residual(self, latent: torch.Tensor) -> torch.Tensor:
        latent = self._to_motion_dtype(latent)
        return self.latent_temporal_residual(latent)

    def _run_decouplers(self, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m_obj = m
        m_turb_total = torch.zeros_like(m)
        for decoupler in self.motion_decouplers:
            m_obj_next, m_turb = decoupler(m_obj)
            m_turb_total = m_turb_total + m_turb
            m_obj = m_obj_next
        if not self.motion_decouplers:
            raise RuntimeError("No motion decoupler available to process motion features")
        return m_obj, m_turb_total

    def extract_lr_motion(
        self,
        lq_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_decoupling:
            raise RuntimeError("Motion decoupling is disabled in current config")

        m_lr = self._extract_motion_residual(lq_latent)
        m_obj_lr, m_turb_lr = self._run_decouplers(m_lr)
        return m_obj_lr, m_turb_lr

    def extract_motion_and_loss(
        self,
        hr_latent: torch.Tensor,
        lq_latent: torch.Tensor,
        q_stocp_l: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enable_decoupling:
            raise RuntimeError("Motion decoupling is disabled in current config")

        hr_motion_in = self._to_motion_dtype(hr_latent)
        lq_motion_in = self._to_motion_dtype(lq_latent)
        if lq_motion_in.shape[-2:] != hr_motion_in.shape[-2:]:
            lq_motion_in = self._resize_spatial(lq_motion_in, hr_motion_in.shape[-2:])

        m_hr = self.latent_temporal_residual(hr_motion_in)
        m_lr = self.latent_temporal_residual(lq_motion_in)

        m_obj_hr, m_turb_hr = self._run_decouplers(m_hr)
        m_obj_lr, m_turb_lr = self._run_decouplers(m_lr)

        motion_loss, motion_loss_dict = self.motion_loss(
            m_obj_hr=m_obj_hr,
            m_turb_hr=m_turb_hr,
            m_obj_lr=m_obj_lr,
            m_turb_lr=m_turb_lr,
            q_t_l=q_stocp_l,
        )

        return m_obj_lr, m_turb_lr, motion_loss, motion_loss_dict

    def estimate_reliability(
        self,
        m_obj_lr: torch.Tensor,
        m_turb_lr: torch.Tensor,
        q_stocp_l: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            not self.enable_reliability_injection
            and not self.enable_reliability_velocity_scaling
        ) or self.motion_uncertainty is None:
            raise RuntimeError("Motion reliability estimation is disabled in current config")

        module_dtype = self._module_dtype(self.motion_uncertainty)
        module_device = self._module_device(self.motion_uncertainty)
        m_obj_lr = m_obj_lr.to(device=module_device, dtype=module_dtype)
        m_turb_lr = m_turb_lr.to(device=module_device, dtype=module_dtype)
        uncertainty = self.motion_uncertainty(m_obj_lr, m_turb_lr)

        if (
            q_stocp_l is not None
            and self.enable_stocp_uncertainty
            and self.stocp_uncertainty_weight > 0.0
        ):
            q_stocp_l = self._align_uncertainty_to_bcfhw(
                q_stocp_l.detach(),
                target_shape_bcfhw=m_obj_lr.shape,
                device=module_device,
                dtype=module_dtype,
            )
            w = self.stocp_uncertainty_weight
            uncertainty = (1.0 - w) * uncertainty + w * q_stocp_l

        return 1.0 - uncertainty.clamp(0.0, 1.0)

    def attach_dit_reliability_injector(self, transformer: nn.Module) -> None:
        if not self.enable_reliability_injection:
            return

        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None:
            raise ValueError("Transformer does not expose transformer_blocks for reliability injection.")

        config = getattr(transformer, "config", None)
        if config is None:
            raise ValueError("Transformer config is required for reliability injection.")

        hidden_size = int(config.num_attention_heads) * int(config.attention_head_dim)
        patch_size = int(getattr(config, "patch_size", 1))
        patch_size_t = getattr(config, "patch_size_t", None)
        if patch_size_t is not None:
            patch_size_t = int(patch_size_t)

        if self.dit_reliability_injector is None:
            self.dit_reliability_injector = ReliabilityDiTInjector(
                num_layers=len(blocks),
                hidden_size=hidden_size,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
                hidden_channels=self.reliability_injection_hidden_channels,
            )

        self.dit_reliability_injector.attach(transformer)

    def prepare_dit_reliability_context(
        self,
        rho_t: Optional[torch.Tensor],
        latent_shape_bfchw,
    ) -> None:
        if (
            not self.enable_reliability_injection
            or self.dit_reliability_injector is None
            or rho_t is None
        ):
            return
        self.dit_reliability_injector.prepare_context(rho_t, latent_shape_bfchw)

    def clear_dit_reliability_context(self) -> None:
        if self.dit_reliability_injector is not None:
            self.dit_reliability_injector.clear_context()

    def scale_velocity_by_reliability(
        self,
        velocity: torch.Tensor,
        rho_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.enable_reliability_velocity_scaling or rho_t is None:
            return velocity
        if velocity.dim() != 5:
            raise ValueError(f"velocity must be [B, F, C, H, W], got {tuple(velocity.shape)}")
        if rho_t.dim() != 5:
            raise ValueError(f"rho_t must be [B, 1, F, H, W], got {tuple(rho_t.shape)}")

        batch_size, num_frames, _channels, height, width = velocity.shape
        rho = rho_t.to(device=velocity.device, dtype=velocity.dtype).clamp(0.0, 1.0)
        if rho.shape[0] != batch_size:
            raise ValueError(f"rho_t batch size {rho.shape[0]} does not match velocity batch {batch_size}")
        if rho.shape[2:] != (num_frames, height, width):
            rho = F.interpolate(
                rho,
                size=(num_frames, height, width),
                mode="trilinear",
                align_corners=False,
            )
        rho = rho.permute(0, 2, 1, 3, 4).contiguous()
        return rho * velocity

    def rectify_latent(self, z_lr: torch.Tensor, m_turb_lr: torch.Tensor) -> torch.Tensor:
        if not self.enable_turbulence_rectifier or not self.turb_rectifiers:
            return z_lr

        in_dtype = z_lr.dtype
        in_device = z_lr.device
        z_rect = z_lr
        for rectifier in self.turb_rectifiers:
            module_dtype = self._module_dtype(rectifier)
            module_device = self._module_device(rectifier)
            z_rect = rectifier(
                z_rect.to(device=module_device, dtype=module_dtype),
                m_turb_lr.to(device=module_device, dtype=module_dtype),
            )
        return z_rect.to(device=in_device, dtype=in_dtype)

    def enhance_latent(self, z_hat: torch.Tensor, m_obj_lr: torch.Tensor) -> torch.Tensor:
        if not self.enable_object_motion_enhancement or not self.motion_enhancers:
            return z_hat

        in_dtype = z_hat.dtype
        in_device = z_hat.device
        z_enhanced = z_hat
        for enhancer in self.motion_enhancers:
            module_dtype = self._module_dtype(enhancer)
            module_device = self._module_device(enhancer)
            z_enhanced = enhancer(
                z_enhanced.to(device=module_device, dtype=module_dtype),
                m_obj_lr.to(device=module_device, dtype=module_dtype),
            )
        return z_enhanced.to(device=in_device, dtype=in_dtype)


def _read_prefixed_state_from_index(
    transformer_dir: Path,
    prefix: str,
    expected_keys: Optional[set[str]] = None,
) -> Dict[str, torch.Tensor]:
    index_candidates = [
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
    ]

    for index_name in index_candidates:
        index_path = transformer_dir / index_name
        if not index_path.exists():
            continue

        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        prefixed_keys = [
            key
            for key in weight_map
            if key.startswith(prefix)
            and (expected_keys is None or key[len(prefix) :] in expected_keys)
        ]
        if not prefixed_keys:
            return {}

        grouped_files: Dict[str, list[str]] = {}
        for key in prefixed_keys:
            grouped_files.setdefault(weight_map[key], []).append(key)

        state_dict: Dict[str, torch.Tensor] = {}
        for shard_file, shard_keys in grouped_files.items():
            shard_path = transformer_dir / shard_file
            with safe_open(shard_path.as_posix(), framework="pt", device="cpu") as f:
                for key in shard_keys:
                    state_dict[key[len(prefix) :]] = f.get_tensor(key)

        return state_dict

    return {}


def _read_prefixed_state_from_single_file(
    transformer_dir: Path,
    prefix: str,
    expected_keys: Optional[set[str]] = None,
) -> Dict[str, torch.Tensor]:
    file_candidates = ["diffusion_pytorch_model.safetensors", "model.safetensors"]
    for filename in file_candidates:
        ckpt_path = transformer_dir / filename
        if not ckpt_path.exists():
            continue

        state_dict: Dict[str, torch.Tensor] = {}
        with safe_open(ckpt_path.as_posix(), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix) and (
                    expected_keys is None or key[len(prefix) :] in expected_keys
                ):
                    state_dict[key[len(prefix) :]] = f.get_tensor(key)
        return state_dict

    return {}


def _read_checkpoint_weight_keys(transformer_dir: Path) -> list[str]:
    index_candidates = [
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
    ]
    for index_name in index_candidates:
        index_path = transformer_dir / index_name
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                return list(json.load(f).get("weight_map", {}).keys())

    file_candidates = ["diffusion_pytorch_model.safetensors", "model.safetensors"]
    for filename in file_candidates:
        checkpoint_path = transformer_dir / filename
        if checkpoint_path.exists():
            with safe_open(checkpoint_path.as_posix(), framework="pt", device="cpu") as f:
                return list(f.keys())
    return []


def _infer_motion_state_prefix(
    checkpoint_keys: list[str],
    expected_keys: set[str],
    preferred_prefix: str,
) -> Optional[str]:
    preferred_prefix = preferred_prefix.rstrip(".") + "."

    def coverage(prefix: str) -> int:
        return sum(
            key.startswith(prefix) and key[len(prefix) :] in expected_keys
            for key in checkpoint_keys
        )

    if coverage(preferred_prefix):
        return preferred_prefix
    if any(key in expected_keys for key in checkpoint_keys):
        return ""

    anchor_roots = (
        "motion_decouplers.0.",
        "turb_rectifiers.0.",
        "motion_enhancers.0.",
        "dit_reliability_injector.",
    )
    anchors = sorted(
        key for key in expected_keys if key.startswith(anchor_roots)
    )[:16]
    candidates = {
        checkpoint_key[: -len(anchor)]
        for checkpoint_key in checkpoint_keys
        for anchor in anchors
        if checkpoint_key.endswith(anchor)
    }
    if not candidates:
        return None

    detected = max(candidates, key=coverage)
    return detected if coverage(detected) else None


def load_motion_pipeline_weights(
    motion_pipeline: Stage1MotionPipeline,
    transformer_dir: Path | str,
    state_prefix: str = MOTION_STATE_PREFIX,
) -> bool:
    transformer_dir = Path(transformer_dir)
    if not transformer_dir.exists():
        return False

    preferred_prefix = state_prefix.rstrip(".") + "."
    current_state = motion_pipeline.state_dict()
    expected_keys = set(current_state)

    motion_file = transformer_dir / "motion_modules.safetensors"
    state_dict: Dict[str, torch.Tensor] = {}

    if motion_file.exists():
        raw = load_file(motion_file.as_posix())
        detected_prefix = _infer_motion_state_prefix(
            list(raw), expected_keys, preferred_prefix
        )
        if detected_prefix is None:
            return False
        for k, v in raw.items():
            local_key = k[len(detected_prefix) :] if detected_prefix else k
            if local_key in expected_keys:
                state_dict[local_key] = v
    else:
        checkpoint_keys = _read_checkpoint_weight_keys(transformer_dir)
        detected_prefix = _infer_motion_state_prefix(
            checkpoint_keys, expected_keys, preferred_prefix
        )
        if detected_prefix is None:
            return False
        state_dict = _read_prefixed_state_from_index(
            transformer_dir, detected_prefix, expected_keys
        )
        if not state_dict:
            state_dict = _read_prefixed_state_from_single_file(
                transformer_dir, detected_prefix, expected_keys
            )

    if not state_dict:
        return False

    compatible_state_dict: Dict[str, torch.Tensor] = {}
    skipped_shape_keys: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key, value in state_dict.items():
        if key not in current_state:
            continue
        if tuple(value.shape) != tuple(current_state[key].shape):
            skipped_shape_keys.append((key, tuple(value.shape), tuple(current_state[key].shape)))
            continue
        compatible_state_dict[key] = value

    missing_keys = sorted(expected_keys - set(compatible_state_dict))
    if skipped_shape_keys or missing_keys:
        details = []
        if missing_keys:
            details.append(f"missing {len(missing_keys)} keys")
        if skipped_shape_keys:
            details.append(f"shape mismatch for {len(skipped_shape_keys)} keys")
        raise RuntimeError(
            "The motion checkpoint is incomplete or incompatible ("
            + ", ".join(details)
            + "). Use the complete released CAMO checkpoint."
        )

    motion_pipeline.load_state_dict(compatible_state_dict, strict=True)
    return True
