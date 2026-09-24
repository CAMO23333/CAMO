from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RhoTokenAdaLNModulation(nn.Module):
    """
    AdaLN-style residual modulation generated only from reliability tokens.

    hidden_states: [B, S, D]
    rho_tokens:    [B, S, 1], where larger values mean more reliable motion decomposition.
    """

    def __init__(self, hidden_size: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.hidden_channels = int(hidden_channels)
        self.net = nn.Sequential(
            nn.Linear(1, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_size * 2),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, hidden_states: torch.Tensor, rho_tokens: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 3:
            return hidden_states

        rho_tokens = rho_tokens.to(device=hidden_states.device)
        if rho_tokens.shape[1] != hidden_states.shape[1]:
            return hidden_states

        # The injector can be fp32 during mixed-precision training while the DiT
        # hidden states are bf16/fp16, so run the small MLP in its parameter dtype.
        net_dtype = self.net[0].weight.dtype
        scale_shift = self.net(rho_tokens.to(dtype=net_dtype)).to(dtype=hidden_states.dtype)
        delta_scale, delta_shift = scale_shift.chunk(2, dim=-1)
        rho_tokens = rho_tokens.to(dtype=hidden_states.dtype)
        return hidden_states * (1.0 + rho_tokens * delta_scale) + rho_tokens * delta_shift


class ReliabilityDiTInjector(nn.Module):
    """
    Inject rho_t into CogVideoX DiT blocks without passing motion/turbulence features.

    The module registers forward hooks on norm1/norm2 of each transformer block and
    applies an extra AdaLN-like residual scale/shift to video tokens only. It
    intentionally does not modify the original timestep AdaLN parameters inside the
    pretrained DiT.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        patch_size: int = 2,
        patch_size_t: Optional[int] = 2,
        hidden_channels: int = 32,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        self.patch_size_t = int(patch_size_t) if patch_size_t is not None else None
        self.hidden_channels = int(hidden_channels)
        self.align_corners = bool(align_corners)
        self.norm_names = ("norm1", "norm2")

        self.modulators = nn.ModuleList(
            [
                RhoTokenAdaLNModulation(
                    hidden_size=self.hidden_size,
                    hidden_channels=self.hidden_channels,
                )
                for _ in range(self.num_layers * len(self.norm_names))
            ]
        )

        self._rho_tokens: Optional[torch.Tensor] = None
        self._hook_handles: list[Any] = []

    def attach(self, transformer: nn.Module) -> None:
        if self._hook_handles:
            return

        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None:
            raise ValueError("Transformer does not expose transformer_blocks for reliability injection.")
        if len(blocks) != self.num_layers:
            raise ValueError(
                f"Reliability injector expected {self.num_layers} transformer blocks, got {len(blocks)}."
            )

        ref_param = next(blocks[0].parameters(), None)
        if ref_param is not None and ref_param.is_floating_point():
            self.to(device=ref_param.device, dtype=ref_param.dtype)

        for layer_idx, block in enumerate(blocks):
            for norm_idx, norm_name in enumerate(self.norm_names):
                norm_module = getattr(block, norm_name, None)
                if norm_module is None:
                    continue
                mod_idx = layer_idx * len(self.norm_names) + norm_idx
                self._hook_handles.append(norm_module.register_forward_hook(self._make_hook(mod_idx)))

        if not self._hook_handles:
            raise ValueError("No norm1/norm2 modules were found for reliability injection.")

    def detach_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _make_hook(self, mod_idx: int):
        def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
            rho_tokens = self._rho_tokens
            if rho_tokens is None:
                return output

            if isinstance(output, tuple):
                if not output or not torch.is_tensor(output[0]):
                    return output
                hidden_states = self.modulators[mod_idx](output[0], rho_tokens)
                return (hidden_states, *output[1:])

            if torch.is_tensor(output):
                return self.modulators[mod_idx](output, rho_tokens)

            return output

        return hook

    def prepare_context(
        self,
        rho_t: torch.Tensor,
        latent_shape_bfchw: Sequence[int],
    ) -> None:
        """
        Args:
            rho_t: [B, 1, F, H, W]
            latent_shape_bfchw: DiT input latent shape [B, F, C, H, W]
        """
        if rho_t.dim() != 5:
            raise ValueError(f"rho_t must be [B, 1, F, H, W], got {tuple(rho_t.shape)}")
        if len(latent_shape_bfchw) != 5:
            raise ValueError(f"latent shape must be [B, F, C, H, W], got {tuple(latent_shape_bfchw)}")

        batch_size, num_frames, _channels, height, width = [int(v) for v in latent_shape_bfchw]
        if rho_t.shape[0] != batch_size:
            raise ValueError(f"rho_t batch size {rho_t.shape[0]} does not match latent batch {batch_size}")

        token_f = num_frames
        if self.patch_size_t is not None:
            token_f = (num_frames + self.patch_size_t - 1) // self.patch_size_t
        token_h = (height + self.patch_size - 1) // self.patch_size
        token_w = (width + self.patch_size - 1) // self.patch_size

        rho = rho_t.float().clamp(0.0, 1.0)
        rho = F.interpolate(
            rho,
            size=(token_f, token_h, token_w),
            mode="trilinear",
            align_corners=self.align_corners,
        )
        self._rho_tokens = rho.flatten(2).transpose(1, 2).contiguous()

    def clear_context(self) -> None:
        self._rho_tokens = None
