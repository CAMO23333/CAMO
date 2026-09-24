import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv3d(nn.Module):
    """
    A lightweight 3D local block:
        depthwise 3D conv -> pointwise 1x1x1 conv -> GELU
    """

    def __init__(self, channels: int, kernel_size=(3, 3, 3)):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)

        self.dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.act(x)
        return x


class LocalRefineBlock(nn.Module):
    """
    A local convolutional refinement block:
        [z, e_obj] -> 1x1x1 conv -> DWConv3D -> 1x1x1 conv -> residual add
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = None):
        super().__init__()
        hidden_channels = hidden_channels or out_channels

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            DepthwiseSeparableConv3d(hidden_channels),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # close to identity at init
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual + self.block(x)


class MotionTemporalAdapter(nn.Module):
    """
    Adapt object-motion features to latent channel/spatial layout without temporal downsampling.

    Input:
        m_obj: [B, C_m, F_m, H_m, W_m]

    Output:
        e_obj: [B, C_z, F_z, H_z, W_z]
    """

    def __init__(
        self,
        motion_channels: int,
        latent_channels: int,
        hidden_channels: int = None,
        temporal_kernel: int = 5,
        align_corners: bool = False,
    ):
        super().__init__()

        self.motion_channels = motion_channels
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels or latent_channels
        self.temporal_kernel = temporal_kernel
        self.align_corners = align_corners

        self.proj = nn.Sequential(
            nn.Conv3d(motion_channels, self.hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv3d(
                self.hidden_channels,
                latent_channels,
                kernel_size=(temporal_kernel, 3, 3),
                stride=(1, 1, 1),
                padding=(temporal_kernel // 2, 1, 1),
                bias=True,
            ),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, m_obj: torch.Tensor, target_shape) -> torch.Tensor:
        """
        Args:
            m_obj: [B, C_m, F_m, H_m, W_m]
            target_shape: shape of z_sr, i.e., [B, C_z, F_z, H_z, W_z]
        """
        _, _, Fz, Hz, Wz = target_shape

        # spatial alignment before channel projection
        if m_obj.shape[-2:] != (Hz, Wz):
            m_obj = F.interpolate(
                m_obj,
                size=(m_obj.shape[2], Hz, Wz),
                mode="trilinear",
                align_corners=self.align_corners,
            )

        e_obj = self.proj(m_obj)

        # force exact F/H/W alignment with z_sr
        if e_obj.shape[2:] != (Fz, Hz, Wz):
            e_obj = F.interpolate(
                e_obj,
                size=(Fz, Hz, Wz),
                mode="trilinear",
                align_corners=self.align_corners,
            )

        return e_obj


class MotionConditionedLatentAttention(nn.Module):
    """
    Motion-conditioned latent attention without explicit warping.

    For each latent frame z_t, the current latent queries attend to adjacent latent
    frames z_{t-1} and z_{t+1}. The adapted object-motion feature e_obj modulates
    Q and K with scale-shift parameters.

    Inputs:
        z:     [B, C, F, H, W]
        e_obj: [B, C, F, H, W]

    Output:
        z_prop: [B, C, F, H, W]
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = None,
        num_heads: int = 4,
        window_size: Union[int, Tuple[int, int]] = 8,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}.")

        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        self.channels = channels
        self.hidden_channels = hidden_channels or channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.q_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.k_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.v_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)

        self.q_norm = nn.LayerNorm(channels)
        self.k_norm = nn.LayerNorm(channels)

        # motion-conditioned scale-shift for Q and K
        self.q_mod = nn.Sequential(
            nn.Conv3d(channels, self.hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(self.hidden_channels, channels * 2, kernel_size=1, bias=True),
        )
        self.k_mod = nn.Sequential(
            nn.Conv3d(channels, self.hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(self.hidden_channels, channels * 2, kernel_size=1, bias=True),
        )

        # motion-gated fusion: z_prop = z + g * (z_attn - z)
        self.gate = nn.Sequential(
            nn.Conv3d(channels, self.hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(self.hidden_channels, channels, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # start from weak motion modulation and near-identity fusion
        nn.init.zeros_(self.q_mod[-1].weight)
        nn.init.zeros_(self.q_mod[-1].bias)
        nn.init.zeros_(self.k_mod[-1].weight)
        nn.init.zeros_(self.k_mod[-1].bias)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def _shift_prev(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z[:, :, :1], z[:, :, :-1]], dim=2)

    def _shift_next(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z[:, :, 1:], z[:, :, -1:]], dim=2)

    def _neighbor_masks(self, num_frames: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        prev_mask = torch.ones((1, 1, num_frames, 1, 1), device=device, dtype=dtype)
        next_mask = torch.ones((1, 1, num_frames, 1, 1), device=device, dtype=dtype)
        prev_mask[:, :, 0] = 0.0
        next_mask[:, :, -1] = 0.0
        return prev_mask, next_mask

    def _pad_to_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, _, H, W = x.shape
        Wh, Ww = self.window_size
        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0))
        return x, pad_h, pad_w

    def _window_partition(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int, int]]:
        """
        Args:
            x: [B, C, F, H, W]

        Returns:
            windows: [B*F*num_windows, Wh*Ww, C]
            meta: shape metadata for reverse
        """
        B, C, Fz, H, W = x.shape
        x, pad_h, pad_w = self._pad_to_window(x)
        _, _, _, Hp, Wp = x.shape
        Wh, Ww = self.window_size

        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, F, Hp, Wp, C]
        x = x.view(B * Fz, Hp // Wh, Wh, Wp // Ww, Ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = x.view(-1, Wh * Ww, C)
        meta = (B, C, Fz, H, W, pad_h, pad_w)
        return windows, meta

    def _window_reverse(self, windows: torch.Tensor, meta: Tuple[int, int, int, int, int, int, int]) -> torch.Tensor:
        B, C, Fz, H, W, pad_h, pad_w = meta
        Wh, Ww = self.window_size
        Hp = H + pad_h
        Wp = W + pad_w

        x = windows.view(B * Fz, Hp // Wh, Wp // Ww, Wh, Ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Fz, Hp, Wp, C)
        x = x[:, :, :H, :W, :]
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def _attend_one_direction(
        self,
        z_cur: torch.Tensor,
        z_ref: torch.Tensor,
        e_obj: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(z_cur)
        k = self.k_proj(z_ref)
        v = self.v_proj(z_ref)

        q_gamma, q_beta = torch.chunk(self.q_mod(e_obj), chunks=2, dim=1)
        k_gamma, k_beta = torch.chunk(self.k_mod(e_obj), chunks=2, dim=1)

        q_win, meta = self._window_partition(q)
        k_win, _ = self._window_partition(k)
        v_win, _ = self._window_partition(v)
        q_gamma_win, _ = self._window_partition(q_gamma)
        q_beta_win, _ = self._window_partition(q_beta)
        k_gamma_win, _ = self._window_partition(k_gamma)
        k_beta_win, _ = self._window_partition(k_beta)

        q_win = (1.0 + q_gamma_win) * self.q_norm(q_win) + q_beta_win
        k_win = (1.0 + k_gamma_win) * self.k_norm(k_win) + k_beta_win

        num_windows, num_tokens, channels = q_win.shape

        q_win = q_win.view(num_windows, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_win = k_win.view(num_windows, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_win = v_win.view(num_windows, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q_win, k_win.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v_win)
        out = out.permute(0, 2, 1, 3).contiguous().view(num_windows, num_tokens, channels)
        out = self._window_reverse(out, meta)
        out = self.out_proj(out)
        return out

    def forward(
        self,
        z: torch.Tensor,
        e_obj: torch.Tensor,
    ) -> torch.Tensor:
        B, C, Fz, H, W = z.shape

        z_prev = self._shift_prev(z)
        z_next = self._shift_next(z)

        prev_context = self._attend_one_direction(z, z_prev, e_obj)
        next_context = self._attend_one_direction(z, z_next, e_obj)

        prev_mask, next_mask = self._neighbor_masks(Fz, z.device, z.dtype)
        denom = (prev_mask + next_mask).clamp_min(1.0)
        z_attn = (prev_mask * prev_context + next_mask * next_context) / denom

        # If F == 1, no valid temporal neighbor exists. Keep identity.
        if Fz == 1:
            z_attn = z

        gate = torch.sigmoid(self.gate(e_obj))
        z_prop = z + gate * z_attn
        return z_prop


class ObjectMotionGuidedLatentPropagation(nn.Module):
    """
    Object-motion-guided latent enhancement for strengthening temporal information in z_sr.

    This version does not predict explicit offsets and does not warp latent features. Instead,
    it uses object-motion-conditioned latent attention to propagate temporal information from
    adjacent latent frames.

    Input:
        z_sr:  [B, C_z, F_z, H_z, W_z]
        m_obj: [B, C_m, F_m, H_m, W_m]

    Output:
        z_hat_sr: [B, C_z, F_z, H_z, W_z]

    Supports:
        - different C_z and C_m
        - different F/H/W between z_sr and m_obj
    """

    def __init__(
        self,
        latent_channels: int,
        motion_channels: int,
        hidden_channels: int = None,
        temporal_kernel: int = 5,
        num_heads: int = 4,
        window_size: Union[int, Tuple[int, int]] = 8,
        num_refine_blocks: int = 2,
        align_corners: bool = False,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.motion_channels = motion_channels
        self.hidden_channels = hidden_channels or latent_channels
        self.temporal_kernel = temporal_kernel
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_refine_blocks = num_refine_blocks
        self.align_corners = align_corners

        # 1) Temporal compression adapter: C_m -> C_z and F_m -> F_z
        self.motion_adapter = MotionTemporalAdapter(
            motion_channels=motion_channels,
            latent_channels=latent_channels,
            hidden_channels=self.hidden_channels,
            temporal_kernel=temporal_kernel,
            align_corners=align_corners,
        )

        # 2) Motion-conditioned latent attention, replacing offset prediction + warping
        self.motion_attention = MotionConditionedLatentAttention(
            channels=latent_channels,
            hidden_channels=self.hidden_channels,
            num_heads=num_heads,
            window_size=window_size,
        )

        # 3) Local convolutional refinement
        refine_blocks = []
        for _ in range(num_refine_blocks):
            refine_blocks.append(
                LocalRefineBlock(
                    in_channels=latent_channels * 2,
                    out_channels=latent_channels,
                    hidden_channels=self.hidden_channels,
                )
            )
        self.refine_blocks = nn.ModuleList(refine_blocks)

    def _ensure_bcfhw(self, x: torch.Tensor, tensor_name: str) -> torch.Tensor:
        """
        Ensure tensor is [B, C, F, H, W].
        """
        if x.dim() != 5:
            raise ValueError(
                f"{tensor_name} must be a 5D tensor with shape [B, C, F, H, W], "
                f"but got {tuple(x.shape)}"
            )
        return x

    def _check_input_shapes(self, z: torch.Tensor, m: torch.Tensor) -> None:
        Bz, Cz, _, _, _ = z.shape
        Bm, Cm, _, _, _ = m.shape

        if Bz != Bm:
            raise ValueError(f"Batch size mismatch: z_sr {Bz} vs m_obj {Bm}")
        if Cz != self.latent_channels:
            raise ValueError(f"z_sr channel mismatch: expected {self.latent_channels}, got {Cz}")
        if Cm != self.motion_channels:
            raise ValueError(f"m_obj channel mismatch: expected {self.motion_channels}, got {Cm}")

    def forward(
        self,
        z_sr: torch.Tensor,
        m_obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_sr:  [B, C_z, F_z, H_z, W_z]
            m_obj: [B, C_m, F_m, H_m, W_m]
        """
        z = self._ensure_bcfhw(z_sr, "z_sr")
        m = self._ensure_bcfhw(m_obj, "m_obj")
        self._check_input_shapes(z, m)

        # 1) Adapt object-motion feature to latent scale
        e_obj = self.motion_adapter(m, target_shape=z.shape)  # [B, C_z, F_z, H_z, W_z]

        # 2) Motion-conditioned latent attention + motion-gated fusion
        z_prop = self.motion_attention(z, e_obj)

        # 3) Local convolutional refinement
        z_hat = z_prop
        for block in self.refine_blocks:
            z_hat = block(torch.cat([z_hat, e_obj], dim=1), z_hat)
        return z_hat

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_object_motion_guided_latent_propagation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ObjectMotionGuidedLatentPropagation(
        latent_channels=16,
        motion_channels=128,
        hidden_channels=32,
        temporal_kernel=5,
        num_heads=4,
        window_size=8,
        num_refine_blocks=2,
        align_corners=False,
    ).to(device)

    # z_sr: [B, C, F, H, W]
    z_sr = torch.randn(2, 16, 7, 40, 80, device=device)

    # m_obj: [B, C, F, H, W]
    m_obj = torch.randn(2, 128, 7, 40, 80, device=device)

    model.eval()
    with torch.no_grad():
        z_hat = model(z_sr, m_obj)

    print("Input z_sr shape               :", tuple(z_sr.shape))
    print("Input m_obj shape              :", tuple(m_obj.shape))
    print("Output z_hat shape             :", tuple(z_hat.shape))
    print("Params:", count_params(model))

    assert z_hat.shape == z_sr.shape, "Output shape must match z_sr shape."
    print("Test passed.")


if __name__ == "__main__":
    test_object_motion_guided_latent_propagation()
