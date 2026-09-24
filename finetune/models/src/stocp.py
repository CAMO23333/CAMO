import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class STOCPPriorEstimator(nn.Module):
    """
    Network-facing spatio-temporal optical coherent prior estimator.

    Input:
        y_lr: low-quality video tensor in [B, T, C, H, W], range [0, 1].

    Output:
        Q: thermo-optical perturbation intensity in [B, T, 1, H, W].
        rho: coherent-motion reliability in [B, T, 1, H, W].

    Optional output:
        Q_latent and rho_latent aligned to the requested latent resolution.

    Video loading and visualization intentionally live outside this module.
    """

    def __init__(
        self,
        patch_size: int = 32,
        stride: int = 8,
        max_shift: int = 16,
        radiance_gamma: float = 0.6,
        eps: float = 1e-6,
        w_phi: float = 1.0,
        w_rad: float = 1.0,
        w_edge: float = 1.0,
        rho_lambda: float = 2.0,
        q_low: float = 0.02,
        q_high: float = 0.98,
        norm_mode: str = "fixed",
        phi_scale: float = 1.39,
        rad_scale: float = 0.185,
        edge_scale: float = 0.258,
    ):
        super().__init__()

        if patch_size < 2:
            raise ValueError("patch_size must be >= 2")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if max_shift < 0 or max_shift > patch_size // 2:
            raise ValueError("max_shift must be in [0, patch_size // 2]")
        if norm_mode not in {"fixed", "robust"}:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")
        if min(phi_scale, rad_scale, edge_scale) <= 0.0:
            raise ValueError("phi_scale, rad_scale and edge_scale must be positive")
        if min(w_phi, w_rad, w_edge) < 0.0:
            raise ValueError("STOCP residual weights must be non-negative")
        if w_phi + w_rad + w_edge <= 0.0:
            raise ValueError("At least one STOCP residual weight must be positive")

        self.patch_size = patch_size
        self.stride = stride
        self.max_shift = max_shift
        self.radiance_gamma = radiance_gamma
        self.eps = eps

        self.w_phi = w_phi
        self.w_rad = w_rad
        self.w_edge = w_edge
        self.rho_lambda = rho_lambda

        self.q_low = q_low
        self.q_high = q_high

        self.norm_mode = norm_mode
        self.phi_scale = phi_scale
        self.rad_scale = rad_scale
        self.edge_scale = edge_scale

        hann_1d = torch.hann_window(patch_size, periodic=False)
        hann_2d = torch.outer(hann_1d, hann_1d)
        self.register_buffer(
            "hann",
            hann_2d.view(1, 1, 1, patch_size, patch_size),
            persistent=False,
        )

        fy = torch.fft.fftfreq(patch_size).view(1, 1, patch_size, 1)
        fx = torch.fft.fftfreq(patch_size).view(1, 1, 1, patch_size)
        self.register_buffer("fy", fy, persistent=False)
        self.register_buffer("fx", fx, persistent=False)

        yy, xx = torch.meshgrid(
            torch.arange(patch_size),
            torch.arange(patch_size),
            indexing="ij",
        )
        sy = torch.where(yy > patch_size // 2, yy - patch_size, yy)
        sx = torch.where(xx > patch_size // 2, xx - patch_size, xx)
        shift_mask = (sy.abs() <= max_shift) & (sx.abs() <= max_shift)
        self.register_buffer(
            "shift_mask",
            shift_mask.view(1, 1, patch_size, patch_size),
            persistent=False,
        )

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(
        self,
        y_lr: torch.Tensor,
        latent_hw: Optional[Tuple[int, int]] = None,
        latent_temporal_downsample: int = 4,
    ) -> Dict[str, torch.Tensor]:
        return self._forward_impl(y_lr, latent_hw, latent_temporal_downsample)

    def _forward_impl(
        self,
        y_lr: torch.Tensor,
        latent_hw: Optional[Tuple[int, int]] = None,
        latent_temporal_downsample: int = 4,
    ) -> Dict[str, torch.Tensor]:
        if y_lr.dim() != 5:
            raise ValueError(f"y_lr must be [B,T,C,H,W], got {tuple(y_lr.shape)}")
        if not y_lr.is_floating_point():
            raise TypeError(f"y_lr must be floating point, got {y_lr.dtype}")
        if latent_temporal_downsample < 1:
            raise ValueError("latent_temporal_downsample must be >= 1")
        if latent_hw is not None and min(latent_hw) < 1:
            raise ValueError(f"latent_hw must be positive, got {latent_hw}")

        B, T, C, H, W = y_lr.shape
        if T < 1 or C < 1 or H < 1 or W < 1:
            raise ValueError(f"y_lr has an invalid shape: {tuple(y_lr.shape)}")

        if T == 1:
            Q = torch.zeros(B, T, 1, H, W, device=y_lr.device, dtype=y_lr.dtype)
            rho = torch.ones_like(Q)
            out = {"Q": Q, "rho": rho}
            if latent_hw is not None:
                Q_latent = self._resize_bthw(Q, latent_hw, latent_temporal_downsample)
                rho_latent = self._resize_bthw(rho, latent_hw, latent_temporal_downsample)
                out.update({"Q_latent": Q_latent, "rho_latent": rho_latent})
            return out

        theta = self._pseudo_radiance(y_lr)

        theta_pad, _ = self._pad_video(theta)
        Hp, Wp = theta_pad.shape[-2:]

        pair_Q_list = []

        for t in range(T - 1):
            q_pair = self._process_pair(
                theta_pad[:, t],
                theta_pad[:, t + 1],
            )
            pair_Q_list.append(q_pair)

        pair_Q = torch.stack(pair_Q_list, dim=1)

        Q_pad = torch.zeros(
            B, T, 1, Hp, Wp,
            device=y_lr.device,
            dtype=y_lr.dtype,
        )

        Q_pad[:, 0] = pair_Q[:, 0]
        Q_pad[:, -1] = pair_Q[:, -1]

        if T > 2:
            Q_pad[:, 1:-1] = 0.5 * (pair_Q[:, :-1] + pair_Q[:, 1:])

        Q = Q_pad[..., :H, :W].contiguous()

        rho = torch.exp(-self.rho_lambda * Q)
        rho = rho.clamp(0.0, 1.0)

        out = {
            "Q": Q,
            "rho": rho,
        }

        if latent_hw is not None:
            Q_latent = self._resize_bthw(Q, latent_hw, latent_temporal_downsample)
            rho_latent = self._resize_bthw(rho, latent_hw, latent_temporal_downsample)
            out.update({
                "Q_latent": Q_latent,
                "rho_latent": rho_latent,
            })

        return out

    def _pseudo_radiance(self, y: torch.Tensor) -> torch.Tensor:
        if y.size(2) == 1:
            gray = y
        else:
            gray = y.mean(dim=2, keepdim=True)

        gray = gray.clamp(min=0.0)

        theta = torch.log(gray.pow(self.radiance_gamma) + self.eps)

        min_v = theta.amin(dim=(-2, -1), keepdim=True)
        max_v = theta.amax(dim=(-2, -1), keepdim=True)
        theta = (theta - min_v) / (max_v - min_v + self.eps)

        return theta

    def _process_pair(
        self,
        theta_t: torch.Tensor,
        theta_tp1: torch.Tensor,
    ) -> torch.Tensor:

        _, _, H, W = theta_t.shape

        patches_t = self._extract_patches(theta_t)
        patches_tp1 = self._extract_patches(theta_tp1)

        pt = patches_t * self.hann
        pp = patches_tp1 * self.hann

        Ft = torch.fft.fft2(pt.squeeze(2), dim=(-2, -1))
        Fp = torch.fft.fft2(pp.squeeze(2), dim=(-2, -1))

        dy, dx = self._estimate_shift(Ft, Fp)

        E_phi = self._phase_residual(Ft, Fp, dy, dx)

        patches_tp1_aligned = self._warp_patches(patches_tp1, dy, dx)
        E_rad = (patches_t - patches_tp1_aligned).abs().mean(dim=(2, 3, 4))

        edge_t = self._sobel_mag(patches_t)
        edge_tp1 = self._sobel_mag(patches_tp1_aligned)
        E_edge = (edge_t - edge_tp1).abs().mean(dim=(2, 3, 4))

        E_phi_n, E_rad_n, E_edge_n = self._normalize_residuals(
            E_phi,
            E_rad,
            E_edge,
        )

        weight_sum = self.w_phi + self.w_rad + self.w_edge + self.eps

        Q_patch = (
            self.w_phi * E_phi_n +
            self.w_rad * E_rad_n +
            self.w_edge * E_edge_n
        ) / weight_sum

        Q_patch = Q_patch.clamp(0.0, 1.0)

        Q_map = self._patch_scalar_to_map(Q_patch, H, W)

        return Q_map

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        s = self.stride

        patches = F.unfold(x, kernel_size=p, stride=s)
        patches = patches.view(B, C, p, p, -1)
        patches = patches.permute(0, 4, 1, 2, 3).contiguous()

        return patches

    def _estimate_shift(
        self,
        Ft: torch.Tensor,
        Fp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        p = self.patch_size

        cross = Fp * torch.conj(Ft)
        cross = cross / (cross.abs() + self.eps)

        corr = torch.fft.ifft2(cross, dim=(-2, -1)).real
        corr = corr.masked_fill(~self.shift_mask, float("-inf"))

        idx = corr.flatten(-2).argmax(dim=-1)

        iy = idx // p
        ix = idx % p

        real_dtype = Ft.real.dtype

        dy = torch.where(iy > p // 2, iy - p, iy).to(device=Ft.device, dtype=real_dtype)
        dx = torch.where(ix > p // 2, ix - p, ix).to(device=Ft.device, dtype=real_dtype)

        return dy, dx

    def _phase_residual(
        self,
        Ft: torch.Tensor,
        Fp: torch.Tensor,
        dy: torch.Tensor,
        dx: torch.Tensor,
    ) -> torch.Tensor:

        phase_diff = torch.angle(Fp) - torch.angle(Ft)

        dy = dy.to(device=Ft.device, dtype=phase_diff.dtype)
        dx = dx.to(device=Ft.device, dtype=phase_diff.dtype)

        expected = 2.0 * math.pi * (
            self.fy.to(dtype=phase_diff.dtype) * dy[..., None, None] +
            self.fx.to(dtype=phase_diff.dtype) * dx[..., None, None]
        )

        residual = self._wrap_phase(phase_diff + expected)

        amp = torch.sqrt(Ft.abs() * Fp.abs() + self.eps)
        amp = amp.clone()
        amp[..., 0, 0] = 0.0

        E_phi = (amp * residual.abs()).sum(dim=(-2, -1)) / (
            amp.sum(dim=(-2, -1)) + self.eps
        )

        return E_phi

    def _warp_patches(
        self,
        patches: torch.Tensor,
        dy: torch.Tensor,
        dx: torch.Tensor,
    ) -> torch.Tensor:

        B, N, C, p, _ = patches.shape
        L = B * N

        patches_flat = patches.view(L, C, p, p)
        dy_flat = dy.reshape(L)
        dx_flat = dx.reshape(L)

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, p, device=patches.device, dtype=patches.dtype),
            torch.linspace(-1, 1, p, device=patches.device, dtype=patches.dtype),
            indexing="ij",
        )

        base_grid = torch.stack([xx, yy], dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(L, 1, 1, 1)

        offset_x = 2.0 * dx_flat.view(L, 1, 1) / max(p - 1, 1)
        offset_y = 2.0 * dy_flat.view(L, 1, 1) / max(p - 1, 1)

        grid = base_grid.clone()
        grid[..., 0] = grid[..., 0] + offset_x
        grid[..., 1] = grid[..., 1] + offset_y

        warped = F.grid_sample(
            patches_flat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        warped = warped.view(B, N, C, p, p)

        return warped

    def _sobel_mag(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, C, p, _ = patches.shape
        x = patches.view(B * N, C, p, p)

        gx = F.conv2d(x, self.sobel_x.to(dtype=x.dtype), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(dtype=x.dtype), padding=1)

        mag = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)

        return mag.view(B, N, C, p, p)

    def _patch_scalar_to_map(
        self,
        score: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:

        B, N = score.shape
        p = self.patch_size
        s = self.stride

        cols = score.unsqueeze(1).expand(B, p * p, N).contiguous()

        out = F.fold(
            cols,
            output_size=(H, W),
            kernel_size=p,
            stride=s,
        )

        norm_cols = torch.ones_like(cols)
        norm = F.fold(
            norm_cols,
            output_size=(H, W),
            kernel_size=p,
            stride=s,
        )

        out = out / (norm + self.eps)

        return out

    def _robust_norm(self, x: torch.Tensor) -> torch.Tensor:
        low = torch.quantile(x, self.q_low, dim=1, keepdim=True)
        high = torch.quantile(x, self.q_high, dim=1, keepdim=True)

        x = (x - low) / (high - low + self.eps)
        x = x.clamp(0.0, 1.0)

        return x

    def _fixed_norm(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        return (x / (scale + self.eps)).clamp(0.0, 1.0)

    def _normalize_residuals(
        self,
        E_phi: torch.Tensor,
        E_rad: torch.Tensor,
        E_edge: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.norm_mode == "fixed":
            E_phi_n = self._fixed_norm(E_phi, self.phi_scale)
            E_rad_n = self._fixed_norm(E_rad, self.rad_scale)
            E_edge_n = self._fixed_norm(E_edge, self.edge_scale)
        elif self.norm_mode == "robust":
            E_phi_n = self._robust_norm(E_phi)
            E_rad_n = self._robust_norm(E_rad)
            E_edge_n = self._robust_norm(E_edge)
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

        return E_phi_n, E_rad_n, E_edge_n

    @staticmethod
    def _wrap_phase(x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))

    def _pad_video(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, T, C, H, W = x.shape
        p = self.patch_size
        s = self.stride

        def target_len(L: int) -> int:
            if L <= p:
                return p
            n = math.ceil((L - p) / s) + 1
            return (n - 1) * s + p

        Ht = target_len(H)
        Wt = target_len(W)

        pad_h = Ht - H
        pad_w = Wt - W

        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)

        x4 = x.view(B * T, C, H, W)

        mode = "reflect" if H > 1 and W > 1 else "replicate"

        x4 = F.pad(
            x4,
            pad=(0, pad_w, 0, pad_h),
            mode=mode,
        )

        x = x4.view(B, T, C, Ht, Wt)

        return x, (pad_h, pad_w)

    @staticmethod
    def _latent_temporal_len(num_frames: int, temporal_downsample: int = 4) -> int:
        if temporal_downsample <= 1 or num_frames <= 1:
            return num_frames

        return math.ceil((num_frames - 1) / temporal_downsample) + 1

    @classmethod
    def _resize_bthw(
        cls,
        x: torch.Tensor,
        size_hw: Tuple[int, int],
        temporal_downsample: int = 4,
    ) -> torch.Tensor:

        B, T, C, H, W = x.shape
        h, w = size_hw
        latent_t = cls._latent_temporal_len(T, temporal_downsample)

        x5 = x.permute(0, 2, 1, 3, 4).contiguous()
        x5 = F.interpolate(
            x5,
            size=(latent_t, h, w),
            mode="trilinear",
            align_corners=False,
        )

        return x5.permute(0, 2, 1, 3, 4).contiguous()
