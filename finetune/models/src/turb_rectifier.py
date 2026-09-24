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


class TurbulenceMotionAdapter(nn.Module):
    """
    Adapt turbulence-motion features to latent channel/spatial layout without temporal downsampling.

    Input:
        m_turb_lr: [B, C_m, F_m, H_m, W_m]

    Output:
        e_turb: [B, C_z, F_z, H_z, W_z]
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

    def forward(self, m_turb_lr: torch.Tensor, target_shape) -> torch.Tensor:
        """
        Args:
            m_turb_lr: [B, C_m, F_m, H_m, W_m]
            target_shape: shape of z_lr, i.e., [B, C_z, F_z, H_z, W_z]
        """
        _, _, Fz, Hz, Wz = target_shape

        # spatial alignment before channel projection
        if m_turb_lr.shape[-2:] != (Hz, Wz):
            m_turb_lr = F.interpolate(
                m_turb_lr,
                size=(m_turb_lr.shape[2], Hz, Wz),
                mode="trilinear",
                align_corners=self.align_corners,
            )

        e_turb = self.proj(m_turb_lr)

        # force exact F/H/W alignment with z_lr
        if e_turb.shape[2:] != (Fz, Hz, Wz):
            e_turb = F.interpolate(
                e_turb,
                size=(Fz, Hz, Wz),
                mode="trilinear",
                align_corners=self.align_corners,
            )

        return e_turb


class TurbulenceAwareLatentRectifier(nn.Module):
    """
    Turbulence-aware latent rectifier for preliminary turbulence suppression on z_lr.

    Input:
        z_lr:       [B, C_z, F_z, H_z, W_z]
        m_turb_lr:  [B, C_m, F_m, H_m, W_m]

    Output:
        z_tilde_lr: [B, C_z, F_z, H_z, W_z]

    Supports:
        - different C_z and C_m
        - different F/H/W between z_lr and m_turb_lr
    """

    def __init__(
        self,
        z_channels: int,
        motion_channels: int,
        hidden_channels: int = None,
        proj_channels: int = None,
        temporal_kernel: int = 5,
        align_corners: bool = False,
    ):
        super().__init__()

        self.z_channels = z_channels
        self.motion_channels = motion_channels
        self.hidden_channels = hidden_channels or z_channels
        self.proj_channels = proj_channels or z_channels
        self.temporal_kernel = temporal_kernel
        self.align_corners = align_corners

        # 1) Turbulence-motion adapter: C_m -> C_z and F_m -> F_z
        self.motion_adapter = TurbulenceMotionAdapter(
            motion_channels=motion_channels,
            latent_channels=z_channels,
            hidden_channels=self.proj_channels,
            temporal_kernel=temporal_kernel,
            align_corners=align_corners,
        )

        # 2) Residual prediction F_r([z_lr, e_turb])
        self.residual_head = nn.Sequential(
            nn.Conv3d(z_channels * 2, self.hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            DepthwiseSeparableConv3d(self.hidden_channels),
            nn.Conv3d(self.hidden_channels, z_channels, kernel_size=1, bias=True),
        )

        # 3) Gate prediction F_g
        self.gate_head = nn.Sequential(
            nn.Conv3d(z_channels, self.hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(self.hidden_channels, z_channels, kernel_size=1, bias=True),
        )

        # 4) Modulation F_m -> gamma, beta
        self.modulation_head = nn.Sequential(
            nn.Conv3d(z_channels, self.hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(self.hidden_channels, z_channels * 2, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # close to identity at init
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -2.0)

        nn.init.zeros_(self.modulation_head[-1].weight)
        nn.init.zeros_(self.modulation_head[-1].bias)

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
            raise ValueError(f"Batch size mismatch: z_lr {Bz} vs m_turb_lr {Bm}")
        if Cz != self.z_channels:
            raise ValueError(f"z_lr channel mismatch: expected {self.z_channels}, got {Cz}")
        if Cm != self.motion_channels:
            raise ValueError(f"m_turb_lr channel mismatch: expected {self.motion_channels}, got {Cm}")

    def forward(self, z_lr: torch.Tensor, m_turb_lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_lr:      [B, C_z, F_z, H_z, W_z]
            m_turb_lr: [B, C_m, F_m, H_m, W_m]
        """
        z = self._ensure_bcfhw(z_lr, "z_lr")
        m = self._ensure_bcfhw(m_turb_lr, "m_turb_lr")
        self._check_input_shapes(z, m)

        # 1) Adapt turbulence-motion feature to latent scale
        e_turb = self.motion_adapter(m, target_shape=z.shape)  # [B, C_z, F_z, H_z, W_z]

        # 2) Local residual prediction
        r = self.residual_head(torch.cat([z, e_turb], dim=1))

        # 3) Gated rectification
        g = torch.sigmoid(self.gate_head(e_turb))
        z_rect = z + g * r

        # 4) Scale-shift modulation
        gamma_beta = self.modulation_head(e_turb)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        z_tilde = (1.0 + gamma) * z_rect + beta

        return z_tilde

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_turbulence_aware_latent_rectifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TurbulenceAwareLatentRectifier(
        z_channels=16,
        motion_channels=128,
        hidden_channels=32,
        proj_channels=32,
        temporal_kernel=5,
        align_corners=False,
    ).to(device)

    # z_lr: [B, C, F, H, W]
    z_lr = torch.randn(2, 16, 7, 40, 80, device=device)

    # m_turb_lr: [B, C, F, H, W]
    m_turb_lr = torch.randn(2, 128, 7, 40, 80, device=device)

    model.eval()
    with torch.no_grad():
        z_tilde = model(z_lr, m_turb_lr)
        e_turb = model.motion_adapter(m_turb_lr, target_shape=z_lr.shape)

    print("Input z_lr shape                :", tuple(z_lr.shape))
    print("Input m_turb_lr shape           :", tuple(m_turb_lr.shape))
    print("Adapted e_turb shape            :", tuple(e_turb.shape))
    print("Output z_tilde shape            :", tuple(z_tilde.shape))
    print("Params:", count_params(model))

    assert z_tilde.shape == z_lr.shape, "Output shape must match z_lr shape."
    assert e_turb.shape == z_lr.shape, "Adapted turbulence feature shape must match z_lr shape."
    print("Test passed.")


if __name__ == "__main__":
    test_turbulence_aware_latent_rectifier()
