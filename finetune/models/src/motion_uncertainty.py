import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union


class MotionUncertaintyEstimator(nn.Module):
    """
    Estimate uncertainty U_t from decoupled object-motion and turbulence-motion features.

    Inputs:
        m_obj:  [B, C, F, H, W]
        m_turb: [B, C, F, H, W]

    Output:
        U_t: [B, 1, F, H, W], range in [0, 1]

    The module uses two uncertainty cues:
        1) Object-motion temporal instability:
              R_m = |M_{t+1} - 2M_t + M_{t-1}|
        2) Object-turbulence feature conflict:
              R_int = ||M_t|| * ||T_t|| * (1 - cos(M_t, T_t))

    Notes:
        - Residual decomposition uncertainty is not used.
        - Temporal consistency uncertainty of turbulence motion is not used.
        - All tensors are expected to be in [B, C, F, H, W].
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 16,
        eps: float = 1e-6,
        normalize_cues: bool = True,
        init_uncertainty: float = 0.5,
    ):
        super().__init__()

        if not (0.0 < init_uncertainty < 1.0):
            raise ValueError("init_uncertainty must be in (0, 1).")

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.eps = eps
        self.normalize_cues = normalize_cues
        self.init_uncertainty = init_uncertainty

        self.uncertainty_head = nn.Sequential(
            nn.Conv3d(2, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        final = self.uncertainty_head[-1]
        nn.init.zeros_(final.weight)
        init_bias = math.log(self.init_uncertainty / (1.0 - self.init_uncertainty))
        nn.init.constant_(final.bias, init_bias)

    def _ensure_bcfhw(self, x: torch.Tensor, tensor_name: str) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"{tensor_name} must be a 5D tensor with shape [B, C, F, H, W], "
                f"but got {tuple(x.shape)}"
            )
        return x

    def _check_input_shapes(self, m_obj: torch.Tensor, m_turb: torch.Tensor) -> None:
        Bo, Co, Fo, Ho, Wo = m_obj.shape
        Bt, Ct, Ft, Ht, Wt = m_turb.shape

        if Bo != Bt:
            raise ValueError(f"Batch size mismatch: m_obj {Bo} vs m_turb {Bt}")
        if Co != self.channels:
            raise ValueError(f"m_obj channel mismatch: expected {self.channels}, got {Co}")
        if Ct != self.channels:
            raise ValueError(f"m_turb channel mismatch: expected {self.channels}, got {Ct}")
        if (Fo, Ho, Wo) != (Ft, Ht, Wt):
            raise ValueError(
                f"m_obj and m_turb must have the same F/H/W, "
                f"but got {(Fo, Ho, Wo)} vs {(Ft, Ht, Wt)}"
            )

    def _normalize_map(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_cues:
            return x

        dims = (2, 3, 4)
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps)

    def object_temporal_instability(self, m_obj: torch.Tensor) -> torch.Tensor:
        """
        Compute object-motion temporal instability.

        Input:
            m_obj: [B, C, F, H, W]

        Output:
            R_m: [B, 1, F, H, W]
        """
        self._ensure_bcfhw(m_obj, "m_obj")

        F_len = m_obj.shape[2]
        if F_len < 3:
            return torch.zeros(
                (m_obj.shape[0], 1, F_len, m_obj.shape[3], m_obj.shape[4]),
                device=m_obj.device,
                dtype=m_obj.dtype,
            )

        m_prev = torch.cat([m_obj[:, :, :1], m_obj[:, :, :-1]], dim=2)
        m_next = torch.cat([m_obj[:, :, 1:], m_obj[:, :, -1:]], dim=2)

        d2 = m_next - 2.0 * m_obj + m_prev
        r_m = d2.abs().mean(dim=1, keepdim=True)

        scale = m_obj.abs().mean(dim=1, keepdim=True).detach()
        r_m = r_m / (scale + self.eps)

        return r_m

    def motion_turbulence_conflict(self, m_obj: torch.Tensor, m_turb: torch.Tensor) -> torch.Tensor:
        """
        Compute feature-space conflict between object motion and turbulence motion.

        Inputs:
            m_obj:  [B, C, F, H, W]
            m_turb: [B, C, F, H, W]

        Output:
            R_int: [B, 1, F, H, W]
        """
        self._ensure_bcfhw(m_obj, "m_obj")
        self._ensure_bcfhw(m_turb, "m_turb")

        mag_obj = torch.sqrt(torch.mean(m_obj * m_obj, dim=1, keepdim=True) + self.eps)
        mag_turb = torch.sqrt(torch.mean(m_turb * m_turb, dim=1, keepdim=True) + self.eps)

        dot = torch.mean(m_obj * m_turb, dim=1, keepdim=True)
        cos = dot / (mag_obj * mag_turb + self.eps)
        cos = torch.clamp(cos, min=-1.0, max=1.0)

        r_int = mag_obj * mag_turb * (1.0 - cos)

        scale = (mag_obj * mag_turb).detach()
        r_int = r_int / (scale + self.eps)

        return r_int

    def forward(
        self,
        m_obj: torch.Tensor,
        m_turb: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            m_obj:  [B, C, F, H, W]
            m_turb: [B, C, F, H, W]

        Returns:
            U_t: [B, 1, F, H, W]
        """
        m_obj = self._ensure_bcfhw(m_obj, "m_obj")
        m_turb = self._ensure_bcfhw(m_turb, "m_turb")
        self._check_input_shapes(m_obj, m_turb)

        r_m = self.object_temporal_instability(m_obj)
        r_int = self.motion_turbulence_conflict(m_obj, m_turb)

        r_m_in = self._normalize_map(r_m)
        r_int_in = self._normalize_map(r_int)

        cue = torch.cat([r_m_in, r_int_in], dim=1)
        U_t = torch.sigmoid(self.uncertainty_head(cue))

        return U_t


def test_motion_uncertainty_estimator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, F_len, H, W = 2, 16, 7, 40, 80

    m_obj = torch.randn(B, C, F_len, H, W, device=device)
    m_turb = torch.randn(B, C, F_len, H, W, device=device)

    model = MotionUncertaintyEstimator(
        channels=C,
        hidden_channels=16,
        normalize_cues=True,
        init_uncertainty=0.5,
    ).to(device)

    model.eval()
    with torch.no_grad():
        U_t = model(m_obj, m_turb)

    print("Input m_obj shape              :", tuple(m_obj.shape))
    print("Input m_turb shape             :", tuple(m_turb.shape))
    print("Output U_t shape               :", tuple(U_t.shape))
    print("U_t range                      :", float(U_t.min()), float(U_t.max()))

    assert U_t.shape == (B, 1, F_len, H, W), "U_t shape mismatch."
    assert torch.all(U_t >= 0) and torch.all(U_t <= 1), "U_t must be in [0, 1]."
    print("Test passed.")


if __name__ == "__main__":
    test_motion_uncertainty_estimator()
