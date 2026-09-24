import torch
import torch.nn as nn
from typing import Dict, Tuple


class MotionDecouplingLoss(nn.Module):
    """
    Loss for motion decoupling:
        1) shared latent temporal residual consistency between HR-clean and LR-degraded
        2) degradation residual on HR-clean should be close to zero

    Expected input shape:
        [B, C, F, H, W]
    """

    def __init__(
        self,
        lambda_obj: float = 1.0,
        lambda_zero: float = 0.1,
        lambda_ratio: float = 1.0,
        ratio_eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__()

        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        self.lambda_obj = lambda_obj
        self.lambda_zero = lambda_zero
        self.lambda_ratio = lambda_ratio
        self.ratio_eps = ratio_eps
        self.reduction = reduction

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean()
        return x.sum()

    def _check_5d(self, x: torch.Tensor, name: str) -> None:
        if x.dim() != 5:
            raise ValueError(f"{name} must be 5D [B, C, F, H, W], but got shape {tuple(x.shape)}")

    def _check_same_shape(self, x: torch.Tensor, y: torch.Tensor, name_x: str, name_y: str) -> None:
        if x.shape != y.shape:
            raise ValueError(
                f"{name_x} and {name_y} must have the same shape, "
                f"but got {tuple(x.shape)} vs {tuple(y.shape)}"
            )

    def l1_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._reduce(torch.abs(x - y))

    def zero_loss(self, x: torch.Tensor) -> torch.Tensor:
        return self._reduce(torch.abs(x))

    def object_consistency_loss(
        self,
        m_obj_hr: torch.Tensor,
        m_obj_lr: torch.Tensor,
    ) -> torch.Tensor:
        self._check_5d(m_obj_hr, "m_obj_hr")
        self._check_5d(m_obj_lr, "m_obj_lr")
        self._check_same_shape(m_obj_hr, m_obj_lr, "m_obj_hr", "m_obj_lr")
        return self.l1_loss(m_obj_hr, m_obj_lr)

    def turbulence_zero_loss(self, m_turb_hr: torch.Tensor) -> torch.Tensor:
        self._check_5d(m_turb_hr, "m_turb_hr")
        return self.zero_loss(m_turb_hr)

    def _align_ratio_target(
        self,
        q_t_l: torch.Tensor,
        target_fhw: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if q_t_l.dim() != 5:
            raise ValueError(
                f"q_t_l must be 5D [B,1,F,H,W] or [B,F,1,H,W], got {tuple(q_t_l.shape)}"
            )

        if q_t_l.shape[1] != 1 and q_t_l.shape[2] == 1:
            q_t_l = q_t_l.permute(0, 2, 1, 3, 4).contiguous()
        if q_t_l.shape[1] != 1:
            raise ValueError(f"q_t_l channel dimension must be 1, got {tuple(q_t_l.shape)}")

        q_t_l = q_t_l.to(device=device, dtype=dtype).detach().clamp(0.0, 1.0)
        if q_t_l.shape[2:] != target_fhw:
            q_t_l = torch.nn.functional.interpolate(
                q_t_l,
                size=target_fhw,
                mode="trilinear",
                align_corners=False,
            )
        return q_t_l.clamp(0.0, 1.0)

    def ratio_prior_loss(
        self,
        m_obj_lr: torch.Tensor,
        m_turb_lr: torch.Tensor,
        q_t_l: torch.Tensor,
    ) -> torch.Tensor:
        self._check_5d(m_obj_lr, "m_obj_lr")
        self._check_5d(m_turb_lr, "m_turb_lr")
        self._check_same_shape(m_obj_lr, m_turb_lr, "m_obj_lr", "m_turb_lr")

        e_mot = m_obj_lr.abs().mean(dim=1, keepdim=True)
        e_turb = m_turb_lr.abs().mean(dim=1, keepdim=True)
        ratio = e_turb / (e_mot + e_turb + self.ratio_eps)

        q_t_l = self._align_ratio_target(
            q_t_l,
            target_fhw=ratio.shape[2:],
            device=ratio.device,
            dtype=ratio.dtype,
        )
        return self._reduce(torch.abs(ratio - q_t_l))

    def forward(
        self,
        m_obj_hr: torch.Tensor,
        m_turb_hr: torch.Tensor,
        m_obj_lr: torch.Tensor,
        m_turb_lr: torch.Tensor = None,
        q_t_l: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            m_obj_hr:  object-motion feature from HR-clean, [B,C,F,H,W]
            m_turb_hr: turbulence-motion feature from HR-clean, [B,C,F,H,W]
            m_obj_lr:  object-motion feature from LR-turbulent, [B,C,F,H,W]
            m_turb_lr: turbulence-motion feature from LR-turbulent, [B,C,F,H,W]
            q_t_l: STOCP uncertainty target for LR-turbulent, [B,1,F,H,W]

        Returns:
            total_loss, loss_dict
        """
        loss_obj = self.object_consistency_loss(m_obj_hr, m_obj_lr)
        loss_zero = self.turbulence_zero_loss(m_turb_hr)
        loss_ratio = torch.zeros((), device=loss_obj.device, dtype=loss_obj.dtype)

        if q_t_l is not None and m_turb_lr is not None and self.lambda_ratio != 0.0:
            loss_ratio = self.ratio_prior_loss(m_obj_lr, m_turb_lr, q_t_l)

        total_loss = (
            self.lambda_obj * loss_obj
            + self.lambda_zero * loss_zero
            + self.lambda_ratio * loss_ratio
        )

        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_obj": loss_obj.detach(),
            "loss_zero": loss_zero.detach(),
            "loss_ratio": loss_ratio.detach(),
        }

        return total_loss, loss_dict


def test_motion_decoupling_loss() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = MotionDecouplingLoss(
        lambda_obj=1.0,
        lambda_zero=0.1,
        lambda_ratio=1.0,
        reduction="mean",
    ).to(device)

    # [B, C, F, H, W]
    m_obj_hr = torch.randn(2, 16, 25, 40, 80, device=device)
    m_turb_hr = torch.randn(2, 16, 25, 40, 80, device=device)
    m_obj_lr = torch.randn(2, 16, 25, 40, 80, device=device)
    m_turb_lr = torch.randn(2, 16, 25, 40, 80, device=device)

    q_t_l = torch.rand(2, 1, 25, 40, 80, device=device)
    total_loss, loss_dict = criterion(m_obj_hr, m_turb_hr, m_obj_lr, m_turb_lr, q_t_l)

    assert total_loss.dim() == 0, "total_loss should be a scalar tensor"
    assert torch.isfinite(total_loss), "total_loss should be finite"
    expected_keys = {"loss_total", "loss_obj", "loss_zero", "loss_ratio"}
    assert expected_keys.issubset(loss_dict.keys()), "loss_dict misses expected keys"

    # Shape mismatch should raise
    bad_m_obj_lr = torch.randn(2, 16, 6, 24, 40, device=device)
    got_error = False
    try:
        _ = criterion(m_obj_hr, m_turb_hr, bad_m_obj_lr)
    except ValueError:
        got_error = True
    assert got_error, "Shape mismatch must raise ValueError"

    print("MotionDecouplingLoss test passed.")


if __name__ == "__main__":
    test_motion_decoupling_loss()
