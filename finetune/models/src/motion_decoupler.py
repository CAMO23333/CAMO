import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionDecoupler(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # gate net
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 4, channels, 1),
            nn.GELU(),
            nn.Conv3d(channels, channels, 1),
            nn.Sigmoid()
        )

        # smooth motion estimator
        self.smooth_net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 1),
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 1),
        )

    def temporal_diff(self, x):
        """
        x: [B, C, T, H, W]
        """
        dt = x[:, :, 1:] - x[:, :, :-1]
        dt = F.pad(dt, (0, 0, 0, 0, 0, 1))
        return dt

    def temporal_diff2(self, x):
        """
        x: [B, C, T, H, W]
        """
        dt = self.temporal_diff(x)
        d2t = dt[:, :, 1:] - dt[:, :, :-1]
        d2t = F.pad(d2t, (0, 0, 0, 0, 0, 1))
        return d2t

    def temporal_haar_dwt(self, x):
        """
        Temporal Haar DWT along time dimension.

        Input:
            x: [B, C, T, H, W]

        Returns:
            low: [B, C, ceil(T/2), H, W]
            high: [B, C, ceil(T/2), H, W]
            orig_T: original time length T
        """
        B, C, T, H, W = x.shape
        orig_T = T

        # If T is odd, pad one frame at the end by replication
        if T % 2 == 1:
            x = F.pad(x, (0, 0, 0, 0, 0, 1), mode="replicate")
            T = T + 1

        x_even = x[:, :, 0:T:2, :, :]
        x_odd = x[:, :, 1:T:2, :, :]

        # Haar DWT
        low = (x_even + x_odd) / math.sqrt(2.0)
        high = (x_even - x_odd) / math.sqrt(2.0)

        return low, high, orig_T

    def temporal_haar_idwt_components(self, low, high, orig_T):
        """
        Reconstruct low-frequency-only signal and high-frequency-only signal
        back to the original time length.

        Inputs:
            low:  [B, C, T_half, H, W]
            high: [B, C, T_half, H, W]
            orig_T: original temporal length before padding

        Returns:
            x_low_rec:  [B, C, orig_T, H, W]
            x_high_rec: [B, C, orig_T, H, W]
        """
        B, C, T_half, H, W = low.shape
        T_pad = T_half * 2

        device = low.device
        dtype = low.dtype

        x_low_rec = torch.zeros((B, C, T_pad, H, W), device=device, dtype=dtype)
        x_high_rec = torch.zeros((B, C, T_pad, H, W), device=device, dtype=dtype)

        # inverse Haar using only low component
        x_low_rec[:, :, 0:T_pad:2, :, :] = low / math.sqrt(2.0)
        x_low_rec[:, :, 1:T_pad:2, :, :] = low / math.sqrt(2.0)

        # inverse Haar using only high component
        x_high_rec[:, :, 0:T_pad:2, :, :] = high / math.sqrt(2.0)
        x_high_rec[:, :, 1:T_pad:2, :, :] = -high / math.sqrt(2.0)

        # crop back to original T if padded
        x_low_rec = x_low_rec[:, :, :orig_T, :, :]
        x_high_rec = x_high_rec[:, :, :orig_T, :, :]

        return x_low_rec, x_high_rec

    def temporal_dwt_decompose(self, x):
        """
        DWT-based temporal decomposition.

        Input:
            x: [B, C, T, H, W]

        Returns:
            M_lp: [B, C, T, H, W]  reconstructed low-frequency component
            M_hp: [B, C, T, H, W]  reconstructed high-frequency component
        """
        low, high, orig_T = self.temporal_haar_dwt(x)
        M_lp, M_hp = self.temporal_haar_idwt_components(low, high, orig_T)
        return M_lp, M_hp

    def forward(self, M):
        """
        M: [B, C, F, H, W]
        return:
            M_obj:  [B, C, F, H, W]
            M_turb: [B, C, F, H, W]
        """
        # DWT-based temporal low/high decomposition
        M_lp, M_hp = self.temporal_dwt_decompose(M)

        dt = self.temporal_diff(M)
        d2t = self.temporal_diff2(M)

        G = self.gate(torch.cat([M_lp, M_hp, dt.abs(), d2t.abs()], dim=1))
        M_base = M_lp + G * M_hp
        M_obj = self.smooth_net(M_base) + M_base
        M_turb = M - M_obj

        return M_obj, M_turb


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # -------------------------
    # Test code
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, num_frames, H, W = 2, 128, 7, 40, 80   # deliberately use odd frame count to test padding branch
    # [B, C, F, H, W]
    x = torch.randn(B, C, num_frames, H, W).to(device)

    model = MotionDecoupler(channels=C).to(device)
    model.eval()

    with torch.no_grad():
        # [B, C, F, H, W]
        M_obj, M_turb = model(x)

    print("Input shape   :", x.shape)
    print("M_obj shape   :", M_obj.shape)
    print("M_turb shape  :", M_turb.shape)
    print("Params:", count_params(model))