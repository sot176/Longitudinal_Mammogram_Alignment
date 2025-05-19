import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousPosEncoding(nn.Module):
    def __init__(self, dim, drop=0.1, maxtime=5, num_steps=100):
        """
        Continuous sinusoidal positional encoding with linear interpolation over time.

        Args:
            dim (int): Dimension of the encoding.
            drop (float): Dropout rate.
            maxtime (float): Maximum time value for normalization.
            num_steps (int): Number of discrete time steps for encoding table.
        """
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.maxtime = maxtime
        self.num_steps = num_steps

        # Precompute sinusoidal encodings
        position = torch.linspace(0, maxtime, steps=num_steps).unsqueeze(1)  # (S, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe = torch.zeros(num_steps, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, xs, times):
        """
        Args:
            xs (Tensor): Input tensor of shape (N, B, C).
            times (Tensor): Time values of shape (B,).

        Returns:
            Tensor: Time-encoded input of shape (N, B, C).
        """
        times = torch.clamp(times, 0, self.maxtime) * (self.num_steps - 1) / self.maxtime
        t_floor = torch.floor(times).long()
        t_ceil = torch.ceil(times).long()
        alpha = (times - t_floor).unsqueeze(1)  # (B, 1)

        # Linear interpolation
        pe_floor = self.pe[t_floor]  # (B, C)
        pe_ceil = self.pe[t_ceil]    # (B, C)
        pe_interp = (1 - alpha) * pe_floor + alpha * pe_ceil  # (B, C)

        return self.dropout(xs + pe_interp.unsqueeze(0))  # (N, B, C)

class CumulativeProbabilityLayer(nn.Module):
    def __init__(self, num_features, max_followup):
        """
        Predict cumulative cancer probabilities via time-dependent hazard estimation.

        Args:
            num_features (int): Feature size from the model.
            max_followup (int): Number of follow-up years (prediction steps).
        """
        super().__init__()
        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)

        # Lower-triangular mask (T x T)
        mask = torch.tril(torch.ones(max_followup, max_followup)).T
        self.register_buffer("upper_triangular_mask", mask)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input features of shape (B, C)

        Returns:
            Tensor: Cumulative probability over time (B, T)
        """
        B = x.size(0)
        raw_hazards = self.relu(self.hazard_fc(x))  # (B, T)
        base_hazard = self.base_hazard_fc(x)        # (B, 1)

        expanded = raw_hazards.unsqueeze(-1).expand(B, -1, raw_hazards.size(1))  # (B, T, T)
        masked = expanded * self.upper_triangular_mask  # (B, T, T)

        cum_probs = masked.sum(dim=1) + base_hazard  # (B, T)
        return cum_probs


class TemporalRiskPredictionWithCumulativeProbLayer(nn.Module):
    def __init__(self, num_years=5, time_encoding_dim=512):
        """
        Temporal risk prediction with cumulative probability estimation and feature alignment.
        Args:
            num_years (int): Number of follow-up years.
            time_encoding_dim (int): Dimensionality for positional time encoding.
        """
        super().__init__()
        self.positional_encoding = ContinuousPosEncoding(dim=time_encoding_dim)

        # Risk layers for different feature branches
        self.cumulative_prob_layer_fused = CumulativeProbabilityLayer(num_features=3 * 512, max_followup=num_years + 1)
        self.cumulative_prob_layer_cur = CumulativeProbabilityLayer(num_features=512, max_followup=num_years + 1)
        self.cumulative_prob_layer_pri = CumulativeProbabilityLayer(num_features=512, max_followup=num_years + 1)

    def forward(self, f_cur, f_pri, f_pri_aligned, f_dif, time_gap):
        """
        Args:
            f_cur (Tensor): Current features (B, C, H, W)
            f_pri (Tensor): Prior features (B, C, H, W)
            f_pri_aligned (Tensor): Aligned prior features (B, C, H, W)
            f_dif (Tensor): Differential features (B, C, H, W)
            time_gap (Tensor): Time gap (B,)
        Returns:
            dict: Dictionary of risk predictions (fused, cur, pri)
        """
        B, C, H, W = f_dif.shape

        # Encode time into differential features
        fdif_flat = f_dif.flatten(2).permute(2, 0, 1)             # (H*W, B, C)
        fdif_time_encoded = self.positional_encoding(fdif_flat, time_gap) \
                                        .permute(1, 2, 0).view(B, C, H, W)

        # Global average pooling
        f_cur_pooled = F.adaptive_avg_pool2d(f_cur, 1)
        f_pri_aligned_pooled = F.adaptive_avg_pool2d(f_pri_aligned, 1)
        f_pri_pooled = F.adaptive_avg_pool2d(f_pri, 1)
        fdif_pooled = F.adaptive_avg_pool2d(fdif_time_encoded, 1)

        # Concatenate pooled features
        fused_feat = torch.cat([f_cur_pooled, f_pri_aligned_pooled, fdif_pooled], dim=1).flatten(1)
        f_cur_flat = f_cur_pooled.flatten(1)
        f_pri_flat = f_pri_pooled.flatten(1)

        # Predict risk
        pred = {
            "pred_fused": torch.sigmoid(self.cumulative_prob_layer_fused(fused_feat)),
            "pred_cur": torch.sigmoid(self.cumulative_prob_layer_cur(f_cur_flat)),
            "pred_pri": torch.sigmoid(self.cumulative_prob_layer_pri(f_pri_flat)),
        }
        return pred


class TemporalRiskPredictionWithCumulativeProbLayer_no_alignment(nn.Module):
    def __init__(self, num_years=5, time_encoding_dim=512):
        """
        Temporal risk prediction without feature alignment.
        Args:
            num_years (int): Number of follow-up years.
            time_encoding_dim (int): Dimensionality for time encoding (not used here).
        """
        super().__init__()
        self.positional_encoding = ContinuousPosEncoding(dim=time_encoding_dim)

        self.cumulative_prob_layer_fused = CumulativeProbabilityLayer(num_features=2 * 512, max_followup=num_years + 1)
        self.cumulative_prob_layer_cur = CumulativeProbabilityLayer(num_features=512, max_followup=num_years + 1)
        self.cumulative_prob_layer_pri = CumulativeProbabilityLayer(num_features=512, max_followup=num_years + 1)

    def forward(self, f_cur, f_pri, time_gap):
        """
        Args:
            f_cur (Tensor): Current features (B, C, H, W)
            f_pri (Tensor): Prior features (B, C, H, W)
            time_gap (Tensor): Time gap (B,)
        Returns:
            dict: Dictionary of risk predictions (fused, cur, pri)
        """
        # Global average pooling
        f_cur_pooled = F.adaptive_avg_pool2d(f_cur, 1)
        f_pri_pooled = F.adaptive_avg_pool2d(f_pri, 1)

        f_cur_flat = f_cur_pooled.flatten(1)
        f_pri_flat = f_pri_pooled.flatten(1)
        fused_feat = torch.cat([f_cur_flat, f_pri_flat], dim=1)

        pred = {
            "pred_fused": torch.sigmoid(self.cumulative_prob_layer_fused(fused_feat)),
            "pred_cur": torch.sigmoid(self.cumulative_prob_layer_cur(f_cur_flat)),
            "pred_pri": torch.sigmoid(self.cumulative_prob_layer_pri(f_pri_flat)),
        }
        return pred