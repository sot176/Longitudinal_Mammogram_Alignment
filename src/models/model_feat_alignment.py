import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, resnet


# 1. ResNet-18 Encoder for feature extraction
class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool & fc layers

    def forward(self, x):
        """
        Args:
            x (Tensor): Input image tensor of shape [B, C, H, W]

        Returns:
            Tensor: Extracted feature maps of shape [B, C', H', W']
        """
        return self.features(x)

# 2. Spatial Transformer for applying deformation fields
class SpatialTransformerBlock(nn.Module):
    def __init__(self, mode="bilinear"):
        """
        Applies a deformation field to an input tensor using grid sampling.

        Args:
            mode (str): Interpolation mode to use ('bilinear' or 'nearest')
        """
        super().__init__()
        self.mode = mode

    def forward(self, f_pri, deformation_field):
        """
        Args:
            f_pri (Tensor): Prior feature map of shape [B, C, H, W]
            deformation_field (Tensor): Flow field of shape [B, 2, H, W] (dx, dy)

        Returns:
            Tensor: Warped feature map of shape [B, C, H, W]
        """
        B, _, H, W = f_pri.shape

        # Generate identity grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=f_pri.device),
            torch.arange(W, device=f_pri.device),
            indexing="ij"
        )
        grid = torch.stack((grid_y, grid_x), dim=0).float()  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)           # [B, 2, H, W]

        # Add deformation
        new_grid = grid + deformation_field.to(f_pri.device)  # [B, 2, H, W]

        # Normalize to [-1, 1]
        new_grid[:, 0, :, :] = 2.0 * (new_grid[:, 0, :, :] / (H - 1) - 0.5)
        new_grid[:, 1, :, :] = 2.0 * (new_grid[:, 1, :, :] / (W - 1) - 0.5)

        # Reshape to [B, H, W, 2] and flip last dim to match grid_sample format
        new_grid = new_grid.permute(0, 2, 3, 1)[..., [1, 0]]  # [B, H, W, 2]

        # Apply spatial transformation
        f_pri_aligned = F.grid_sample(
            f_pri, new_grid, mode=self.mode, align_corners=True
        )

        return f_pri_aligned


class SpatialTransformerImage(nn.Module):
    """
    Applies a deformation field to warp an image using bilinear or nearest-neighbor sampling.
    """

    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        """
        Args:
            src (Tensor): Source image tensor of shape (B, C, H, W)
            flow (Tensor): Deformation field of shape (B, 2, h, w) with (dx, dy) offsets

        Returns:
            Tensor: Warped image of shape (B, C, H, W)
        """
        B, C, H, W = src.shape
        _, _, h, w = flow.shape

        # Upsample flow to match image resolution
        flow_upsampled = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=True)

        # Rescale flow to full image resolution
        scale_y, scale_x = H / h, W / w
        flow_upsampled[:, 0] *= scale_x  # dx
        flow_upsampled[:, 1] *= scale_y  # dy

        # Create mesh grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=src.device),
            torch.arange(W, device=src.device),
            indexing="ij"
        )
        grid = torch.stack((y_grid, x_grid), dim=0).float()  # (2, H, W)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)           # (B, 2, H, W)

        # Apply deformation
        new_coords = grid + flow_upsampled

        # Normalize to [-1, 1] for grid_sample
        new_coords[:, 0] = 2.0 * (new_coords[:, 0] / (H - 1) - 0.5)
        new_coords[:, 1] = 2.0 * (new_coords[:, 1] / (W - 1) - 0.5)

        # Convert to shape (B, H, W, 2) and swap axes to (x, y)
        grid_normalized = new_coords.permute(0, 2, 3, 1)[..., [1, 0]]

        # Warp the source image
        warped_img = F.grid_sample(src, grid_normalized, mode=self.mode, align_corners=True)

        return warped_img


class AlignmentBlockDeformable(nn.Module):
    """
    Predicts a dense deformation field to align feature maps via convolutional layers.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Output 2-channel deformation field: (dx, dy)
        self.deformation_head = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, f_cur, f_pri):
        """
        Args:
            f_cur (Tensor): Current feature map (B, C, H, W)
            f_pri (Tensor): Prior feature map (B, C, H, W)

        Returns:
            Tensor: Deformation field (B, 2, H, W)
        """
        x = torch.cat([f_cur, f_pri], dim=1)         # (B, 2*C, H, W)
        x = self.relu(self.bn1(self.conv1(x)))       # Feature extraction
        flow = self.deformation_head(x)              # Predict deformation

        return flow


# 5. Full Model
class FeatureAlignmentModel(nn.Module):
    """
    Feature Alignment Model for aligning prior and current feature maps using a predicted deformation field.

    Args:
        in_channels (int): Number of channels in input feature maps.
    """

    def __init__(self, in_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.alignment_block = AlignmentBlockDeformable(in_channels)
        self.spatial_transformer = SpatialTransformerBlock(mode="bilinear")

    def forward(self, f_cur, f_pri):
        """
        Args:
            f_cur (Tensor): Current feature map of shape (B, C, H, W)
            f_pri (Tensor): Prior feature map of shape (B, C, H, W)

        Returns:
            dict: Contains aligned features, deformation field, and intermediate representations.
        """
        # Step 1: Predict deformation field from concatenated current and prior features
        deformation_field = self.alignment_block(f_cur, f_pri)

        # Step 2: Warp prior feature using the predicted deformation field
        f_pri_aligned = self.spatial_transformer(f_pri, deformation_field)

        # Step 3: Normalize feature maps before computing differences
        f_cur_norm = self._normalize(f_cur)
        f_pri_aligned_norm = self._normalize(f_pri_aligned)

        # Step 4: Compute absolute difference between aligned prior and current features
        diff_feat = torch.abs(f_cur_norm - f_pri_aligned_norm)

        return {
            "aligned_prior": f_pri_aligned,
            "current_features": f_cur,
            "prior_feature_before_alignment": f_pri,
            "differential_feature": diff_feat,
            "deformation_field": deformation_field,
        }

    @staticmethod
    def _normalize(feat):
        """Per-channel z-score normalization."""
        return (feat - feat.mean(dim=(2, 3), keepdim=True)) / (feat.std(dim=(2, 3), keepdim=True) + 1e-6)