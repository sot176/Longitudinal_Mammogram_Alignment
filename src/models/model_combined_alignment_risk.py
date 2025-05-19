import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_feat_alignment import (FeatureAlignmentModel,
                                             ResNet18Encoder,
                                             SpatialTransformerBlock)
from src.models.model_risk_prediction import (
    TemporalRiskPredictionWithCumulativeProbLayer,
    TemporalRiskPredictionWithCumulativeProbLayer_no_alignment)

class CombinedAlignmentRiskModel(nn.Module):
    def __init__(self, in_channels=512, num_years=5):
        """
        Model combining feature alignment with temporal risk prediction.
        """
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.alignment = FeatureAlignmentModel(in_channels)
        self.risk_head = TemporalRiskPredictionWithCumulativeProbLayer(num_years)

    def forward(self, img_cur, img_pri, time_gap):
        # Ensure inputs are 3-channel
        img_cur = self._expand_channels(img_cur)
        img_pri = self._expand_channels(img_pri)

        # Encode current and prior images
        fcur = self.encoder(img_cur)
        fpri = self.encoder(img_pri)

        # Align features
        alignment_out = self.alignment(fcur, fpri)

        # Risk prediction
        risk_out = self.risk_head(
            alignment_out["current_features"],
            alignment_out["prior_feature_before_alignment"],
            alignment_out["aligned_prior"],
            alignment_out["differential_feature"],
            time_gap,
        )

        return {
            "risk_prediction": risk_out,
            "deformation_field": alignment_out["deformation_field"],
            "aligned_prior_feature": alignment_out["aligned_prior"],
            "prior_feature_before_alignment": alignment_out["prior_feature_before_alignment"],
            "current_feature": alignment_out["current_features"],
            "diff_feature": alignment_out["differential_feature"],
        }

    @staticmethod
    def _expand_channels(img):
        return img if img.shape[1] == 3 else img.repeat(1, 3, 1, 1)


class RiskModelNoAlignment(nn.Module):
    def __init__(self, in_channels=512, num_years=5):
        """
        Baseline model without alignment module.
        """
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.risk_head = TemporalRiskPredictionWithCumulativeProbLayer_no_alignment(num_years)

    def forward(self, img_cur, img_pri, time_gap):
        # Ensure inputs are 3-channel
        img_cur = self._expand_channels(img_cur)
        img_pri = self._expand_channels(img_pri)

        # Encode both images
        fcur = self.encoder(img_cur)
        fpri = self.encoder(img_pri)

        # Predict risk
        risk_out = self.risk_head(fcur, fpri, time_gap)

        return {
            "risk_prediction": risk_out
        }

    @staticmethod
    def _expand_channels(img):
        return img if img.shape[1] == 3 else img.repeat(1, 3, 1, 1)


class CombinedImgAlignmentRiskModel(nn.Module):
    def __init__(self, num_years=5, registration_model=None):
        """
        Combined model for image-based alignment and temporal risk prediction.

        Args:
            num_years (int): Number of years for risk prediction.
            registration_model (nn.Module): Pretrained image registration model.
        """
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.risk_head = TemporalRiskPredictionWithCumulativeProbLayer(num_years)

        # Set up pretrained registration model
        self.registration_model = registration_model
        if self.registration_model is not None:
            self.registration_model.eval()
            for param in self.registration_model.parameters():
                param.requires_grad = False

    def forward(self, img_cur, img_pri, time_gap):
        # Step 1: Apply image registration to align prior image
        with torch.no_grad():
            reg_outputs = self.registration_model(img_cur, img_pri)
            warped_pri_img = reg_outputs[0]          # Affine + deformable aligned image
            deformation_field = reg_outputs[1]       # Displacement field

        # Step 2: Ensure inputs are 3-channel
        img_cur = self._expand_channels(img_cur)
        img_pri = self._expand_channels(img_pri)
        warped_pri_img = self._expand_channels(warped_pri_img)

        # Step 3: Extract features
        f_cur = self.encoder(img_cur)
        f_pri = self.encoder(img_pri)
        f_pri_aligned = self.encoder(warped_pri_img)

        # Step 4: Normalize and compute difference
        f_cur_norm = self._normalize_features(f_cur)
        f_pri_aligned_norm = self._normalize_features(f_pri_aligned)
        f_diff = torch.abs(f_cur_norm - f_pri_aligned_norm)

        # Step 5: Risk prediction
        risk_pred = self.risk_head(f_cur, f_pri, f_pri_aligned, f_diff, time_gap)

        return {
            "risk_prediction": risk_pred,
            "deformation_field": deformation_field,
            "aligned_prior_feature": f_pri_aligned,
            "prior_feature_before_alignment": f_pri,
            "current_feature": f_cur,
            "diff_feature": f_diff,
        }

    @staticmethod
    def _expand_channels(img):
        """Ensure the image has 3 channels by repeating if necessary."""
        return img if img.shape[1] == 3 else img.repeat(1, 3, 1, 1)

    @staticmethod
    def _normalize_features(feat):
        """Normalize features across spatial dimensions."""
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True) + 1e-6
        return (feat - mean) / std


class CombinedImgAlignmentRiskModel_downsample_img_deformation_field(nn.Module):
    def __init__(self, num_years=5, registration_model=None):
        """
        Initialize the combined model.

        Args:
            in_channels (int): Number of input channels for the encoder.
            num_years (int): Number of years for risk prediction.
            registration_model (nn.Module): Pretrained registration model.
        """
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.registration_model = registration_model
        self.feat_transformer = SpatialTransformerBlock(mode="bilinear")
        self.risk_head = TemporalRiskPredictionWithCumulativeProbLayer(num_years)

        if self.registration_model is not None:
            self.registration_model.eval()
            for param in self.registration_model.parameters():
                param.requires_grad = False

    def forward(self, img_cur, img_pri, time_gap):
        # Step 1: Align images using registration model
        with torch.no_grad():
            warped_img, deformation_field = self.registration_model(img_cur, img_pri)

        img_cur = self._expand_channels(img_cur)
        img_pri = self._expand_channels(img_pri)

        # Step 2: Extract features
        f_cur = self.encoder(img_cur)
        f_pri = self.encoder(img_pri)

        # Step 3: Downsample deformation field to match feature resolution
        deformation_field_ds = F.interpolate(
            deformation_field.detach(),
            size=f_cur.shape[2:],
            mode="bilinear",
            align_corners=True
        )

        # Rescale displacement values
        scale_y = f_cur.shape[2] / img_cur.shape[2]
        scale_x = f_cur.shape[3] / img_cur.shape[3]
        deformation_field_ds[:, 0, :, :] *= scale_x
        deformation_field_ds[:, 1, :, :] *= scale_y

        # Move to same device
        deformation_field_ds = deformation_field_ds.to(f_cur.device)

        # Step 4: Apply deformation field to align prior features
        f_pri_aligned = self.feat_transformer(f_pri, deformation_field_ds)

        # Step 5: Normalize and compute differential features
        f_cur_norm = self._normalize_features(f_cur)
        f_pri_aligned_norm = self._normalize_features(f_pri_aligned)
        f_diff = torch.abs(f_cur_norm - f_pri_aligned_norm)

        # Step 6: Predict risk
        risk_pred = self.risk_head(f_cur, f_pri, f_pri_aligned, f_diff, time_gap)

        return {
            "risk_prediction": risk_pred,
            "deformation_field": deformation_field,
            "aligned_prior_feature": f_pri_aligned,
            "prior_feature_before_alignment": f_pri,
            "current_feature": f_cur,
            "diff_feature": f_diff,
        }

    @staticmethod
    def _expand_channels(img):
        """Ensure input has 3 channels."""
        return img if img.shape[1] == 3 else img.repeat(1, 3, 1, 1)

    @staticmethod
    def _normalize_features(feat):
        """Normalize feature map across spatial dimensions."""
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True) + 1e-6
        return (feat - mean) / std
