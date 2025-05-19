import torch

class NCC:
    """
    Normalized Cross-Correlation (NCC) loss.
    Measures similarity between two images by penalizing intensity differences.
    """

    @staticmethod
    def norm_data(data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data to zero mean and unit variance.
        """
        return (data - data.mean()) / (data.std() + 1e-8)

    def loss(self, data0: torch.Tensor, data1: torch.Tensor) -> torch.Tensor:
        """
        Compute NCC between two tensors.

        Args:
            data0: First image tensor.
            data1: Second image tensor.

        Returns:
            Normalized cross-correlation coefficient.
        """
        ncc_val = (1.0 / (data0.numel() - 1)) * torch.sum(
            self.norm_data(data0) * self.norm_data(data1)
        )
        return ncc_val if not torch.isnan(ncc_val) else torch.tensor(0.0, device=data0.device)


class Grad:
    """
    Gradient-based smoothness regularization (L1 or L2 penalty).
    Encourages smooth displacement fields.
    """

    def __init__(self, penalty: str = "l2", loss_mult: float = None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient regularization loss on a 2D displacement field.

        Args:
            y_pred: Displacement field of shape (B, 2, H, W)

        Returns:
            Scalar gradient loss.
        """
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == "l2":
            dy = dy ** 2
            dx = dx ** 2

        grad_loss = 0.5 * (dx.mean() + dy.mean())
        return grad_loss * self.loss_mult if self.loss_mult is not None else grad_loss

class NJD:
    """
    Negative Jacobian Determinant (NJD) loss.
    Penalizes regions of the deformation field with folding (non-invertibility).
    """

    def __init__(self, Lambda: float = 1e-5):
        self.Lambda = Lambda

    def get_Ja(self, displacement: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian determinant for 2D deformation field.

        Args:
            displacement: Tensor of shape (B, H, W, 2)

        Returns:
            Tensor of Jacobian determinant values.
        """
        D_y = displacement[:, 1:, :-1, :] - displacement[:, :-1, :-1, :]
        D_x = displacement[:, :-1, 1:, :] - displacement[:, :-1, :-1, :]

        D1 = (D_x[..., 0] + 1) * (D_y[..., 1] + 1)
        D2 = D_x[..., 1] * D_y[..., 0]
        return D1 - D2

    def loss(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute NJD loss for a batch of displacement fields.

        Args:
            y_pred: Tensor of shape (B, 2, H, W)

        Returns:
            Scalar NJD loss.
        """
        disp = y_pred.permute(0, 2, 3, 1)  # Shape: (B, H, W, 2)
        Ja = self.get_Ja(disp)
        neg_Jac = 0.5 * (torch.abs(Ja) - Ja)
        return self.Lambda * torch.sum(neg_Jac)


def Regu_loss(y_pred: torch.Tensor) -> torch.Tensor:
    """
    Combined regularization loss: smoothness + negative Jacobian.

    Args:
        y_pred: Displacement field tensor of shape (B, 2, H, W)

    Returns:
        Scalar total regularization loss.
    """
    return Grad(penalty="l2").loss(y_pred) + NJD(Lambda=1e-5).loss(y_pred)