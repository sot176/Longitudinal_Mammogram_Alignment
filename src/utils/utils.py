import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import metrics
from sklearn.utils import resample

from src.utils.c_index import concordance_index_ipcw


def get_risk_loss_BCE(pred, y_true, y_mask):
    """
    Compute the Binary Cross-Entropy (BCE) loss for risk prediction with masking.
    Parameters:
    - pred: Tensor of shape (batch_size, n+1), predicted probabilities.
    - y_true: Tensor of shape (batch_size, n), ground truth binary labels.
    - followup_years: Tensor of shape (batch_size,), last follow-up year for each sample.
    - n_years: Integer, number of years for prediction (n).
    Returns:
    - loss: The computed BCE loss.
    """

    y_mask = y_mask.to(pred.device)
    y_true = y_true.to(pred.device)
    masked_loss = F.binary_cross_entropy(
        pred, y_true.float(), weight=y_mask.float(), size_average=False
    ) / torch.sum(y_mask.float())
    return masked_loss


def normalize_feature_map(feature_map):
    # Min-Max normalization
    mean = feature_map.mean(dim=(1, 2, 3), keepdim=True)  # Mean per channel
    std = feature_map.std(dim=(1, 2, 3), keepdim=True)  # Std per channel
    normalized_map = (feature_map - mean) / (std + 1e-8)  # Small epsilon for stability
    return normalized_map


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def create_logger(log_path):
    # create custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create handlers
    f_handler = logging.FileHandler(log_path)

    # create formatters and add it to handlers
    f_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(f_handler)

    return logger


def bootstrap_auc_by_density(
    event_times,
    predictions,
    event_observed,
    density_categories,
    n_bootstrap=1000,
    alpha=0.05,
):
    auc_results_by_density = {
        "A": {f"Year {i + 1}": [] for i in range(5)},
        "B": {f"Year {i + 1}": [] for i in range(5)},
        "C": {f"Year {i + 1}": [] for i in range(5)},
        "D": {f"Year {i + 1}": [] for i in range(5)},
    }

    # Iterate through each density category (A, B, C, D)
    for density in ["A", "B", "C", "D"]:
        # Filter data by the current density category
        density_indices = np.where(density_categories == density)[0]
        event_times_density = event_times[density_indices]
        predictions_density = predictions[density_indices]
        event_observed_density = event_observed[density_indices]
        # Resample with bootstrapping while keeping the balance between cancer and non-cancer
        cancer_indices = np.where(event_observed_density == 1)[0]
        non_cancer_indices = np.where(event_observed_density == 0)[0]

        # Skip if there are no cancer or non-cancer cases
        if len(cancer_indices) == 0 or len(non_cancer_indices) == 0:
            print(
                f"Skipping density '{density}' due to missing cancer or non-cancer cases."
            )
            continue

        for _ in range(n_bootstrap):
            # Resample cancer and non-cancer indices separately
            cancer_sample = resample(
                cancer_indices, replace=True, n_samples=len(cancer_indices)
            )
            non_cancer_sample = resample(
                non_cancer_indices, replace=True, n_samples=len(non_cancer_indices)
            )
            indices = np.concatenate([cancer_sample, non_cancer_sample])

            # Extract the sampled data
            event_times_sample = event_times_density[indices]
            predictions_sample = predictions_density[indices]
            event_observed_sample = event_observed_density[indices]

            # Compute the AUC for each year
            yearly_aucs_sample = compute_auc_x_year_auc(
                predictions_sample, event_times_sample, event_observed_sample
            )
            for year, auc in yearly_aucs_sample.items():
                auc_results_by_density[density][f"Year {year + 1}"].append(auc)

    # Calculate mean and confidence intervals for each year and density category
    auc_summary_by_density = {}
    for density, auc_results in auc_results_by_density.items():
        auc_summary_by_density[density] = {}
        for year, auc_values in auc_results.items():
            if len(auc_values) > 0:
                lower = np.percentile(auc_values, 100 * alpha / 2)
                upper = np.percentile(auc_values, 100 * (1 - alpha / 2))
                auc_summary_by_density[density][year] = (
                    np.mean(auc_values),
                    (lower, upper),
                )
            else:
                auc_summary_by_density[density][year] = (
                    None,
                    (None, None),
                )  # Handle missing values

    return auc_summary_by_density


def bootstrap_c_index(
    event_times,
    predictions,
    event_observed,
    censoring_dist,
    n_bootstrap=1000,
    alpha=0.05,
):
    c_index_scores = []

    # Identify cancer (event=1) and non-cancer (event=0) indices
    cancer_indices = np.where(event_observed == 1)[0]
    non_cancer_indices = np.where(event_observed == 0)[0]

    for _ in range(n_bootstrap):
        # Stratified resampling
        cancer_sample = resample(
            cancer_indices, replace=True, n_samples=len(cancer_indices)
        )
        non_cancer_sample = resample(
            non_cancer_indices, replace=True, n_samples=len(non_cancer_indices)
        )
        # Combine cancer and non-cancer cases
        indices = np.concatenate([cancer_sample, non_cancer_sample])

        # Create bootstrap samples
        event_times_sample = event_times[indices]
        predictions_sample = predictions[indices]
        event_observed_sample = event_observed[indices]

        # Calculate C-index for the bootstrap sample
        c_index = concordance_index_ipcw(
            event_times_sample,
            predictions_sample,
            event_observed_sample,
            censoring_dist,
        )
        c_index_scores.append(c_index)

    # Compute confidence intervals
    lower = np.percentile(c_index_scores, 100 * alpha / 2)
    upper = np.percentile(c_index_scores, 100 * (1 - alpha / 2))

    return np.mean(c_index_scores), (lower, upper)


def bootstrap_c_index_by_density(
    event_times,
    predictions,
    event_observed,
    density_categories,
    censoring_dist,
    n_bootstrap=1000,
    alpha=0.05,
):
    c_index_results_by_density = {density: [] for density in ["A", "B", "C", "D"]}

    for density in ["A", "B", "C", "D"]:
        # Filter data by density category
        density_indices = np.where(density_categories == density)[0]
        event_times_density = event_times[density_indices]
        predictions_density = predictions[density_indices]
        event_observed_density = event_observed[density_indices]

        # Identify cancer (event=1) and non-cancer (event=0) cases
        cancer_indices = np.where(event_observed_density == 1)[0]
        non_cancer_indices = np.where(event_observed_density == 0)[0]

        # Skip if there are no cancer or non-cancer cases
        if len(cancer_indices) == 0 or len(non_cancer_indices) == 0:
            print(
                f"Skipping density '{density}' due to missing cancer or non-cancer cases."
            )
            continue

        for _ in range(n_bootstrap):
            # Bootstrap resampling within the density category
            cancer_sample = resample(
                cancer_indices, replace=True, n_samples=len(cancer_indices)
            )
            non_cancer_sample = resample(
                non_cancer_indices, replace=True, n_samples=len(non_cancer_indices)
            )
            indices = np.concatenate([cancer_sample, non_cancer_sample])

            # Extract the resampled data
            event_times_sample = event_times_density[indices]
            predictions_sample = predictions_density[indices]
            event_observed_sample = event_observed_density[indices]

            # Compute C-index
            c_index = concordance_index_ipcw(
                event_times_sample,
                predictions_sample,
                event_observed_sample,
                censoring_dist,
            )
            c_index_results_by_density[density].append(c_index)

    # Compute mean and confidence intervals for each density
    c_index_summary_by_density = {}
    for density, c_index_values in c_index_results_by_density.items():
        if len(c_index_values) > 0:
            lower = np.percentile(c_index_values, 100 * alpha / 2)
            upper = np.percentile(c_index_values, 100 * (1 - alpha / 2))
            c_index_summary_by_density[density] = (
                np.mean(c_index_values),
                (lower, upper),
            )
        else:
            c_index_summary_by_density[density] = (
                None,
                (None, None),
            )  # Handle missing values

    return c_index_summary_by_density


def bootstrap_confidence_interval(data, num_samples=1000, confidence_level=0.95):
    """
    Calculate the confidence interval using bootstrapping.
    :param data: List or numpy array of metric values
    :param num_samples: Number of bootstrap samples
    :param confidence_level: Confidence level for the interval (default: 95%)
    :return: (lower_bound, upper_bound)
    """
    data = np.array(data)
    bootstrapped_means = []
    for _ in range(num_samples):
        # Resample with replacement
        sample = np.random.choice(data, size=len(data), replace=True)
        # Calculate the mean of the sample
        bootstrapped_means.append(np.mean(sample))

    # Calculate the confidence interval
    alpha = 1 - confidence_level
    lower_bound = np.percentile(bootstrapped_means, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrapped_means, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound


def bootstrap_auc(
    event_times, predictions, event_observed, n_bootstrap=1000, alpha=0.05
):
    auc_results = {f"Year {i + 1}": [] for i in range(5)}

    cancer_indices = np.where(event_observed == 1)[0]
    non_cancer_indices = np.where(event_observed == 0)[0]

    for _ in range(n_bootstrap):
        # Resample while keeping class balance
        cancer_sample = resample(
            cancer_indices, replace=True, n_samples=len(cancer_indices)
        )
        non_cancer_sample = resample(
            non_cancer_indices, replace=True, n_samples=len(non_cancer_indices)
        )
        # Combine cancer and resampled non-cancer cases
        indices = np.concatenate([cancer_sample, non_cancer_sample])

        # Extract sampled data
        event_times_sample = event_times[indices]
        predictions_sample = predictions[indices]
        event_observed_sample = event_observed[indices]

        # Compute AUC
        yearly_aucs_sample = compute_auc_x_year_auc(
            predictions_sample, event_times_sample, event_observed_sample
        )
        for year, auc in yearly_aucs_sample.items():
            auc_results[f"Year {year + 1}"].append(auc)

    # Calculate mean and confidence intervals for each year
    auc_summary = {}
    for year, auc_values in auc_results.items():
        lower = np.percentile(auc_values, 100 * alpha / 2)
        upper = np.percentile(auc_values, 100 * (1 - alpha / 2))
        auc_summary[year] = (np.mean(auc_values), (lower, upper))

    return auc_summary


def print_results(results):
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")


def compute_auc_x_year_auc(probs, censor_times, golds):
    def include_exam_and_determine_label(prob_arr, censor_time, gold, followup):
        # Check if patient should be included for the current followup year

        valid_pos = (
            gold == 1 and censor_time <= followup
        )  # Event occurred before or at followup year
        valid_neg = censor_time >= followup
        included, label = (valid_pos or valid_neg), valid_pos
        return included, label

    aucs_per_year = {}

    for followup in range(5):  # Followup from 1 to 5 (inclusive)
        probs_for_eval, golds_for_eval = [], []
        for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
            include, label = include_exam_and_determine_label(
                prob_arr, censor_time, gold, followup
            )
            if include:
                # Add the predicted probability at the followup year
                probs_for_eval.append(prob_arr[followup])
                golds_for_eval.append(label)

        try:
            auc = metrics.roc_auc_score(
                golds_for_eval, probs_for_eval, average="samples"
            )
        except Exception as e:
            warnings.warn("Failed to calculate AUC because {}".format(e))
            auc = "NA"
        aucs_per_year[followup] = auc

    return aucs_per_year


def compute_auc_by_density_category(
    predictions, event_times, event_observed, density_categories
):
    """
    Compute AUC for each density category and each year (1 to 5).
    """
    # Initialize a dictionary to store AUCs by density category
    aucs_by_density = {"A": {}, "B": {}, "C": {}, "D": {}}

    # Iterate through the density categories and compute AUC for each
    for density in ["A", "B", "C", "D"]:
        # Filter the data by the current density category
        idx = [i for i, cat in enumerate(density_categories) if cat == density]
        probs = [predictions[i] for i in idx]
        event_times_filtered = [event_times[i] for i in idx]
        event_observed_filtered = [event_observed[i] for i in idx]

        # Calculate the AUC for each year for this density category
        aucs_by_density[density] = compute_auc_x_year_auc(
            probs, event_times_filtered, event_observed_filtered
        )

    return aucs_by_density


def NJD_percentage(displacement):
    """
    Calculate the Jacobian value at each point of the displacement field, displacement field has size b,h,w,c
    Returns the percentage of negative Jacobian Determinants, lower NJD indicates better registration performance
    """

    D_y = displacement[:, 1:, :-1, :] - displacement[:, :-1, :-1, :]
    D_x = displacement[:, :-1, 1:, :] - displacement[:, :-1, :-1, :]
    D1 = (D_x[..., 0] + 1) * (D_y[..., 1] + 1)
    D2 = (D_x[..., 1]) * (D_y[..., 0])
    Ja_value = D1 - D2
    percentage = 100.0 * (np.sum(Ja_value < 0) / np.sum(Ja_value))
    return percentage


class NJD:
    """
    Calculate the Jacobian value at each point of the displacement field, displacement field has size b,h,w,2
    """

    def __init__(self, Lambda=1e-5):
        self.Lambda = Lambda

    def get_Ja(self, displacement):
        D_y = displacement[:, 1:, :-1, :] - displacement[:, :-1, :-1, :]
        D_x = displacement[:, :-1, 1:, :] - displacement[:, :-1, :-1, :]
        D1 = (D_x[..., 0] + 1) * (D_y[..., 1] + 1)
        D2 = (D_x[..., 1]) * (D_y[..., 0])
        Ja_value = D1 - D2
        return Ja_value

    def loss(self, y_pred):
        """
        Penalizing locations where Jacobian of displacement field has negative determinants, y_pred has size b,2,h,w
        """
        displacement = y_pred.permute(0, 2, 3, 1)  # now displacement has shape b,h,w,2
        Ja = self.get_Ja(displacement)
        Neg_Jac = 0.5 * (torch.abs(Ja) - Ja)

        return self.Lambda * torch.sum(Neg_Jac)


class Grad:
    """
    Gradient loss (L2 regularization on the displacement field )
    """

    def __init__(self, penalty="l1", loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        "y_pred has size b,2,h,w"
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == "l2":
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad


def Regu_loss(y_pred):
    """ "
    Regularization Loss consists of a L2 regularization of the displacement field and
    a Jacobian determinant loss to penalizes locations of the displacement field where the Jacobian determinant is negative
    """
    return Grad("l2").loss(y_pred) + NJD(1e-5).loss(y_pred)


def plot_deformation_field(fix_id, mov_id, njd, out_dir, displacement_field_np, J_det):
    f = plt.figure(figsize=(18, 12))
    fig_name = "Deformation_field_Fixed_{}_moving_{}.png".format(
        fix_id[:-4], mov_id[-8:-4]
    )

    ax6 = f.add_subplot(121)
    ax7 = f.add_subplot(122)

    step = 2
    H, W = displacement_field_np.shape[:2]
    y, x = np.mgrid[0:H:step, 0:W:step]
    # Downsample the deformation field for better visualization
    deformation_field_downsampled = displacement_field_np[::step, ::step]
    u = deformation_field_downsampled[..., 0]  # Displacement in x-direction (negated)
    v = deformation_field_downsampled[..., 1]  # Displacement in y-direction (negated)
    ax6.quiver(x, y, u, v, color="red")
    ax6.axis("off")
    ax6.set_aspect("equal")
    ax6.set_title("Deformation Field")

    vmin = -2  # Ensures symmetry around zero
    vmax = 2

    img = ax7.imshow(
        np.squeeze(J_det), cmap="RdBu", interpolation="nearest", vmin=vmin, vmax=vmax
    )
    # Make the colorbar aligned with the y-axis
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Add colorbar with label
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label("")
    # Adjust the size of the ticks and labels
    cbar.ax.tick_params(labelsize=26)  # Increase the size of the tick labels
    cbar.ax.tick_params(width=2)  # Increase the width of the ticks (optional)
    ax7.axis("off")
    ax7.set_title("Jacobian Determinant of Displacement Field")

    plt.suptitle(" NJD (%): " + str(njd), fontsize=10)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, fig_name))
    plt.close()
