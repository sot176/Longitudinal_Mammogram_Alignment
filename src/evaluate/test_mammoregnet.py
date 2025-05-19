import time
import networks
from PIL import Image

from src.utils.utils import *
from src.train.losses_mammoregnet import  *

def normalize(arr):
    rng = arr.max() - arr.min()
    amin = arr.min()
    return (arr - amin) * 255.0 / rng


def plot_images(
    img_fix,
    img_mov,
    warped_img,
    affine_img,
    fix_id,
    mov_id,
    ncc_before,
    ncc_final,
    ncc_affine,
    njd,
    out_dir,
):
    plt.figure(figsize=(18, 12))
    fig_name = f"Fixed_image_{fix_id[:-4]}_moving_image_{mov_id[:-4]}.png"

    # Helper to convert tensor image to numpy
    def to_numpy(img):
        return img.squeeze().cpu().numpy()

    # Plot fixed image
    plt.subplot(2, 3, 1)
    plt.imshow(to_numpy(img_fix), cmap="gray")
    plt.axis("off")
    plt.title(f"Fixed image {fix_id[:-4]}")

    # Plot moving image
    plt.subplot(2, 3, 2)
    plt.imshow(to_numpy(img_mov), cmap="gray")
    plt.axis("off")
    plt.title(f"Moving image {mov_id[:-4]}")

    # Plot warped moving image
    plt.subplot(2, 3, 3)
    plt.imshow(to_numpy(warped_img), cmap="gray")
    plt.axis("off")
    plt.title("Warped moving image")

    # Fixed image overlay moving image
    plt.subplot(2, 3, 4)
    plt.imshow(to_numpy(img_fix), cmap="gray")
    plt.imshow(to_numpy(img_mov), cmap="Blues", alpha=0.6)
    plt.axis("off")
    plt.title("Fixed image overlayed with moving image")

    # Fixed image overlay affine transformed moving image
    plt.subplot(2, 3, 5)
    plt.imshow(to_numpy(img_fix), cmap="gray")
    plt.imshow(to_numpy(affine_img), cmap="Blues", alpha=0.6)
    plt.axis("off")
    plt.title("Fixed image overlayed with affine transformed moving image")

    # Fixed image overlay final transformed moving image
    plt.subplot(2, 3, 6)
    plt.imshow(to_numpy(img_fix), cmap="gray")
    plt.imshow(to_numpy(warped_img), cmap="Blues", alpha=0.6)
    plt.axis("off")
    plt.title("Fixed image overlayed with final transformed moving image")

    plt.suptitle(
        f"NCC before: {ncc_before:.6f} , NCC after: {ncc_final:.6f} , NCC only affine: {ncc_affine:.6f} , NJD (%): {njd}",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    os.makedirs(out_dir, exist_ok=True)  # Ensure output directory exists
    plt.savefig(os.path.join(out_dir, fig_name))
    plt.close()


def plot_deformation_field_jac_det(
    fix_id, mov_id, njd, out_dir, displacement_field_np, J_det
):
    """
    Plot deformation field vectors and the Jacobian determinant side-by-side.

    Args:
        fix_id (str): Identifier for the fixed image.
        mov_id (str): Identifier for the moving image.
        njd (float): Normalized Jacobian determinant metric (NJD percentage).
        out_dir (str): Directory to save the output figure.
        displacement_field_np (np.ndarray): Displacement field array of shape (H, W, 2).
        J_det (np.ndarray): Jacobian determinant array of shape (H, W).
    """
    plt.figure(figsize=(18, 12))

    # Construct filename based on IDs (strip extensions)
    fig_name = f"Deformation_field_Fixed_{fix_id[:-4]}_moving_{mov_id[-8:-4]}.png"

    # Create subplots side-by-side
    ax_deform = plt.subplot(1, 2, 1)
    ax_jacobian = plt.subplot(1, 2, 2)

    # Downsampling step for quiver plot clarity
    step = 2
    H, W = displacement_field_np.shape[:2]

    # Create meshgrid for plotting vectors
    y, x = np.mgrid[0:H:step, 0:W:step]

    # Downsample displacement field for visualization
    deformation_downsampled = displacement_field_np[::step, ::step]

    # Extract vector components (u: x-direction, v: y-direction)
    u = deformation_downsampled[..., 0]
    v = deformation_downsampled[..., 1]

    # Plot deformation vectors as quiver plot
    ax_deform.quiver(x, y, u, v, color="red", angles="xy", scale_units="xy", scale=1)
    ax_deform.axis("off")
    ax_deform.set_aspect("equal")
    ax_deform.set_title("Deformation Field", fontsize=14)

    # Plot Jacobian determinant heatmap
    vmin, vmax = -2, 2  # Symmetric range around zero for color scale
    img = ax_jacobian.imshow(
        np.squeeze(J_det), cmap="RdBu", interpolation="nearest", vmin=vmin, vmax=vmax
    )

    # Create colorbar axis aligned to the right of the heatmap
    divider = make_axes_locatable(ax_jacobian)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Add colorbar with formatting
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=12, width=1.5)

    ax_jacobian.axis("off")
    ax_jacobian.set_title("Jacobian Determinant of Displacement Field", fontsize=14)

    # Add a main title with NJD percentage
    plt.suptitle(f"NJD (%): {njd:.2f}", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save and close the figure
    plt.savefig(os.path.join(out_dir, fig_name))
    plt.close()

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


def test(test_loader, device, path_saved_model, path_logger, out_dir):
    # initialize logger
    logger = create_logger(path_logger)

    # Load trained model
    model = networks.MammoRegNet()
    model.load_state_dict(torch.load(path_saved_model, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()

    # Load transformation blocks
    spatial_transformer = networks.SpatialTransformer_block(mode="nearest").to(device).eval()
    affine_transformer = networks.AffineTransformer_block(mode="nearest").to(device).eval()

    # Loss functions
    ncc_loss = NCC().loss
    jacobian_analyzer = NJD()

    # Metrics tracking
    ncc_before_total = 0.0
    ncc_final_total = 0.0
    ncc_affine_total = 0.0
    njd_total = 0.0
    time_total = 0.0
    ncc_final_list = []
    ncc_affine_list = []
    njd_list = []

    counter = 0
    with torch.no_grad():
        for batch in test_loader:
            for idx in range(batch["img_fix"].shape[0]):
                counter += 1

                img_fix = batch["img_fix"][idx].to(device)
                img_mov = batch["img_mov"][idx].to(device)
                fix_id = batch["img_fix_id"][idx]
                mov_id = batch["img_mov_id"][idx]

                # Forward pass
                start_time = time.time()
                pred = model(img_fix.unsqueeze(0), img_mov.unsqueeze(0))
                time_elapsed = time.time() - start_time
                time_total += time_elapsed

                warped_img = pred[0][0]
                flow_field = pred[1][0]
                affine_img = pred[2][0]

                # Compute NCC metrics
                ncc_before = ncc_loss(img_fix, img_mov).item()
                ncc_final = ncc_loss(img_fix, warped_img).item()
                ncc_affine = ncc_loss(img_fix, affine_img).item()

                ncc_before_total += ncc_before
                ncc_final_total += ncc_final
                ncc_affine_total += ncc_affine
                ncc_final_list.append(ncc_final)
                ncc_affine_list.append(ncc_affine)

                # Downsample and scale flow for NJD
                flow_down = F.interpolate(flow_field.unsqueeze(0).cpu(), size=(32, 16), mode="bilinear",
                                          align_corners=True)
                flow_down[:, 0, :, :] *= 16 / 512  # width scale
                flow_down[:, 1, :, :] *= 32 / 1024  # height scale
                flow_np = flow_field.unsqueeze(0).permute(0, 2, 3, 1).cpu().numpy()
                flow_down_np = flow_down.permute(0, 2, 3, 1).squeeze(0).numpy()

                njd_value = NJD_percentage(flow_np).item()
                njd_total += njd_value
                njd_list.append(njd_value)

                # Generate visualizations
                warped_np = warped_img.squeeze(0).cpu().numpy()
                warped_pil = Image.fromarray(warped_np.astype(np.uint8))
                warped_pil.save(os.path.join(out_dir, f"{mov_id[:-4]}_warped.png"))

                jacobian = jacobian_analyzer.get_Ja(flow_down_np)
                plot_deformation_field_jac_det(fix_id, mov_id, njd_value, out_dir, flow_down_np, jacobian)
                plot_images(img_fix, img_mov, warped_img, affine_img, fix_id, mov_id, ncc_before, ncc_final, ncc_affine,
                            njd_value, out_dir)

    # Compute averages
    avg_ncc_before = ncc_before_total / counter
    avg_ncc_final = ncc_final_total / counter
    avg_ncc_affine = ncc_affine_total / counter
    avg_njd = njd_total / counter
    avg_time = time_total / counter

    # Compute confidence intervals
    results = {
        "NJD": {
            "Mean": avg_njd,
            "95% CI": bootstrap_confidence_interval(np.array(njd_list)),
        },
        "NCC Before": avg_ncc_before,
        "NCC Only Affine": {
            "Mean": avg_ncc_affine,
            "95% CI": bootstrap_confidence_interval(np.array(ncc_affine_list)),
        },
        "Final NCC": {
            "Mean": avg_ncc_final,
            "95% CI": bootstrap_confidence_interval(np.array(ncc_final_list)),
        },
    }

    # Logging
    logger.info(f"Number of image pairs: {counter}")
    logger.info(f"Average NCC before registration: {avg_ncc_before}")
    logger.info(f"Average NCC (final): {avg_ncc_final}")
    logger.info(f"Average NCC (affine only): {avg_ncc_affine}")
    logger.info(f"Average NJD: {avg_njd}")
    logger.info(f"Average registration time per image: {avg_time}")
    logger.info(f"Results with confidence intervals: {results}")

    # Console output
    print(f"\n--- Test Summary ({counter} image pairs) ---")
    print(f"Average NCC Before       : {avg_ncc_before:.4f}")
    print(f"Average NCC Final        : {avg_ncc_final:.4f}")
    print(f"Average NCC Affine Only  : {avg_ncc_affine:.4f}")
    print(f"Average NJD              : {avg_njd:.2f}%")
    print(f"Average Time/Image       : {avg_time:.3f} seconds")
    print(f"NCC Final 95% CI         : {results['Final NCC']['95% CI']}")
    print(f"NCC Affine 95% CI        : {results['NCC Only Affine']['95% CI']}")
    print(f"NJD 95% CI               : {results['NJD']['95% CI']}")