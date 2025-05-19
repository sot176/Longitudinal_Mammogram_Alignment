import matplotlib.pyplot as plt
import numpy as np

# Example C-index values (replace with actual results)
densities = ["A", "B", "C", "D"]
metrics_1 = ["1-Y AUC", "2-Y AUC", "3-Y AUC", "4-Y AUC", "5-Y AUC"]  # AUC for years 1-5
densities_libra = ["Low", "Medium", "High"]

# Feature alignment C-index
feature_c_index = [0.681, 0.636, 0.625, 0.579]
feature_regu_c_index = [0.656, 0.626, 0.643, 0.516]

# Image alignment C-index
image_c_index = [0.792, 0.655, 0.651, 0.685]
image1_c_index = [0.689, 0.648, 0.667, 0.578]

feature_c_index_csaw = [0.587, 0.524, 0.531]
feature_regu_c_index_csaw = [0.572, 0.535, 0.542]

# Image alignment C-index
image_c_index_csaw = [0.619, 0.629, 0.667]
image1_c_index_csaw = [0.627, 0.583, 0.655]


# Weighting values on the x-axis
weightings = ["1/100", "1/75", "1/50", "1/25"]

# AUC data for 5 years and 4 weightings
without_reg_data_auc = np.array(
    [
        [0.642, 0.651, 0.630, 0.600],  # 1-Y AUC
        [0.645, 0.648, 0.626, 0.601],  # 2-Y AUC
        [0.654, 0.649, 0.631, 0.616],  # 3-Y AUC
        [0.653, 0.657, 0.627, 0.607],  # 4-Y AUC
        [0.655, 0.660, 0.628, 0.610],  # 5-Y AUC
    ]
)

# C-index data for 4 weightings
without_reg_data_c_index = np.array([0.649, 0.651, 0.632, 0.607])  # C-index

# NJD data for 4 weightings
without_reg_data_njd = np.array([1.6273, 2.0765, 1.1721, 1.0225])  # NJD

# Create a 1x3 grid of subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

# Define colors for plotting
auc_colors = ["b", "g", "y", "c", "m"]  # Default colors for AUC 1-Y to 5-Y AUC
njd_color = "red"  # Red for NJD
c_index_color = "royalblue"  # Color for C-index plot

# Plot C-index vs Density (subplot 1)
x_positions_csaw = np.arange(len(densities_libra))

# Plot Feature Alignment C-index
ax1.plot(
    x_positions_csaw,
    feature_c_index_csaw,
    marker="s",
    linestyle="--",
    color="royalblue",
    label="FeatAlign",
)
ax1.plot(
    x_positions_csaw,
    feature_regu_c_index_csaw,
    marker="^",
    linestyle="--",
    color="darkorange",
    label="FeatAlignReg",
)

# Plot Image Alignment C-index
ax1.plot(
    x_positions_csaw,
    image_c_index_csaw,
    marker="o",
    linestyle="-",
    color="seagreen",
    label="ImgAlign",
)
ax1.plot(
    x_positions_csaw,
    image1_c_index_csaw,
    marker="D",
    linestyle="-",
    color="crimson",
    label="ImgFeatAlign",
)

# Formatting for the first subplot
ax1.set_xticks(x_positions_csaw)
ax1.set_xticklabels(densities_libra)
ax1.set_xlabel("Density Category", fontsize=18)
ax1.set_ylabel("C-index", fontsize=18)
ax1.set_title("C-index by Density", fontsize=18)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.tick_params(labelsize=18)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=18)

x_positions = np.arange(len(densities))

# Plot Feature Alignment C-index
ax2.plot(
    x_positions,
    feature_c_index,
    marker="s",
    linestyle="--",
    color="royalblue",
    label="FeatAlign",
)
ax2.plot(
    x_positions,
    feature_regu_c_index,
    marker="^",
    linestyle="--",
    color="darkorange",
    label="FeatAlignReg",
)

# Plot Image Alignment C-index
ax2.plot(
    x_positions,
    image_c_index,
    marker="o",
    linestyle="-",
    color="seagreen",
    label="ImgAlign",
)
ax2.plot(
    x_positions,
    image1_c_index,
    marker="D",
    linestyle="-",
    color="crimson",
    label="ImgFeatAlign",
)

# Formatting for the first subplot
ax2.set_xticks(x_positions)
ax2.set_xticklabels(densities)
ax2.set_xlabel("Density Category", fontsize=18)
ax2.set_ylabel("C-index", fontsize=18)
ax2.set_title("C-index by Density", fontsize=18)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.tick_params(labelsize=18)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=18)

# Plot AUC vs Weighting (subplot 2)
for i in range(5):  # Plot AUC for years 1-5
    ax3.plot(
        weightings,
        without_reg_data_auc[i],
        label=f"{metrics_1[i]}",
        color=auc_colors[i],
    )

# Create a second y-axis for NJD in subplot 2
ax3_right = ax3.twinx()
ax3_right.plot(
    weightings,
    without_reg_data_njd,
    label="NJD",
    linestyle="dashed",
    color=njd_color,
    marker="x",
)
ax3.set_title("AUC vs Weighting", fontsize=18)
ax3.set_xlabel("Weighting", fontsize=18)
ax3.set_ylabel("AUC", fontsize=18)
ax3_right.set_ylabel("NJD", fontsize=18)
ax3.grid(True)
ax3_right.invert_yaxis()
ax3.tick_params(labelsize=18)
ax3_right.tick_params(axis="y", labelsize=18)

handles1, labels1 = ax3.get_legend_handles_labels()
handles2, labels2 = ax3_right.get_legend_handles_labels()
ax3.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    fontsize=18,
    ncol=3,
)

# Plot C-index vs Weighting (subplot 3) (Updated to only include C-index)
ax4.plot(
    weightings,
    without_reg_data_c_index,
    label="C-index",
    color=c_index_color,
    marker="s",
    linestyle="-",
)
ax4_right = ax4.twinx()
ax4_right.plot(
    weightings,
    without_reg_data_njd,
    label="NJD",
    linestyle="dashed",
    color=njd_color,
    marker="x",
)
ax4_right.set_ylabel("NJD", fontsize=18)
ax4_right.tick_params(axis="y", labelsize=18)
ax4_right.invert_yaxis()

ax4.set_title("C-index vs Weighting", fontsize=18)
ax4.set_xlabel("Weighting", fontsize=18)
ax4.set_ylabel("C-index", fontsize=18)
ax4.grid(True)
ax4.tick_params(labelsize=18)
# Place the legend below the third plot
handles1, labels1 = ax4.get_legend_handles_labels()
handles2, labels2 = ax4_right.get_legend_handles_labels()
ax4.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    fontsize=18,
    ncol=2,
)

# Adjust layout for better spacing
plt.tight_layout()


# Show the figure
plt.show()
