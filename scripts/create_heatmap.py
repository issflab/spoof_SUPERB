# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # -----------------------------
# # Data
# # -----------------------------
# models = [
#     "FBANK","APC","NPC","Mockingjay","mockingjay_960hr","Audio Albert","TERA",
#     "DeCoAR 2.0","wav2vec","modified CPC","wav2vec 2.0 Base","wav2vec 2.0 Large",
#     "HuBERT Base","HuBERTLarge","MR - HUBERT","XLS-R","Unispeech-SAT","Data2Vec",
#     "WAVLABLM","WavLM Large","SSAST","Byol-Audio","MAE_AST_FRAME"
# ]

# baseline = [
#     48.98,29.06,34.96,36.57,31.13,29.88,27.46,27.18,27.00,45.15,
#     16.47,11.25,21.63,18.48,17.35,6.60,11.21,23.96,17.27,13.56,
#     19.77,30.33,22.83
# ]

# codec = [
#     49.61,36.51,39.62,41.05,34.82,34.73,39.12,33.48,36.07,48.48,
#     24.56,24.05,29.31,27.55,27.83,20.27,21.32,30.91,26.39,22.96,
#     35.74,35.50,31.46
# ]

# noise = [
#     45.17,14.16,21.29,57.96,57.38,58.36,56.51,9.93,23.03,30.09,
#     26.82,30.17,12.46,4.83,5.77,3.77,2.32,11.33,6.17,4.71,
#     23.36,32.45,17.52
# ]

# reverb = [
#     51.2,50.39,29.54,45.29,50.95,42.00,46.77,32.38,23.05,17.48,
#     21.21,27.01,21.97,31.25,19.91,21.37,21.44,29.88,28.51,19.79,
#     26.70,34.08,23.10
# ]

# channel = [
#     45.51,25.81,28.09,30.14,27.88,25.61,30.65,24.91,25.65,32.18,
#     17.93,18.27,21.35,17.29,19.77,14.19,12.72,23.64,17.83,15.95,
#     27.56,25.69,21.08
# ]

# df = pd.DataFrame({
#     "Baseline": baseline,
#     "Codec": codec,
#     "Noise": noise,
#     "Reverb": reverb,
#     "Channel": channel
# }, index=models)

# df["Mean"] = df[["Codec", "Noise", "Reverb", "Channel"]].mean(axis=1)

# # Sort by baseline
# df = df.sort_values("Baseline")

# # -----------------------------
# # Plot
# # -----------------------------
# sns.set(style="white", font_scale=0.9)

# fig = plt.figure(figsize=(11, 9))
# gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 4.2, 1.2], wspace=0.1)

# # Baseline panel
# ax0 = fig.add_subplot(gs[0, 0])
# sns.heatmap(
#     df[["Baseline"]],
#     annot=True,
#     fmt=".1f",
#     cmap="Greys",
#     cbar=False,
#     linewidths=0.5,
#     ax=ax0,
#     yticklabels=df.index
# )
# ax0.set_title("Baseline", fontsize=11)
# ax0.set_xlabel("")
# ax0.set_ylabel("")
# ax0.tick_params(axis="y", labelsize=9, rotation=0)
# ax0.tick_params(axis="x", labelsize=9)

# # Main degradation heatmap
# ax1 = fig.add_subplot(gs[0, 1])
# sns.heatmap(
#     df[["Codec", "Noise", "Reverb", "Channel"]],
#     annot=True,
#     fmt=".1f",
#     cmap="YlGnBu",
#     cbar_kws={"label": "EER (%)"},
#     linewidths=0.5,
#     ax=ax1,
#     yticklabels=False   # <- important
# )
# ax1.set_title("Acoustic Degradations", fontsize=11)
# ax1.set_xlabel("")
# ax1.set_ylabel("")
# ax1.tick_params(axis="x", labelsize=9, rotation=0)

# # Mean panel
# ax2 = fig.add_subplot(gs[0, 2])
# sns.heatmap(
#     df[["Mean"]],
#     annot=True,
#     fmt=".1f",
#     cmap="Greys",
#     cbar=False,
#     linewidths=0.5,
#     ax=ax2,
#     yticklabels=False   # <- important
# )
# ax2.set_title("Mean", fontsize=11)
# ax2.set_xlabel("")
# ax2.set_ylabel("")
# ax2.tick_params(axis="x", labelsize=9)

# plt.suptitle("EER Across Acoustic Degradations", fontsize=14, y=0.98)
# plt.tight_layout(rect=[0, 0, 1, 0.97])
# plt.show()


########### EER heatmap code below, using absolute EER values and categorized by model type ###########
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # -----------------------------
# # Keep only models from taxonomy figure
# # and preserve category order
# # -----------------------------
# ordered_models = [
#     "FBANK",  # Baseline

#     # Generative
#     "APC",
#     "NPC",
#     "Mockingjay",
#     "TERA",
#     "DeCoAR 2.0",

#     # Discriminative
#     "wav2vec",
#     "wav2vec 2.0 Base",
#     "wav2vec 2.0 Large",
#     "HuBERT Base",
#     "HuBERT Large",
#     "MR-HuBERT",
#     "XLS-R",
#     "UniSpeech-SAT",
#     "Data2Vec",
#     "WAVLABLM",
#     "WavLM Large",

#     # Hybrid
#     "SSAST",
#     "MAE-AST-FRAME",
# ]

# # -----------------------------
# # Data from your absolute EER table
# # -----------------------------
# data = {
#     "FBANK":            [48.98, 49.61, 45.17, 51.20, 45.51],
#     "APC":              [29.06, 36.51, 14.16, 50.39, 25.81],
#     "NPC":              [34.96, 39.62, 21.29, 29.54, 28.09],
#     "Mockingjay":       [36.57, 41.05, 57.96, 45.29, 30.14],
#     "TERA":             [27.46, 39.12, 56.51, 46.77, 30.65],
#     "DeCoAR 2.0":       [27.18, 33.48,  9.93, 32.38, 24.91],

#     "wav2vec":          [27.00, 36.07, 23.03, 23.05, 25.65],
#     "wav2vec 2.0 Base": [16.47, 24.56, 26.82, 21.21, 17.93],
#     "wav2vec 2.0 Large":[11.25, 24.05, 30.17, 27.01, 18.27],
#     "HuBERT Base":      [21.63, 29.31, 12.46, 21.97, 21.35],
#     "HuBERT Large":     [18.48, 27.55,  4.83, 31.25, 17.29],
#     "MR-HuBERT":        [17.35, 27.83,  5.77, 19.91, 19.77],
#     "XLS-R":            [ 6.60, 20.27,  3.77, 21.37, 14.19],
#     "UniSpeech-SAT":    [11.21, 21.32,  2.32, 21.44, 12.72],
#     "Data2Vec":         [23.96, 30.91, 11.33, 29.88, 23.64],
#     "WAVLABLM":         [17.27, 26.39,  6.17, 28.51, 17.83],
#     "WavLM Large":      [13.56, 22.96,  4.71, 19.79, 15.95],

#     "SSAST":            [19.77, 35.74, 23.36, 26.70, 27.56],
#     "MAE-AST-FRAME":    [22.83, 31.46, 17.52, 23.10, 21.08],
# }

# columns = ["Baseline", "Codec", "Noise", "Reverb", "Channel"]
# df = pd.DataFrame.from_dict(data, orient="index", columns=columns)

# # Keep only requested models and exact order
# df = df.loc[ordered_models]

# # Mean over degradation columns only
# df["Mean"] = df[["Codec", "Noise", "Reverb", "Channel"]].mean(axis=1)

# # -----------------------------
# # Plot
# # -----------------------------
# sns.set(style="white", font_scale=0.9)

# fig = plt.figure(figsize=(12, 9))
# gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 4.3, 1.15], wspace=0.06)

# ax0 = fig.add_subplot(gs[0, 0])  # baseline
# ax1 = fig.add_subplot(gs[0, 1])  # degradation heatmap
# ax2 = fig.add_subplot(gs[0, 2])  # mean

# # Light grayscale for side panels
# side_cmap = sns.light_palette("gray", as_cmap=True)

# # Left: Baseline
# sns.heatmap(
#     df[["Baseline"]],
#     annot=True,
#     fmt=".1f",
#     cmap=side_cmap,
#     cbar=False,
#     linewidths=0.5,
#     linecolor="white",
#     ax=ax0,
#     yticklabels=df.index
# )

# # Middle: degradation heatmap
# sns.heatmap(
#     df[["Codec", "Noise", "Reverb", "Channel"]],
#     annot=True,
#     fmt=".1f",
#     cmap="YlGnBu",
#     cbar_kws={"label": "EER (%)"},
#     linewidths=0.5,
#     linecolor="white",
#     ax=ax1,
#     yticklabels=False
# )

# # Right: Mean
# sns.heatmap(
#     df[["Mean"]],
#     annot=True,
#     fmt=".1f",
#     cmap=side_cmap,
#     cbar=False,
#     linewidths=0.5,
#     linecolor="white",
#     ax=ax2,
#     yticklabels=False
# )

# # Titles
# ax0.set_title("Baseline", fontsize=11)
# ax1.set_title("Acoustic Degradations", fontsize=11)
# ax2.set_title("Mean", fontsize=11)

# # Clean axis labels
# for ax in [ax0, ax1, ax2]:
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.tick_params(axis="x", labelsize=9, rotation=0)

# ax0.tick_params(axis="y", labelsize=10, rotation=0)

# # -----------------------------
# # Category separators
# # -----------------------------
# # Row boundaries after:
# # FBANK = row 1
# # Generative ends after row 7
# # Discriminative ends after row 17
# separator_rows = [1, 7, 17]

# for ax in [ax0, ax1, ax2]:
#     for y in separator_rows:
#         ax.hlines(y, *ax.get_xlim(), colors="black", linewidth=1.5)

# plt.suptitle("EER Across Acoustic Degradations", fontsize=14, y=0.98)
# plt.tight_layout(rect=[0.06, 0, 1, 0.97])
# plt.show()

# plt.savefig("outputs/figures/acoustic_eer_heatmap_absolute_eer_categorized.png", dpi=300, bbox_inches="tight")

####### Relative EER Heatmap code below, using relative EER change from baseline and categorized by model type #######
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Ordered models (same as before)
# -----------------------------
ordered_models = [
    "FBANK",
    "APC","NPC","Mockingjay","TERA","DeCoAR 2.0",
    "wav2vec","wav2vec 2.0 Base","wav2vec 2.0 Large",
    "HuBERT Base","HuBERT Large","MR-HuBERT","XLS-R",
    "UniSpeech-SAT","Data2Vec","WAVLABLM","WavLM Large",
    "SSAST","MAE-AST-FRAME"
]

# -----------------------------
# Relative change data (%)
# Replace VQ-APC if needed
# -----------------------------
data = {
    "FBANK":            [-2.26],
    "APC":              [9.14],
    "NPC":              [-15.23],
    "Mockingjay":       [19.25],
    "TERA":             [57.55],
    "DeCoAR 2.0":       [-7.38],

    "wav2vec":          [-0.19],
    "wav2vec 2.0 Base": [37.40],
    "wav2vec 2.0 Large":[121.11],
    "HuBERT Base":      [-1.65],
    "HuBERT Large":     [9.47],
    "MR-HuBERT":        [5.59],
    "XLS-R":            [125.76],
    "UniSpeech-SAT":    [28.90],
    "Data2Vec":         [-0.08],
    "WAVLABLM":         [14.22],
    "WavLM Large":      [16.91],

    "SSAST":            [43.35],
    "MAE-AST-FRAME":    [2.01],
}

df_mean = pd.DataFrame.from_dict(data, orient="index", columns=["Mean"])
df_mean = df_mean.loc[ordered_models]

# -----------------------------
# Full relative change table
# -----------------------------
df = pd.DataFrame({
    "Codec": [
        1.29,25.64,13.33,12.25,42.46,23.18,
        33.59,49.12,113.78,35.51,49.08,60.4,207.12,
        90.19,29.01,52.81,69.32,80.78,37.8
    ],
    "Noise": [
        -7.78,-51.27,-39.1,58.49,105.79,-63.47,
        -14.7,62.84,168.18,-42.39,-73.86,-66.74,-42.88,
        -79.3,-52.71,-64.27,-65.27,18.16,-23.26
    ],
    "Reverb": [
        4.53,73.4,-15.5,23.84,70.32,19.13,
        -14.63,28.78,140.09,1.57,69.1,14.76,223.79,
        91.26,24.71,65.08,45.94,35.05,1.18
    ],
    "Channel": [
        -7.08,-11.18,-19.65,-17.58,11.62,-8.35,
        -5,8.86,62.4,-1.29,-6.44,13.95,115,
        13.47,-1.34,3.24,17.63,39.4,-7.67
    ]
}, index=ordered_models)

df["Mean"] = df_mean["Mean"]

# -----------------------------
# Plot
# -----------------------------
sns.set(style="white", font_scale=0.9)

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.2], wspace=0.05)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# ---- Main heatmap (relative change)
sns.heatmap(
    df[["Codec","Noise","Reverb","Channel"]],
    annot=True,
    fmt=".1f",
    cmap="RdBu_r",
    center=0,
    vmin=-100,
    vmax=150,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Relative Change (%)"},
    ax=ax1
)

ax1.set_title("Relative EER Change (Degradation Robustness)")
ax1.set_ylabel("")
ax1.set_xlabel("")
ax1.tick_params(axis="y", labelsize=10, rotation=0)

# ---- Mean column
sns.heatmap(
    df[["Mean"]],
    annot=True,
    fmt=".1f",
    cmap=sns.light_palette("gray", as_cmap=True),
    cbar=False,
    linewidths=0.5,
    linecolor="white",
    ax=ax2,
    yticklabels=False
)

ax2.set_title("Mean")

# -----------------------------
# Category separators
# -----------------------------
separator_rows = [1, 7, 17]

for ax in [ax1, ax2]:
    for y in separator_rows:
        ax.hlines(y, *ax.get_xlim(), colors="black", linewidth=1.5)

plt.suptitle("Relative EER Change Across Acoustic Degradations", fontsize=14)
plt.tight_layout()
plt.show()

plt.savefig("outputs/figures/acoustic_eer_heatmap_relative_eer_categorized.png", dpi=300, bbox_inches="tight")
