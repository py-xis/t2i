import numpy as np
from Levenshtein import distance as levenshtein_distance
from matplotlib import pyplot as plt
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 15,  # Set font size to 11pt
        "axes.labelsize": 15,  # -> axis labels
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)
plt.tight_layout(rect=[0, 0.03, 1, 0.85])

def plot_hist_levenshtein(levenshtein_distances, save_path = None):
    plt.figure(figsize=(10, 6))
    plt.hist(
        levenshtein_distances, bins=33, edgecolor="black", density=True
    )  # normalize the histogram
    plt.title("levenshtein distances")
    plt.xlabel("Levenshtein distance")
    plt.ylabel("Frequency")
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def get_levenshtein_distances(outs, targets):
    levs = []
    for output_text, target_text in zip(outs, targets):
        levs.append(levenshtein_distance(output_text.lower(), target_text.lower()))
    return np.array(levs)
