from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

test_set = True

ratio_str = ["1_0", "0_5", "0_25", "0_1", "0_05", "0_01", "0_005", "0_002"]
if test_set:
    cnn_files = [Path("outputs", "cnn_test_" + ratio + ".nc") for ratio in ratio_str]
    glp_files = [Path("outputs", "glp_test_" + ratio + ".nc") for ratio in ratio_str]
    aug_files = [Path("outputs", "aug_test_" + ratio + ".nc") for ratio in ratio_str]
else:
    cnn_files = [
        Path("outputs", "cnn_validation_" + ratio + ".nc") for ratio in ratio_str
    ]
    glp_files = [
        Path("outputs", "glp_validation_" + ratio + ".nc") for ratio in ratio_str
    ]
    aug_files = [
        Path("outputs", "aug_validation_" + ratio + ".nc") for ratio in ratio_str
    ]


def accuracy_per_file(files):
    """Function that computes accuracy per file, returns list of accuracies"""

    accuracies = []
    for fn in files:
        ds = xr.open_dataset(fn)
        mean_preds = ds["predictions"].mean(dim="class")
        true_positive_count = (
            np.logical_and(ds["labels"] == 1, mean_preds.round() == 1).sum().item()
        )
        true_negative_count = (
            np.logical_and(ds["labels"] == 0, mean_preds.round() == 0).sum().item()
        )
        accuracy = (
            (true_positive_count + true_negative_count) * 100 / ds.sizes["example"]
        )
        accuracies.append(accuracy)
    return accuracies


cnn_accuracies = accuracy_per_file(cnn_files)
glp_accuracies = accuracy_per_file(glp_files)
aug_accuracies = accuracy_per_file(aug_files)

print("\nAccuracies")
print("CNN")
print(" & ".join([f"{acc:.1f}" for acc in cnn_accuracies]))
print("Augmented CNN")
print(" & ".join([f"{acc:.1f}" for acc in aug_accuracies]))
print("GLNN")
print(" & ".join([f"{acc:.1f}" for acc in glp_accuracies]))

ratios = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.002]

F, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_xscale("log")
ax.plot(ratios, cnn_accuracies, label="CNN")
ax.plot(ratios, glp_accuracies, label="GLP")
ax.plot(ratios, aug_accuracies, label="Aug. CNN")
ax.xaxis.set_inverted(True)
ax.set_xticks(ratios, labels=[str(int(8000 * ratio)) for ratio in ratios])
ax.legend()
ax.set_xlabel("Training set size (log scale)")
ax.set_ylabel("Accuracy")

if test_set:
    F.savefig("test_accuracy_curve.png", bbox_inches="tight", dpi=300)
else:
    F.savefig("accuracy_curve.png", bbox_inches="tight", dpi=300)
