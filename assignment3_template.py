"""CSE 242 Assignment 3, Question 1: hard-margin linear SVM (30 points).
"""

import argparse
from pathlib import Path
import math


SVM_X = ((1, 1), (1, 2), (2, 1), (0, 0), (1, 0), (0, 1))
SVM_Y = (1, 1, 1, -1, -1, -1)


def plot_svm_data(output_path):
    """Generate an annotated scatter plot to help inspect Q1(a) by eye.

    Point IDs are 1-based and match the source table order. This helper
    does not identify support vectors or draw the separating boundary.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib could not be imported. Use the README's clean virtual "
            "environment and install requirements.txt."
        ) from exc

    fig, ax = plt.subplots(figsize=(6, 6))
    # Draw each class separately so it gets its own color, marker and legend entry.
    for label, color, marker in ((1, "tab:blue", "o"), (-1, "tab:orange", "s")):
        points = [x for x, y in zip(SVM_X, SVM_Y) if y == label]
        ax.scatter(
            [x[0] for x in points], [x[1] for x in points],
            color=color, marker=marker, s=90, label=f"y = {label:+d}",
        )
    # Label each point with its 1-based ID so it can be referred to in the report.
    for point_id, (x1, x2) in enumerate(SVM_X, start=1):
        ax.annotate(str(point_id), (x1, x2), xytext=(7, 7),
                    textcoords="offset points")
    # Fixed axis limits and equal scaling keep distances (and the margin) visually true.
    ax.set(xlabel="x1", ylabel="x2", title="Assignment 3: SVM training data",
           xlim=(-0.5, 2.5), ylim=(-0.5, 2.5))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    # Create the output folder if needed, then save and free the figure.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def identify_support_vectors():
    """Q1(a), 10 points together with the plot: identify support vectors by eye.

    Return their 1-based point IDs and explain your choice in the report.
    """
    # Read off the plot: class +1 is the L-shaped group {1, 2, 3} and class -1
    # is {4, 5, 6}. The closest opposite-class pairs are point 1 (1,1) against
    # points 5 (1,0) and 6 (0,1), so these three touch the margin.
    # Points 2 and 3 are farther from the gap, and point 4 (0,0) is hidden
    # behind points 5 and 6, so none of them is a support vector.
    return (1, 5, 6)


def derive_svm_separator():
    """Q1(b), 10 points for the boundary and 10 for the geometric margin.

    Use a linear, hard-margin SVM. The decision boundary is w dot x + b = 0;
    retain the intercept b. Use the support vectors to derive it.

    Return (weights, bias, geometric_margin), with weights a pair of numbers.
    Explain the derivation and your margin convention in the written report.
    """
    # Support vectors lie exactly on the margins, i.e. y * (w . x + b) = 1:
    #   point 1 (1,1), y=+1:  w1 + w2 + b =  1
    #   point 5 (1,0), y=-1:  w1      + b = -1
    #   point 6 (0,1), y=-1:       w2 + b = -1
    # Subtracting the last two gives w1 = w2 = w. Then 2w + b = 1 and
    # w + b = -1 give w = 2, b = -3, i.e. the line 2*x1 + 2*x2 - 3 = 0.
    weights = (2.0, 2.0)
    bias = -3.0
    # Geometric margin = distance from the boundary to the closest points
    # = 1 / ||w|| = 1 / (2*sqrt(2)) ~ 0.3536 (the full street width is 2 / ||w||).
    geometric_margin = 1.0 / math.hypot(*weights)
    return weights, bias, geometric_margin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plot", "svm"),
                        default="plot", help="Default: generate the SVM data plot.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "svm_data.png",
                        help="Output image path for --task plot.")
    args = parser.parse_args()
    # "plot" only draws the data; "svm" prints the answers to Q1(a) and Q1(b).
    if args.task == "plot":
        print(f"Saved plot to {plot_svm_data(args.output)}")
    else:
        print("Support vector point IDs:", identify_support_vectors())
        weights, bias, margin = derive_svm_separator()
        print(f"Weights: {weights}; bias: {bias}; geometric margin: {margin}")


if __name__ == "__main__":
    main()
