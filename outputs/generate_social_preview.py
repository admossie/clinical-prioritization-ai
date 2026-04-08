from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "social_preview.png"


def main() -> None:
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("#f6f8fc")
    ax.set_facecolor("#f6f8fc")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0.78), 1, 0.22, color="#24314f"))
    ax.add_patch(
        FancyBboxPatch(
            (0.055, 0.16),
            0.58,
            0.47,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor="#ffffff",
            edgecolor="#d9e0ef",
            linewidth=1.2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.69, 0.18),
            0.23,
            0.16,
            boxstyle="round,pad=0.014,rounding_size=0.03",
            facecolor="#e8f6ee",
            edgecolor="#b8ddc7",
            linewidth=1.0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.69, 0.39),
            0.23,
            0.16,
            boxstyle="round,pad=0.014,rounding_size=0.03",
            facecolor="#fff3dd",
            edgecolor="#e5cf95",
            linewidth=1.0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.69, 0.60),
            0.23,
            0.16,
            boxstyle="round,pad=0.014,rounding_size=0.03",
            facecolor="#fde7e8",
            edgecolor="#e2b2b5",
            linewidth=1.0,
        )
    )

    ax.text(
        0.055,
        0.89,
        "Clinical Prioritization AI",
        fontsize=28,
        fontweight="bold",
        color="#ffffff",
    )
    ax.text(
        0.055,
        0.82,
        "Capacity-aware readmission prioritization for clinical operations",
        fontsize=13,
        color="#dbe3f4",
    )

    ax.text(
        0.085,
        0.56,
        "From risk scores to actionable queues",
        fontsize=22,
        fontweight="bold",
        color="#24314f",
    )
    ax.text(
        0.085,
        0.49,
        "Calibrated ML, fairness checks, workflow simulation,",
        fontsize=14,
        color="#4f5d78",
    )
    ax.text(
        0.085,
        0.445,
        "explainability, and a Streamlit triage dashboard.",
        fontsize=14,
        color="#4f5d78",
    )

    ax.text(
        0.085, 0.33, "Model outputs", fontsize=13, fontweight="bold", color="#24314f"
    )
    ax.text(0.085, 0.275, "- Percentile-based risk tiers", fontsize=13, color="#24314f")
    ax.text(
        0.085, 0.225, "- Queue capture and ROI metrics", fontsize=13, color="#24314f"
    )
    ax.text(
        0.085, 0.175, "- CSV queue export for operations", fontsize=13, color="#24314f"
    )

    ax.text(0.725, 0.69, "High tier", fontsize=18, fontweight="bold", color="#8d2130")
    ax.text(0.725, 0.63, "Focused intervention queue", fontsize=12, color="#7a4a4f")

    ax.text(0.725, 0.48, "Medium tier", fontsize=18, fontweight="bold", color="#946200")
    ax.text(0.725, 0.42, "Monitor or step-up resources", fontsize=12, color="#7a6640")

    ax.text(0.725, 0.27, "Low tier", fontsize=18, fontweight="bold", color="#147a43")
    ax.text(
        0.725,
        0.21,
        "Lower-priority under current capacity",
        fontsize=12,
        color="#4e705b",
    )

    ax.text(0.055, 0.07, "GitHub release: v1.0.0", fontsize=11, color="#6e7b93")
    ax.text(
        0.82,
        0.07,
        "admossie/clinical-prioritization-ai",
        fontsize=11,
        color="#6e7b93",
        ha="right",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
