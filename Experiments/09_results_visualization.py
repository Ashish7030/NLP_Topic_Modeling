import os
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

methods = ["NMF", "BERT (Doc)", "BERT (Token)", "LDA"]

results = {
    "LLM Topic Quality": [4.80, 4.35, 4.35, 4.20],
    "C_v Coherence": [0.6292, 0.5246, 0.4701, 0.4274],
    "Topic Diversity": [0.8767, 0.8000, 0.9867, 0.7733],
    "Topic Overlap": [0.0077, 0.0155, 0.0007, 0.0243],
}


def create_horizontal_bar(
    values,
    title,
    xlabel,
    filename,
    xlim,
    decimals
):
    df = pd.DataFrame({
        "Method": methods,
        "Score": values
    })

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    bars = ax.barh(
        df["Method"],
        df["Score"]
    )

    ax.invert_yaxis()

    for bar, value in zip(bars, df["Score"]):
        ax.text(
            value + (xlim[1] - xlim[0]) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.{decimals}f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title(
        title,
        fontsize=12,
        pad=12
    )

    ax.set_xlabel(
        xlabel,
        fontsize=10
    )

    ax.set_ylabel(
        "Topic Modeling Method",
        fontsize=10
    )

    ax.set_xlim(xlim)

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.5
    )

    ax.set_axisbelow(True)

    plt.tight_layout()

    path = os.path.join(FIG_DIR, filename)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


create_horizontal_bar(
    results["LLM Topic Quality"],
    "LLM-Based Topic Quality Evaluation",
    "Mean Score (1.0 - 5.0 Scale)",
    "fig_llm_topic_quality.png",
    (0, 5.5),
    2
)

create_horizontal_bar(
    results["C_v Coherence"],
    "Topic Coherence Across Methods",
    "C_v Score",
    "fig_topic_coherence.png",
    (0, 0.7),
    4
)

create_horizontal_bar(
    results["Topic Diversity"],
    "Topic Diversity Across Methods",
    "Diversity Score",
    "fig_topic_diversity.png",
    (0, 1.05),
    4
)

create_horizontal_bar(
    results["Topic Overlap"],
    "Topic Overlap Across Methods",
    "Overlap Score",
    "fig_topic_overlap.png",
    (0, 0.027),
    4
)

print("Figures created successfully.")