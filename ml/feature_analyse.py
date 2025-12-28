import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def feature_analyse():
    data_path = r"c:\Users\USTC\Desktop\monolayer\ML\kappa.csv"
    features = [
        "Density (g/cm3)",
        "kx (W/m/K)",
        "5-ring(%)",
        "6-ring(%)",
        "7-ring(%)",
        "8-ring(%)",
        "9-ring(%)",
        "avg_bond_length",
        "avg_bond_angle_deg",
        "var_bond_angle_deg2",
        "var_bond_length",
    ]

    df = pd.read_csv(data_path)
    df = df.dropna(subset=features)

    output_dir = os.path.join(os.path.dirname(__file__), "feature")
    os.makedirs(output_dir, exist_ok=True)

    folders = df["Folder Name"].dropna().unique()
    palette = [
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#c5b0d5",
        "#ff9896",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
        "#f7b6d2",
        "#c49c94",
    ]

    for col in features:
        series = df[col].dropna()
        if series.empty:
            continue
        plt.figure(figsize=(4, 6))

        values_list = []
        colors = []
        labels = []
        for idx, folder in enumerate(folders):
            vals = df[df["Folder Name"] == folder][col].dropna().values
            if vals.size == 0:
                continue
            values_list.append(vals)
            colors.append(palette[idx % len(palette)])
            labels.append(folder)

        if not values_list:
            plt.close()
            continue

        n = len(values_list)
        positions = np.linspace(1 - 0.15, 1 + 0.15, n)
        violin = plt.violinplot(
            values_list,
            positions=positions,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(violin["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.6)

        plt.boxplot(
            series.values,
            positions=[1],
            widths=0.08,
            vert=True,
            showfliers=False,
        )
        handles = [
            mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.6, label=l)
            for c, l in zip(colors, labels)
        ]
        if handles:
            plt.legend(handles=handles, loc="best", fontsize=8)
        plt.xticks([1], [col], rotation=45, ha="right")
        plt.ylabel(col)
        plt.tight_layout()
        safe_name = (
            col.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("%", "pct")
        )
        plt.savefig(
            os.path.join(output_dir, f"{safe_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    feature_analyse()


