from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    BASE_DIR = Path(__file__).resolve().parents[2]
    out_path = BASE_DIR / "figures" / "domain_theta_contrast_social_vs_self.png"
    
    # Values derived from outputs (display-order encoding for Life-Satisfaction)
    data = [
        {"domain": "Gender-Science", "mean": 17.29, "hdi_low": 16.5, "hdi_high": 18.25, "type": "social"},
        {"domain": "Age", "mean": 18.48, "hdi_low": 17.5, "hdi_high": 19.25, "type": "social"},
        {"domain": "Sexuality", "mean": 19.12, "hdi_low": 18.25, "hdi_high": 20.0, "type": "social"},
        {"domain": "Life-Satisfaction\n(display order)", "mean": 28.49, "hdi_low": 8.50, "hdi_high": 53.75, "type": "self"},
    ]
    df = pd.DataFrame(data)
    
    # Reverse so top is first
    df = df.iloc[::-1].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    
    colors = {"social": "#1f77b4", "self": "#ff7f0e"}
    
    y_pos = np.arange(len(df))
    
    for i, row in df.iterrows():
        color = colors[row["type"]]
        ax.plot([row["hdi_low"], row["hdi_high"]], [y_pos[i], y_pos[i]], color=color, lw=3, zorder=1)
        ax.scatter([row["mean"]], [y_pos[i]], color=color, s=80, zorder=2, edgecolor='k', alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["domain"], fontsize=11)
    
    all_hi = max(df["hdi_high"].max(), df["mean"].max())
    ax.set_xlim(0, all_hi + 10)
    ax.set_xlabel(r"$\theta$ Posterior Mean & 94% HDI (degrees)", fontsize=11)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Public Social-Category',
               markerfacecolor='#1f77b4', markersize=9, markeredgecolor='k', lw=3),
        Line2D([0], [0], marker='o', color='w', label='Self-Referential (display order)',
               markerfacecolor='#ff7f0e', markersize=9, markeredgecolor='k', lw=3),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, fontsize=10)
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, zorder=0)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
