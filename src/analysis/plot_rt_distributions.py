from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_rt_distributions(domain: str, raw_parquet_path: Path, cleaned_parquet_path: Path, figure_dir: Path):
    if not raw_parquet_path.exists() or not cleaned_parquet_path.exists():
        print(f"Skipping RT distribution plots for {domain}: missing parquet files.")
        return

    raw_df = pd.read_parquet(raw_parquet_path, columns=["rt_raw_ms"])
    clean_df = pd.read_parquet(cleaned_parquet_path, columns=["rt"])

    raw_rt = raw_df["rt_raw_ms"].dropna().to_numpy(dtype=float)
    clean_rt = clean_df["rt"].dropna().to_numpy(dtype=float)
    
    # 1. Pre-cleaning distribution
    plt.figure(figsize=(8, 5))
    # use log scale for raw because of extreme outliers (e.g. 43000ms+)
    bins = np.logspace(np.log10(max(1, raw_rt.min())), np.log10(max(10, raw_rt.max())), 100)
    plt.hist(np.clip(raw_rt, 1, None), bins=bins, color="gray", alpha=0.7, edgecolor="k")
    plt.xscale("log")
    plt.axvline(300, color="red", linestyle="--", label="Lower bound (300 ms)")
    plt.axvline(10000, color="red", linestyle="--", label="Upper bound (10000 ms)")
    plt.xlabel("Reaction Time (ms) [Log Scale]")
    plt.ylabel("Count")
    plt.title(f"{domain} Pre-cleaning RT Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / f"{domain.lower().replace('-', '_')}_rt_distribution.png", dpi=300)
    plt.close()

    # 2. Post-cleaning distribution
    plt.figure(figsize=(8, 5))
    # linear scale for cleaned since it's bounded [300, 10000]
    bins = np.linspace(300, 10000, 100)
    plt.hist(clean_rt, bins=bins, color="tab:green", alpha=0.8, edgecolor="k")
    plt.xlabel("Reaction Time (ms)")
    plt.ylabel("Count")
    plt.title(f"{domain} Post-cleaning RT Distribution")
    plt.tight_layout()
    plt.savefig(figure_dir / f"{domain.lower().replace('-', '_')}_rt_distribution_cleaned.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    plot_rt_distributions(
        domain="Age_IAT",
        raw_parquet_path=BASE_DIR / "data" / "processed" / "age_iat" / "age_iat_trials_standardized.parquet",
        cleaned_parquet_path=BASE_DIR / "data" / "processed" / "age_iat" / "age_iat_trials_standardized.parquet",
        figure_dir=BASE_DIR / "figures" / "age_iat"
    )
