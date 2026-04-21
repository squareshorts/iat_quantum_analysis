import pandas as pd

df = pd.read_csv("223251_curve.csv")

# pool blocks
df_pooled = (
    df
    .groupby("bin", as_index=False)
    .agg(
        x=("x_mean", "mean"),
        y=("rt_mean", "mean")
    )
)

print(df_pooled)
