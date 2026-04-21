import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("223251_curve.csv")

df_pooled = (
    df
    .groupby("bin", as_index=False)
    .agg(
        x=("x_mean", "mean"),
        y=("rt_mean", "mean")
    )
)

plt.figure()
plt.plot(df_pooled["x"], df_pooled["y"], "o-")
plt.xlabel("Within-block position (x)")
plt.ylabel("Mean RT (ms)")
plt.title("Pooled IAT adaptation curve")
plt.show()
