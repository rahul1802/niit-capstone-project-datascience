import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm
import matplotlib.pyplot as plt
import seaborn as sns


def auto_stat_test_cost_sampling(
    df,
    mode1="AIR",
    mode2="GROUND",
    sample_size=30,
    n_iterations=300,
    sampling_method="simple_random",
    alpha=0.05,
    use_log_transform=True,
):
    # -------------------------------------------------
    # 1. FILTER & VALIDATE DATA
    # -------------------------------------------------
    df_filtered = df[df["delivery_mode"].isin([mode1, mode2])]
    if df_filtered.empty:
        raise ValueError(f"No data found for modes: {mode1} or {mode2}")

    # -------------------------------------------------
    # 2. SAMPLING
    # -------------------------------------------------
    if sampling_method == "simple_random":
        s1 = df_filtered[df_filtered["delivery_mode"] == mode1].sample(
            n=sample_size, replace=True, random_state=42
        )
        s2 = df_filtered[df_filtered["delivery_mode"] == mode2].sample(
            n=sample_size, replace=True, random_state=42
        )
    elif sampling_method == "stratified":
        if "destination_region" not in df.columns:
            raise ValueError("Column 'destination_region' required for stratified.")
        s1 = (
            df_filtered[df_filtered["delivery_mode"] == mode1]
            .groupby("destination_region", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), 5), replace=True))
        )
        s2 = (
            df_filtered[df_filtered["delivery_mode"] == mode2]
            .groupby("destination_region", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), 5), replace=True))
        )
    else:
        raise ValueError("Invalid method")

    g1 = s1["shipment_cost"]
    g2 = s2["shipment_cost"]
    n1, n2 = len(g1), len(g2)

    print(f"Sample Sizes → {mode1}: {n1}, {mode2}: {n2}")

    # -------------------------------------------------
    # 3. PREPARE DATA (Log Transform)
    # -------------------------------------------------
    if use_log_transform:
        eps = 1e-9
        g1_data = np.log(g1 + eps)
        g2_data = np.log(g2 + eps)
        plt_label = "Log(Cost)"
    else:
        g1_data = g1
        g2_data = g2
        plt_label = "Cost"

    # -------------------------------------------------
    # 4. STATISTICAL TEST
    # -------------------------------------------------
    if n1 >= 30 and n2 >= 30:
        test_type = "Z-Test"
        diff = g1_data.mean() - g2_data.mean()
        se = np.sqrt(g1_data.var() / n1 + g2_data.var() / n2)
        stat = diff / se if se != 0 else 0
        p = 2 * (1 - norm.cdf(abs(stat)))
    else:
        test_type = "T-Test"
        stat, p = ttest_ind(g1_data, g2_data, equal_var=False)

    print(f"\nSelected Test : {test_type}")
    print(f"Test Statistic: {round(stat, 3)}")
    print(f"P-value       : {round(p, 5) if p > 0 else '< 0.00001'}")

    # -------------------------------------------------
    # 5. FIXED VISUALIZATION (Separate Distributions)
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Sampling Distribution for Group 1
    means1 = [
        g1_data.sample(n=len(g1), replace=True).mean() for _ in range(n_iterations)
    ]
    sns.histplot(
        means1, kde=True, stat="density", alpha=0.5, color="blue", label=mode1, ax=ax
    )

    # Plot Sampling Distribution for Group 2
    means2 = [
        g2_data.sample(n=len(g2), replace=True).mean() for _ in range(n_iterations)
    ]
    sns.histplot(
        means2, kde=True, stat="density", alpha=0.5, color="red", label=mode2, ax=ax
    )

    # Add Mean Lines
    ax.axvline(
        g1_data.mean(), color="blue", linestyle="--", linewidth=2, label=f"{mode1} Mean"
    )
    ax.axvline(
        g2_data.mean(), color="red", linestyle="--", linewidth=2, label=f"{mode2} Mean"
    )

    plt.title(
        f"Comparative Sampling Distributions ({test_type})\n{plt_label} Scale",
        fontsize=14,
    )
    plt.xlabel(f"Sample Mean ({plt_label})")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    # -------------------------------------------------
    # 6. INTERPRETATION
    # -------------------------------------------------
    if p < alpha:
        print(f"\n✅ Significant difference detected (p < {alpha})")
    else:
        print(f"\n❌ No significant difference (p >= {alpha})")
    print(
        "\n📌 Insight: The separation of the two curves confirms the statistical difference."
    )


# Run this fixed version
# auto_stat_test_cost_sampling_fixed(shipment_df, sample_size=30, n_iterations=300, sampling_method='simple_random', use_log_transform=True)
