from scipy.stats import chi2_contingency
import pandas as pd


def carrier_delivery_chi_square_test(df, alpha=0.05):

    # -------------------------------------------------
    # CONTINGENCY TABLE
    # -------------------------------------------------

    table = pd.crosstab(df["carrier_name"], df["on_time_flag"])

    print("Contingency Table:\\n")
    display(table)

    # -------------------------------------------------
    # CHI-SQUARE TEST
    # -------------------------------------------------

    chi2, p, dof, expected = chi2_contingency(table)

    print("\nChi-Square Statistic :", round(chi2, 3))
    print("P-value              :", round(p, 5))
    print("Degrees of Freedom   :", dof)

    # -------------------------------------------------
    # HYPOTHESIS RESULT
    # -------------------------------------------------

    if p < alpha:
        print("\n✅ Carrier significantly impacts delivery performance")
    else:
        print(
            "\n❌ No significant relationship between carrier and delivery performance"
        )

    # -------------------------------------------------
    # BUSINESS ANALYSIS
    # -------------------------------------------------

    performance = (
        df.groupby("carrier_name")["on_time_flag"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )

    print("\nCarrier Performance (%):\n")
    display(performance)

    # -------------------------------------------------
    # BUSINESS INSIGHT
    # -------------------------------------------------

    print(""""📌 Business Insight:
    Carriers with higher delayed shipment percentages
    should be reviewed for route optimization,
    SLA violations, or operational inefficiencies.
    """)
