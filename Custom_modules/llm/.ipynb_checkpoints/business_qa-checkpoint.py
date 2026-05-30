import pandas as pd
from llm.prompt_builder import build_business_prompt
from llm.groq_client import ask_groq

def q1_carriers_routes_highest_delay(shipment_df):
    summary = (
        shipment_df.groupby(["carrier_name", "route"])["on_time_flag"]
        .apply(lambda x: (1 - x.mean()) * 100)
        .reset_index(name="delay_rate_pct")
        .sort_values("delay_rate_pct", ascending=False)
        .head(10)
    )
    metrics_text = summary.to_string(index=False)

    prompt = build_business_prompt(
        objective="Delivery Performance: Identify delays and improve on-time delivery",
        question="Which carriers and routes have the highest delay rates?",
        metrics_text=metrics_text
    )
    return ask_groq(prompt), summary

def q2_shipment_otd_miss_pct(shipment_df):
    missed_pct = (1 - shipment_df["on_time_flag"].mean()) * 100
    metrics_text = f"Missed on-time delivery percentage: {missed_pct:.2f}%"

    prompt = build_business_prompt(
        objective="Delivery Performance: Identify delays and improve on-time delivery",
        question="What percentage of shipments miss on-time delivery targets?",
        metrics_text=metrics_text
    )
    return ask_groq(prompt), missed_pct