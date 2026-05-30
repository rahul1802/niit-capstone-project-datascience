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
        metrics_text=metrics_text,
    )
    return ask_groq(prompt), summary


def q2_shipment_otd_miss_pct(shipment_df):
    missed_pct = (1 - shipment_df["on_time_flag"].mean()) * 100
    metrics_text = f"Missed on-time delivery percentage: {missed_pct:.2f}%"

    prompt = build_business_prompt(
        objective="Delivery Performance: Identify delays and improve on-time delivery",
        question="What percentage of shipments miss on-time delivery targets?",
        metrics_text=metrics_text,
    )
    return ask_groq(prompt), missed_pct


def q1_carriers_routes_highest_delay_2(warehouse_df):
    # 1. Identify the specific plants mentioned in the question
    # Adjust 'PLANT03' / 'PLANT01' to match your actual ID format (e.g., 'Plant 3')
    target_plants = ["PLANT03", "PLANT01"]
    
    # 2. Prepare the data
    # We need to calculate metrics and ensure we have Lead Time if it exists
    agg_dict = {"log_operational_cost": "mean"}
    
    # Check if 'lead_time' or similar column exists in your dataframe
    if "lead_time" in warehouse_df.columns:
        agg_dict["lead_time"] = "mean"
    elif "delivery_delay" in warehouse_df.columns:
        agg_dict["lead_time"] = "mean" # Map delay to lead_time for the prompt
    else:
        # Fallback if no lead time column exists
        pass 

    # Group by warehouse
    summary = (
        warehouse_df.groupby(["warehouse_id", "warehouse_utilization_pct"])
        .agg(agg_dict)
        .reset_index()
    )

    # 3. Filter for the specific target plants (Plant 3 & Plant 1)
    # If they exist in the data, we want them in the context
    target_data = summary[summary["warehouse_id"].isin(target_plants)]
    
    # 4. Also get the "Worst Performers" to give the LLM context on high costs/delays
    # Sort by Operational Cost (Highest first) to find the most problematic plants
    worst_performers = summary.sort_values("log_operational_cost", ascending=False).head(5)
    
    # Combine: Specific plants + Top 5 worst overall context
    # This ensures the LLM sees the specific problem plants AND the broader trend
    if not target_data.empty:
        context_data = pd.concat([target_data, worst_performers]).drop_duplicates().reset_index(drop=True)
    else:
        # Fallback if target plants not found, just show worst performers
        context_data = worst_performers

    # Sort the final display by Cost (Highest first) so the "problem" is obvious
    context_data = context_data.sort_values("log_operational_cost", ascending=False)

    # Format for the prompt
    metrics_text = context_data.to_string(index=False)

    prompt = build_business_prompt(
        objective="Identify delays and improve on-time delivery; the overall objective is to reduce operational cost.",
        question="Why is the Lead time high in Plant 3 and Plant 1, and how does it affect the operational cost? Return the output in table or bullet points.",
        metrics_text=metrics_text,
    )
    
    return ask_groq(prompt), context_data
