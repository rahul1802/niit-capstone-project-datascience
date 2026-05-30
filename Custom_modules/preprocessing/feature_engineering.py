import pandas as pd
import numpy as np

def create_delivery_features(df):
    # Calculate SLA threshold (e.g., 75th percentile)
    sla_threshold = df['delivery_time_days'].quantile(0.75)
    df['on_time_flag'] = (df['delivery_time_days'] <= sla_threshold).astype(int)
    df['delay_magnitude'] = (df['delivery_time_days'] - sla_threshold).clip(lower=0)
    df['route'] = df['warehouse_id'] + " to " + df['destination_region']
    return df

def create_cost_features(df):
    df['cost_per_kg'] = df['shipment_cost'] / (df['weight'] + 0.001)
    cost_threshold = df['shipment_cost'].quantile(0.80)
    df['high_cost_flag'] = (df['shipment_cost'] > cost_threshold).astype(int)
    return df

def create_warehouse_features(df):
    # Categorize utilization for easier reporting
    df['utilization_category'] = pd.cut(df['warehouse_utilization_pct'], 
                                        bins=[0, 60, 90, 1000], 
                                        labels=['Low', 'Optimal', 'Near Capacity'])
    # Efficiency metrics
    df['labor_productivity'] = df['current_inventory_units'] / (df['labour_hours_per_day'] + 0.1)
    df['cost_per_unit'] = df['operational_cost_per_day'] / (df['current_inventory_units'] + 0.1)
    df['log_operational_cost'] = np.log1p(df['operational_cost_per_day'])
    return df

def create_inventory_features(df):
    df['stockout_flag'] = (df['stockout_days'] > 0).astype(int)
    df['understock_flag'] = (df['stock_on_hand'] < df['reorder_level']).astype(int)
    # Ratio indicating replenishment priority
    df['reorder_urgency'] = df['reorder_level'] / (df['stock_on_hand'] + 1)
    return df

def feature_engineering_pipeline(shipment_df, warehouse_df, inventory_df):
    shipment_df = create_delivery_features(shipment_df)
    shipment_df = create_cost_features(shipment_df)
    warehouse_df = create_warehouse_features(warehouse_df)
    inventory_df = create_inventory_features(inventory_df)
    return shipment_df, warehouse_df, inventory_df