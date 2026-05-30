# %% [markdown]
# # Problem Statement:
# A leading logistics provider — LogiTech — operates across multiple regions with numerous warehouses, carriers, and product categories. The organization is experiencing inefficiencies in delivery reliability, warehouse utilization, inventory management, and cost control. The core problem is: how can operational data be analyzed to systematically identify bottlenecks and improve end-to-end logistics performanceme, Delivery_Mode, Weight, and Destination_Region?
#

# %%
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

# -----------------------------
# LOAD DATA
# -----------------------------
inventory = pd.read_excel("InventoryStockMovement.xlsx")
warehouse = pd.read_csv("WarehouseOperations.csv")
shipment = pd.read_excel("ShipmentDelivery.xlsx")

# -----------------------------
# OPTIONAL SAFE RENAME
# -----------------------------
warehouse.rename(
    columns={"Warehouse_Utilization_%": "Warehouse_Utilization_Pct"}, inplace=True
)

# -----------------------------
# CLEAN INVENTORY DATA
# -----------------------------
inventory["Stock_On_Hand"] = pd.to_numeric(inventory["Stock_On_Hand"], errors="coerce")
inventory["Reorder_Level"] = pd.to_numeric(inventory["Reorder_Level"], errors="coerce")
inventory["Reorder_Flag"] = pd.to_numeric(inventory["Reorder_Flag"], errors="coerce")
inventory["Avg_Lead_Time_Days"] = pd.to_numeric(
    inventory["Avg_Lead_Time_Days"], errors="coerce"
)
inventory["Stockout_Days"] = pd.to_numeric(inventory["Stockout_Days"], errors="coerce")
inventory["Carrying_Cost_Per_Unit"] = pd.to_numeric(
    inventory["Carrying_Cost_Per_Unit"], errors="coerce"
)

inventory["Reorder_Flag_Check"] = (
    inventory["Stock_On_Hand"] < inventory["Reorder_Level"]
).astype("Int64")

inventory["Reorder_Flag_Mismatch"] = (
    inventory["Reorder_Flag"] != inventory["Reorder_Flag_Check"]
)

inventory["Inventory_Duplicate_Flag"] = inventory.duplicated(
    subset=["Product_ID", "Warehouse_ID"], keep=False
)

inventory_numeric_cols = [
    "Stock_On_Hand",
    "Reorder_Level",
    "Avg_Lead_Time_Days",
    "Stockout_Days",
    "Carrying_Cost_Per_Unit",
]

for col in inventory_numeric_cols:
    q1 = inventory[col].quantile(0.25)
    q3 = inventory[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    inventory[f"{col}_Outlier"] = (inventory[col] < lower) | (inventory[col] > upper)

# -----------------------------
# CLEAN WAREHOUSE DATA
# -----------------------------
warehouse["Warehouse_Capacity_Units"] = pd.to_numeric(
    warehouse["Warehouse_Capacity_Units"], errors="coerce"
)
warehouse["Current_Inventory_Units"] = pd.to_numeric(
    warehouse["Current_Inventory_Units"], errors="coerce"
)
warehouse["Inventory_Turnover_Ratio"] = pd.to_numeric(
    warehouse["Inventory_Turnover_Ratio"], errors="coerce"
)
warehouse["Order_Fulfilment_Rate"] = pd.to_numeric(
    warehouse["Order_Fulfilment_Rate"], errors="coerce"
)
warehouse["Avg_Pick_Pack_Time_Min"] = pd.to_numeric(
    warehouse["Avg_Pick_Pack_Time_Min"], errors="coerce"
)
warehouse["Warehouse_Utilization_Pct"] = pd.to_numeric(
    warehouse["Warehouse_Utilization_Pct"], errors="coerce"
)
warehouse["Labour_Hours_Per_Day"] = pd.to_numeric(
    warehouse["Labour_Hours_Per_Day"], errors="coerce"
)
warehouse["Operational_Cost_Per_Day"] = pd.to_numeric(
    warehouse["Operational_Cost_Per_Day"], errors="coerce"
)
warehouse["Order_Fulfilment_Rate"].fillna(
    warehouse["Order_Fulfilment_Rate"].mean(), inplace=True
)

warehouse["Capacity_Check_Flag"] = (
    warehouse["Current_Inventory_Units"] > warehouse["Warehouse_Capacity_Units"]
)

warehouse_numeric_cols = [
    "Warehouse_Capacity_Units",
    "Current_Inventory_Units",
    "Inventory_Turnover_Ratio",
    "Order_Fulfilment_Rate",
    "Avg_Pick_Pack_Time_Min",
    "Warehouse_Utilization_Pct",
    "Labour_Hours_Per_Day",
    "Operational_Cost_Per_Day",
]

for col in warehouse_numeric_cols:
    q1 = warehouse[col].quantile(0.25)
    q3 = warehouse[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    warehouse[f"{col}_Outlier"] = (warehouse[col] < lower) | (warehouse[col] > upper)

# -----------------------------
# CLEAN SHIPMENT DATA
# -----------------------------
shipment["Dispatch_Date"] = pd.to_datetime(
    shipment["Dispatch_Date"], dayfirst=True, errors="coerce"
)
shipment["Delivery_Date"] = pd.to_datetime(
    shipment["Delivery_Date"], dayfirst=True, errors="coerce"
)

shipment["Delivery_Time_Days"] = pd.to_numeric(
    shipment["Delivery_Time_Days"], errors="coerce"
)
shipment["Weight"] = pd.to_numeric(shipment["Weight"], errors="coerce")
shipment["Shipment_Cost"] = pd.to_numeric(shipment["Shipment_Cost"], errors="coerce")
shipment["Delivery_Accuracy_Flag"] = pd.to_numeric(
    shipment["Delivery_Accuracy_Flag"], errors="coerce"
)
shipment["Damage_Flag"] = pd.to_numeric(shipment["Damage_Flag"], errors="coerce")

shipment["Delivery_Mode"] = shipment["Delivery_Mode"].replace(["", " "], np.nan)
shipment["Shipment_Status"] = shipment["Shipment_Status"].replace(["", " "], np.nan)
shipment.dropna(subset=["Delivery_Mode", "Shipment_Cost"], inplace=True)
shipment.drop_duplicates(inplace=True)

shipment["Computed_Delivery_Days"] = (
    shipment["Delivery_Date"] - shipment["Dispatch_Date"]
).dt.days

shipment["Delivery_Days_Mismatch"] = (
    shipment["Delivery_Time_Days"] != shipment["Computed_Delivery_Days"]
)

shipment["Shipment_ID_Duplicate_Flag"] = shipment.duplicated(
    subset=["Shipment_ID"], keep=False
)

shipment_numeric_cols = ["Delivery_Time_Days", "Weight", "Shipment_Cost"]

for col in shipment_numeric_cols:
    q1 = shipment[col].quantile(0.25)
    q3 = shipment[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    shipment[f"{col}_Outlier"] = (shipment[col] < lower) | (shipment[col] > upper)

# -----------------------------
# SAVE CLEANED FILES
# -----------------------------
inventory.to_csv("inventory_stock_movement_clean.csv", index=False)
warehouse.to_csv("warehouse_operations_clean.csv", index=False)
shipment.to_csv("shipment_delivery_clean.csv", index=False)

# -----------------------------
# STORE IN MYSQL
# -----------------------------
from sqlalchemy import create_engine

# change these values
user = "root"
password = "admin"
host = "localhost"
port = 3306
database = "logitech_phase1"

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

inventory.to_sql(
    "inventory_stock_movement_clean", con=engine, if_exists="replace", index=False
)
warehouse.to_sql(
    "warehouse_operations_clean", con=engine, if_exists="replace", index=False
)
shipment.to_sql("shipment_delivery_clean", con=engine, if_exists="replace", index=False)

print("\nCleaned files saved and MySQL database created successfully.")

# %%
