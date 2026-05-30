# %% [markdown]
# # 🚚 Logistics & Supply Chain Analytics
# ### Course 7 — Final Project

# %%
import pandas as pd
import numpy as np
from Custom_modules.preprocessing.Clean_data import normalize_columns
from Custom_modules.preprocessing.Clean_data import cap_outliers_iqr
from Custom_modules.preprocessing.Clean_data import (
    convert_date_columns,
    clean_text_columns,
)
import warnings

warnings.filterwarnings("ignore")

# %%
# Load dataset
warehouse_df = pd.read_csv("data/warehouse_df.csv")
inventory_df = pd.read_csv("data/inventory_df.csv")
shipment_df = pd.read_csv("data/shipment_df.csv")
warehouse_df = normalize_columns(warehouse_df)
inventory_df = normalize_columns(inventory_df)
shipment_df = normalize_columns(shipment_df)

print("Warehouse shape :", warehouse_df.shape)
print("Inventory shape :", inventory_df.shape)
print("Shipment shape  :", shipment_df.shape)

# %% [markdown]
# ### Data Preprocessing

# %%
# finding and imputing missing values
warehouse_df.isnull().sum()

# %%
inventory_df.isnull().sum()

# %%
shipment_df.isnull().sum()

# %%
# finding and removing duplicates
warehouse_df.duplicated().sum()

# %%
inventory_df.duplicated().sum()

# %%
shipment_df.duplicated().sum()

# %%
# removing the leading and trailing spaces from the object columns in  each dataset
warehouse_df = clean_text_columns(warehouse_df)

# %%
inventory_df = clean_text_columns(inventory_df)

# %%
shipment_df = clean_text_columns(shipment_df)

# %%
# converting the string type to date  types for some columns in shipment dataset
data_cols = ["dispatch_date", "delivery_date"]
shipment_df = convert_date_columns(shipment_df, data_cols)

# %% [markdown]
# ### Feature Engineering

# %%
import Custom_modules.preprocessing.feature_engineering as fe

# %%
shipment_df

# %%
shipment_df = fe.create_delivery_features(shipment_df)
shipment_df.head()

# %%
shipment_df = fe.create_cost_features(shipment_df)
shipment_df.head()

# %%
warehouse_df.head()

# %%
warehouse_df = fe.create_warehouse_features(warehouse_df)
warehouse_df.head()

# %%
inventory_df.head()

# %%
inventory_df = fe.create_inventory_features(inventory_df)
inventory_df.head()

# %% [markdown]
# ## Statistics

# %% [markdown]
# ### Hypothesis Test (Z test/T_test)

# %%
from Custom_modules.statistical.hypothesis_test import auto_stat_test_cost_sampling

# %%
auto_stat_test_cost_sampling(
    shipment_df, sample_size=30, n_iterations=100, sampling_method="stratified"
)

# %%
auto_stat_test_cost_sampling(
    shipment_df, sample_size=20, n_iterations=100, sampling_method="simple_random"
)

# %% [markdown]
# ### Hypothesis Test (Chi Square Test)

# %%
from Custom_modules.statistical.chi_square import carrier_delivery_chi_square_test

carrier_delivery_chi_square_test(shipment_df)

# %% [markdown]
# ## Implementing ML Model

# %% [markdown]
# ### Feature Selection for implementing Supervised ML Model

# %%
from Custom_modules.visualization.plots import perform_feature_selection_eda

# %%
df = shipment_df.copy()
perform_feature_selection_eda(df)

# %% [markdown]
# ## Implementing Regression Models

# %%
### Importing libaries for implementing and evaluating regression models
from Custom_modules.models.linear_regression import run_linear_regression
from Custom_modules.models.random_forest_regression import run_random_forest
from Custom_modules.models.evaluate_model import evaluate_regression
from Custom_modules.visualization.plots import plot_model_results

# %% [markdown]
# ### Linear Regression Model

# %%
model, X_test, y_test = run_linear_regression(df)

# %%
### evaluating linear model and plotting the model result
y_lin_pred = evaluate_regression(model, X_test, y_test)

# %%
# plotting the results for Linear Regression Model
plot_model_results(y_test, y_lin_pred, "Linear Regression")

# %% [markdown]
# ### Random Forest Regression Model

# %%
model_rf, X_test_rf, y_test_rf = run_random_forest(df)

# %%
### evaluating random forest regression  model and plotting the model result
y_rf_pred = evaluate_regression(model_rf, X_test_rf, y_test_rf)

# %%
### plotting the results for random forest regression Model
plot_model_results(y_test_rf, y_rf_pred, "Random Forest Regression")

# %% [markdown]
# ### Implementing the random forest tree

# %%
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# %%
# Extract the first tree estimator
single_tree = model_rf.estimators_[0]

# Plot the tree structure
plt.figure(figsize=(20, 10))  # Set a large figure size so text is readable
plot_tree(
    single_tree,
    feature_names=["weight", "carrier_enc"],  # The features you trained X on
    max_depth=3,  # Limits depth so the plot isn't chaotic
    filled=True,  # Colors the nodes by their predicted values
    rounded=True,  # Makes the node boxes rounded
    fontsize=10,
)
plt.title(
    "Visualizing a Single Decision Tree from the Random Forest",
    fontsize=16,
    fontweight="bold",
)
# plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Implementing Classification  Models

# %%
### Importing libaries for implementing and evaluating classification Models
from Custom_modules.models.logistic_regression import run_logistic_regression
from Custom_modules.models.Knn_Classifier import run_knn
from Custom_modules.models.evaluate_model import evaluate_classification
from Custom_modules.visualization.plots import plot_classification_results

# %% [markdown]
# ### Logistic Regression Model

# %%
model_log, X_test_log, y_test_log = run_logistic_regression(df)

# %%
### evaluating logistics regression classification  model and plotting the model result
cm = evaluate_classification(model_log, X_test_log, y_test_log)

# %%
### plotting the results for logistics  regression Model
plot_classification_results(cm, "Logistic Regression")

# %% [markdown]
# ### K Nearest Classifier Model

# %%
# Find the optimal value of K
from Custom_modules.models.Knn_Classifier import find_optimal_k, run_knn

find_optimal_k(df)

# %%
# therefore we choose the k value as 2
model_knn, X_test_knn, y_test_knn = run_knn(df, 2)

# %%
### evaluating knnclassifier  model and plotting the model result
cm = evaluate_classification(model_knn, X_test_knn, y_test_knn)

# %%
### plotting the results for Knn classifier Model
plot_classification_results(cm, "Knn Classifier")

# %% [markdown]
# ## Implementing Unsupervised Algorithm

# %% [markdown]
# # K Mean's Algorithm

# %%
### Importing libaries for implementing and evaluating Kmeans Models
from Custom_modules.preprocessing.Clean_data import cap_outliers_iqr
from Custom_modules.visualization.plots import perform_feature_selection_eda_Kmeans
from Custom_modules.visualization.plots import (
    plot_Kmeans_cluster_scaled,
    plot_Kmeans_cluster_original,
)
from Custom_modules.models.Kmeans import (
    run_KMeans,
    find_optimal_K_elbowmethod,
    find_optimal_K_silhouette,
)
from Custom_modules.models.evaluate_model import evaluateKmeans

# %%
numerical_ware_df = warehouse_df.select_dtypes(include="number")

# %%
numerical_ware_df.head()

# %%
# removing outliers in each numerical column
cleaned_ware_num_df = cap_outliers_iqr(numerical_ware_df)

# %%
cleaned_ware_num_df.head()

# %%
perform_feature_selection_eda_Kmeans(cleaned_ware_num_df)

# %% [markdown]
# #### Randomly initalizing the n.o of clusters to test the model performance

# %%
features_selected = [
    "operational_cost_per_day",
    "warehouse_capacity_units",
    "labor_productivity",
]
Kmeans_5, X_scaled = run_KMeans(cleaned_ware_num_df, features_selected, 5)

# %%
# evaluating the performance with k=5
labels_5 = Kmeans_5.labels_
evaluateKmeans(X_scaled, labels_5)

# %%
ware_num_df = cleaned_ware_num_df.copy()
ware_num_df["cluster_num_5"] = labels_5

# %%
ware_num_df["cluster_num_5"].value_counts()

# %%
X = ware_num_df[
    ["operational_cost_per_day", "warehouse_capacity_units", "labor_productivity"]
]
labels = ware_num_df["cluster_num_5"]
features_names = X.copy()
centers = Kmeans_5.cluster_centers_

# %% [markdown]
# #### Plotting cluster graph For K=5

# %%
# scaled Feature
features_names = [
    "operational_cost_per_day",
    "warehouse_capacity_units",
    "labor_productivity",
]
plot_Kmeans_cluster_scaled(X_scaled, labels, centers, features_names, y_feature_index=1)

# %% [markdown]
# #### Finding the optimal value of K for representing the number of clusters

# %%
find_optimal_K_elbowmethod(X_scaled)

# %%
find_optimal_K_silhouette(X_scaled)

# %% [markdown]
# <pre><b>Observation:</b>Here both graph are showing different optimal k value in which 2 is showing as the optimal value in silhoutte score graph and 3 is showing the optimal k value in elobw method  but we need a safe side to select the k value so we go for the option 3 as the valid/optimal K value
# <pre>

# %% [markdown]
# #### Plotting cluster graph on  Opeartional Cost vs Labour Productivity

# %% [markdown]
# #### implementing the K means model for K=3

# %%
# plotiing cluster's for k=3
kmeans_3, Xscaled = run_KMeans(cleaned_ware_num_df, features_selected, 3)
labels_3 = kmeans_3.labels_
evaluateKmeans(X_scaled, labels_3)

# %%
ware_num_df["cluster_num_3"] = labels_3
ware_num_df["cluster_num_3"].value_counts()

# %%
X = ware_num_df[
    ["operational_cost_per_day", "warehouse_capacity_units", "labor_productivity"]
]
labels = ware_num_df["cluster_num_3"]
features_names = X.copy()

# %% [markdown]
# #### Plotting cluster graph For K=3

# %%
# without scaled feature
plot_Kmeans_cluster_original(X, labels, features_names, y_feature_index=2)

# %%
# with scaled features
# features_names=['operational_cost_per_day','labour_hours_per_day','warehouse_capacity_units']
# plot_Kmeans_cluster_scaled(X_scaled, labels,centers,features_names,y_feature_index=2)

# %%
from Custom_modules.preprocessing.feature_engineering import (
    feature_engineering_pipeline,
)
from Custom_modules.llm.business_qa import q1_carriers_routes_highest_delay_2

shipment_df, warehouse_df, inventory_df = feature_engineering_pipeline(
    shipment_df, warehouse_df, inventory_df
)

answer1, table1 = q1_carriers_routes_highest_delay_2(warehouse_df)
print("Q1 Answer:\n", answer1)
print(table1)
