# supply_chain_project/visualization/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

def perform_feature_selection_eda(df):
    """
    EDA to justify feature selection for Supervised Learning.
    Objectives: Delivery Performance & Cost Efficiency.
    """
    # 1. Heatmap: Justifying features for Cost Regression
    plt.figure(figsize=(8, 5))
    corr = df[['weight', 'shipment_cost', 'delivery_time_days']].corr()
    sns.heatmap(corr, annot=True, cmap='Reds')
    plt.title("EDA: Correlation for Cost Drivers (Objective 2)")
    plt.show()

    # 2. Boxplot: Delivery Mode vs Cost
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='delivery_mode', y='shipment_cost', data=df)
    plt.title("EDA: Impact of Mode on Cost (Objective 2)")
    plt.show()

    # 3. Barplot: Reliability Check (On-time vs Delayed)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='carrier_name', y='on_time_flag', data=df)
    plt.title("EDA: Carrier Impact on Reliability (Objective 1)")
    plt.show()

def plot_model_results(y_test, y_pred, model_name):
    # 4. Scatter Plot: Actual vs Predicted (Regression)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.title(f"{model_name}: Actual vs Predicted Cost")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.show()

def plot_classification_results(cm, model_name):
    # 5. Confusion Matrix (Classification)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name}: Delivery Accuracy Matrix")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.show()

def perform_feature_selection_eda_Kmeans(df):
    """
    Creates a grid of scatter plots (subplots) to evaluate feature relationships 
    against Operational Cost. This helps justify which features to keep for clustering.
    """
    # 1. Identify numerical columns for analysis
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # We will plot every feature against 'operational_cost_per_day' (our benchmark)
    target_feature = 'operational_cost_per_day'
    features_to_plot = [col for col in numerical_cols if col != target_feature]
    
    # 2. Dynamically calculate grid dimensions
    n_features = len(features_to_plot)
    n_cols = 2
    n_rows = math.ceil(n_features / n_cols)
    
    # 3. Initialize the subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()
    
    # 4. Generate scatter plots in the grid
    for i, col in enumerate(features_to_plot):
        sns.scatterplot(data=df, x=target_feature, y=col, ax=axes[i], color='teal', s=80)
        axes[i].set_title(f'{target_feature} vs {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Operational Cost (Daily)')
        axes[i].set_ylabel(col)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
    # 5. Hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    fig.suptitle("Feature Selection EDA: Numerical Relationships for Clustering", fontsize=16)
    plt.show()


def plot_Kmeans_cluster_scaled(
    X_scaled,
    labels,
    centers,
    feature_names,
    y_feature_index=1
):

    """
    Visualizes scaled K-Means clusters with business-friendly cluster names.
    """

    plt.figure(figsize=(10, 6))

    # X and Y data
    x_data = X_scaled[:, 0]
    y_data = X_scaled[:, y_feature_index]

    # Convert labels to 1D
    labels = labels.squeeze()

    # Number of clusters
    n_clusters = len(set(labels))

    # Dynamic business-friendly cluster names
    if n_clusters == 2:

        cluster_name_map = {
            0: "Small & Cost-Efficient Warehouses",
            1: "Large High-Capacity Warehouses"
        }

    elif n_clusters == 3:

        cluster_name_map = {
            0: "Low-Cost Local Warehouses",
            1: "Balanced Regional Warehouses",
            2: "High-Capacity Distribution Centers"
        }

    elif n_clusters == 5:

        cluster_name_map = {
            0: "Small Local Storage Units",
            1: "Mega Distribution Hubs",
            2: "Cost-Efficient Warehouses",
            3: "High-Capacity Fulfillment Centers",
            4: "Scaling Mid-Size Warehouses"
        }

    else:
        # Default fallback names
        cluster_name_map = {
            cluster: f"Cluster {cluster}"
            for cluster in sorted(set(labels))
        }

    # Replace numeric labels with names
    labels_named = [cluster_name_map[label] for label in labels]

    # Scatter plot
    sns.scatterplot(
        x=x_data,
        y=y_data,
        hue=labels_named,
        palette='viridis',
        s=150,
        edgecolor='black',
        alpha=0.7
    )

    # Plot centroids
    plt.scatter(
        centers[:, 0],
        centers[:, y_feature_index],
        c='red',
        s=400,
        marker='X',
        label='Cluster Centroids',
        edgecolors='white',
        linewidth=2
    )

    # Axis labels
    x_label = feature_names[0].replace('_', ' ').title()
    y_label = feature_names[y_feature_index].replace('_', ' ').title()

    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel(y_label, fontsize=12, fontweight='bold')

    plt.title(
        f"Warehouse Segmentation: {x_label} vs {y_label}",
        fontsize=14,
        fontweight='bold'
    )

    # Legend
    plt.legend(
        title="Warehouse Groups",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    plt.show()


    
def plot_Kmeans_cluster_original(X, labels,feature_names, y_feature_index=1):
    x_data = X['operational_cost_per_day'].squeeze()
    y_data = X.iloc[:, y_feature_index].squeeze()
    labels = labels.squeeze()
    
    if y_feature_index == 1:
        plt.figure(figsize=(10, 6))
    
        n_clusters = len(set(labels))
    
        if n_clusters == 2:
            cluster_name_map = {
                0: "Small & Cost-Efficient Warehouses",
                1: "Large High-Capacity Warehouses"
            }
    
        elif n_clusters == 3:
            cluster_name_map = {
                0: "Low-Cost Local Warehouses",
                1: "Balanced Regional Warehouses",
                2: "High-Capacity Distribution Centers"
            }
    
        elif n_clusters == 5:
            cluster_name_map = {
                0: "Small Local Storage Units",
                1: "Mega Distribution Hubs",
                2: "Cost-Efficient Warehouses",
                3: "High-Capacity Fulfillment Centers",
                4: "Scaling Mid-Size Warehouses"
            }
    
        else:
            cluster_name_map = {
                cluster: f"Cluster {cluster}"
                for cluster in sorted(set(labels))
            }
    
        labels_named = [cluster_name_map[label] for label in labels]
    
        sns.scatterplot(
            x=x_data,
            y=y_data,
            hue=labels_named,
            palette='viridis',
            s=150,
            edgecolor='black',
            alpha=0.7
        )
         # Plot centroids
        plt.scatter(
            centers[:, 0],
            centers[:, y_feature_index],
            c='red',
            s=400,
            marker='X',
            label='Cluster Centroids',
            edgecolors='white',
            linewidth=2
        )
         
    
        x_label = X.columns[0].replace('_', ' ').title()
        y_label = X.columns[y_feature_index].replace('_', ' ').title()
    
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.title(f"Warehouse Segmentation: {x_label} vs {y_label}", fontsize=14, fontweight='bold')
        plt.legend(title="Warehouse Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()
    
    elif y_feature_index == 2:
        plt.figure(figsize=(10, 6))
    
        n_clusters = len(set(labels))
    
        if n_clusters == 3:
            df_temp = pd.DataFrame({'x': x_data, 'y': y_data, 'label': labels})
            cluster_means = df_temp.groupby('label').mean()
            
            cluster_name_map = {}
            for cluster_id, row in cluster_means.iterrows():
                if row['x'] < 500 and row['y'] < 300:
                    cluster_name_map[cluster_id] = "Low productivity Low operational cost"
                elif row['x'] < 1000 and row['y'] >= 300:
                    cluster_name_map[cluster_id] = "High productivity Low operational cost"
                else:
                    cluster_name_map[cluster_id] = "High productivity High operational cost"
        else:
            cluster_name_map = {
                cluster: f"Cluster {cluster}"
                for cluster in sorted(set(labels))
            }
    
        labels_named = [cluster_name_map[label] for label in labels]
    
        palette_map = {
            "High productivity Low operational cost": "green",
            "High productivity High operational cost": "#FFBF00",
            "Low productivity Low operational cost": "yellow"
        }
    
        sns.scatterplot(
            x=x_data,
            y=y_data,
            hue=labels_named,
            palette=palette_map,
            s=150,
            edgecolor='black',
            alpha=0.7
        )
        
        # -------------------------------------------------------------
        # FIX: Calculate real-world centroids directly from the raw data
        # -------------------------------------------------------------
        # This completely bypasses the StandardScaler error!
        df_temp = pd.DataFrame({'x': x_data, 'y': y_data, 'label': labels})
        real_centers = df_temp.groupby('label').mean().reset_index()
        
        plt.scatter(
            real_centers['x'],  # Real-world X coordinates (Operational Cost)
            real_centers['y'],  # Real-world Y coordinates (Labor Productivity)
            c='red',
            s=300,
            marker='X',
            label='Cluster Centroids',
            edgecolors='black',
            linewidth=1.5,
            zorder=5              # Keeps centroids pinned to the very top layer
        )
        # -------------------------------------------------------------
    
        x_label = X.columns[0].replace('_', ' ').title()
        y_label = X.columns[y_feature_index].replace('_', ' ').title()
    
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.title(f"Warehouse Segmentation: {x_label} vs {y_label}", fontsize=14, fontweight='bold')
        plt.legend(title="Warehouse Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()