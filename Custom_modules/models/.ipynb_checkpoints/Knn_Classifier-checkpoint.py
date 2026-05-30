# supply_chain_project/models/knn_classifier_model.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def find_optimal_k(df, max_k=21):
    """
    Optimization function to find best K.
    Uses features that align with Business Objective 1.3: Carrier and Mode impact.
    """
    # 1. Feature Selection based on Business Questions
    # We use 'weight', 'delivery_mode', and 'carrier_name'
    X = df[["weight", "delivery_mode", "carrier_name"]].copy()
    y = df["on_time_flag"]

    # 2. Encoding Categorical Data (Required for KNN)
    le = LabelEncoder()
    X["delivery_mode"] = le.fit_transform(X["delivery_mode"].astype(str))
    X["carrier_name"] = le.fit_transform(X["carrier_name"].astype(str))

    # 3. Scaling (Required for distance-based models like KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Split for optimization
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    k_values = range(1, max_k)
    accuracy_scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = accuracy_score(y_test, knn.predict(X_test))
        accuracy_scores.append(score)

    # Visual: Optimization plot for mentor review
    plt.figure(figsize=(7, 5))
    plt.plot(k_values, accuracy_scores, marker="o", color="green")
    plt.xlabel("K Value")
    plt.ylabel("Accuracy Score")
    plt.title("KNN Optimization: Impact of Carrier & Mode on Accuracy")
    plt.grid(True)
    plt.show()

    best_k = k_values[int(np.argmax(accuracy_scores))]
    print(f"Optimal K for Carrier/Mode analysis: {best_k}")
    return best_k


def run_knn(df,k):
    """
    Final model training using optimized K and business-aligned features.
    """
    # Re-prepare data for final model
    X = df[["weight", "delivery_mode", "carrier_name"]].copy()
    y = df["on_time_flag"]

    # Encode & Scale
    le = LabelEncoder()
    X["delivery_mode"] = le.fit_transform(X["delivery_mode"].astype(str))
    X["carrier_name"] = le.fit_transform(X["carrier_name"].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Final Model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    return model, X_test, y_test
