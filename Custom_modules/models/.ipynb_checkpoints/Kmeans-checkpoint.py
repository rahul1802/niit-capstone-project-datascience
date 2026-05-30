import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def find_optimal_K_elbowmethod(df_scaled):
    inertia = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(k_range, inertia, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()

def find_optimal_K_silhouette(df_scaled):
    scores = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        scores.append(silhouette_score(df_scaled, labels))

    plt.figure(figsize=(7, 4))
    plt.plot(k_range, scores, marker="o", color="orange")
    plt.title("Silhouette Score for Optimal K")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.show()
    return k_range[scores.index(max(scores))]

def run_KMeans(df,features_selected,selected_k=None):
    """
    Final implementation. If selected_k is None, it will find the best one automatically.
    """
    # 1. CRITICAL: Select only the numeric features identified in your EDA
    features = features_selected
    X = df[features].copy()

    # 2. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Decision Logic: Only plot if we don't know the K yet
    if selected_k is None:
        find_optimal_K_elbowmethod(X_scaled)
        selected_k = find_optimal_K_silhouette(X_scaled)
        print(f"Automatically selected Optimal K: {selected_k}")

    # 4. Final Model implementation
    kmeans = KMeans(n_clusters=selected_k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    return kmeans, X_scaled