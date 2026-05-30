# supply_chain_project/models/evaluate.py
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)


def evaluate_regression(model, X_test, y_test):
    preds = model.predict(X_test)
    print(f"R2 Score: {r2_score(y_test, preds):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    return preds


def evaluate_classification(model, X_test, y_test):
    preds = model.predict(X_test)
    print(f"Accuracy Score: {accuracy_score(y_test, preds):.3f}")
    print(f"F1_Score: {f1_score(y_test, preds):.3f}")
    return confusion_matrix(y_test, preds)


def evaluateKmeans(X_scaled, labels):
    """
    Calculates key clustering metrics for mentor validation.
    """
    db_score = davies_bouldin_score(X_scaled, labels)
    sil_score = silhouette_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    print(f"--- KMeans Evaluation Metrics ---")
    print(f"Silhouette Score (Higher is better): {sil_score:.3f}")
    print(f"Davies-Bouldin Index (Lower is better): {db_score:.3f}")
    print(f"Calinski-Harabasz Score: {ch_score:.3f}")

    return {"silhouette": sil_score, "db_index": db_score, "ch_score": ch_score}
