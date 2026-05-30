# supply_chain_project/models/evaluate.py
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score,f1_score,confusion_matrix

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