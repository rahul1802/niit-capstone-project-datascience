# supply_chain_project/models/linear_regression_model.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from preprocessing.encoding_scaling import encode_features

def run_linear_regression(df):
    # Features selected based on EDA
    df['mode_enc'] = encode_features(df,'delivery_mode')
    
    X = df[['weight', 'mode_enc']]
    y = df['shipment_cost']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test