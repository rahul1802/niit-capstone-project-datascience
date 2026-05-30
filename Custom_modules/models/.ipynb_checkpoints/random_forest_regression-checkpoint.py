# supply_chain_project/models/random_forest_model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from preprocessing.encoding_scaling import encode_features

def run_random_forest(df):
    df['carrier_enc'] = encode_features(df,'carrier_name')
    
    X = df[['weight', 'carrier_enc']]
    y = df['shipment_cost']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_test, y_test