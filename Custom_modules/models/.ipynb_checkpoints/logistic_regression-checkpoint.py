# supply_chain_project/models/logistic_regression_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocessing.encoding_scaling import scale_features,encode_features

def run_logistic_regression(df):
    df['mode_enc'] = encode_features(df,'delivery_mode')
    
    X = df[['weight', 'mode_enc']]
    y = df['on_time_flag']
    
    X_scaled = scale_features(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test