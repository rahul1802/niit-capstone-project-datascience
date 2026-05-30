# preprocessing/encoding_scaling.py
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def encode_features(df, col):

    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

    return df[col]



def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled