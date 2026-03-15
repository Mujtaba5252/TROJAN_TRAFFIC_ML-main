import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):

    df = pd.read_csv(path)

    # remove extra spaces in column names
    df.columns = df.columns.str.strip()

    # drop non-numeric / identifier columns
    df = df.drop(columns=[
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp"
    ], errors="ignore")

    # convert labels
    df["Class"] = df["Class"].map({
        "Trojan": 1,
        "Benign": 0
    })

    # remove missing values if any
    df = df.dropna()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    return X, y


def split_data(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler