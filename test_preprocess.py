import pandas as pd
from src.preprocess import load_data, split_data, scale_data


def test_pipeline():

    # create dummy dataset
    df = pd.DataFrame({
        "Flow Duration":[1,2,3,4],
        "Total Fwd Packets":[2,3,4,5],
        "Total Backward Packets":[1,2,1,2],
        "Class":["Trojan","Benign","Trojan","Benign"]
    })

    df.columns = df.columns.str.strip()

    df["Class"] = df["Class"].map({
        "Trojan":1,
        "Benign":0
    })

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train,X_val,X_test,y_train,y_val,y_test = split_data(X,y)

    X_train,X_val,X_test,_ = scale_data(X_train,X_val,X_test)

    assert X_train.shape[1] == 3
    assert len(X_train) > 0