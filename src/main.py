import pandas as pd
from sklearn.tree import DecisionTreeClassifier


FEATURES = ["Age", "Fare", "Sex", "Pclass"]


def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    res = df.copy()
    res["Sex"] = df['Sex'].map({'male': 0, 'female': 1})
    return res


def main():
    train_df = pd.read_csv("src/resources/train.csv")
    test_df = pd.read_csv("src/resources/test.csv")

    predictions_df = predictions(test_df, train_df)
    predictions_df.to_csv("titanic_preds.csv", index=False)


def predictions(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    encoded_test_df = encode_sex(test_df)
    encoded_train_df = encode_sex(train_df)

    model = DecisionTreeClassifier()

    X = encoded_train_df[FEATURES]
    y = encoded_train_df["Survived"]

    model.fit(X, y)
    predictions_df = get_predictions(encoded_test_df, model)
    return predictions_df


def get_predictions(df: pd.DataFrame, model: DecisionTreeClassifier) -> pd.DataFrame:
    pred_col = model.predict(df[FEATURES])
    res = df.copy()[["PassengerId"]]
    res["Survived"] = pred_col
    return res


if __name__ == "__main__":
    main()
