import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_train():
    return pd.read_csv("Data/train-2.csv")

def build_text(df):
    return df["caption"].fillna("") + "[SEP]" + df["content"].fillna("")

def train_baseline():
    df = load_train()

    X = build_text(df)
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=40000)),
        ("clf", LogisticRegression(max_iter=300))
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)

    print(classification_report(y_val, preds))

    return pipe


if __name__ == "__main__":
    model = train_baseline()