import pandas as pd

import re
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

from error_logs import configure_logger

logger = configure_logger()


def nlp_pipeline_generator(text_series):

    for text in text_series:
        if not isinstance(text, str):
            return []

        # convert lowecase
        text = text.lower()

        # Remove Punction and Numbers
        text = re.sub(r"[^a-z\s]", "", text)

        # Tokenize
        token = word_tokenize(text)

        # Remove stopwords
        filtered_tokens = [word for word in token if word not in stop_words]

        # Lemmatize
        tokens = [lemmatizer.lemmatize(t) for t in filtered_tokens]

        return tokens


def model_traning(data):
    features = data.drop(columns="label")
    target = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    ColumnTF = ColumnTransformer(
        [
            (
                "tfidf_text",
                TfidfVectorizer(analyzer=nlp_pipeline_generator, max_features=5000),
                "text",
            ),
            (
                "tfidf_title",
                TfidfVectorizer(analyzer=nlp_pipeline_generator, max_features=5000),
                "title",
            ),
            ("scaled_TitleWC", StandardScaler(), ["title_word_count"]),
            ("scaled_TextWC", StandardScaler(), ["text_word_count"]),
            ("scaled_Textl", StandardScaler(), ["text_len"]),
            ("scaled_Titlel", StandardScaler(), ["title_len"]),
        ]
    )

    pipeline = Pipeline(
        [("vectorizer", ColumnTF), ("RF", RandomForestClassifier(verbose=3))],
        verbose=True,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    model_path = os.path.join("models", "fake_news_full_pipeline.pkl")

    joblib.dump(pipeline, model_path)
    print(f"Model saved successfully at: {model_path}")

    folder_name = "models"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"'{folder_name}' folder created inside your project.")

    model_path = os.path.join(folder_name, "fake_news_full_pipeline.pkl")

    joblib.dump(pipeline, model_path)

    print(f"Success! Your model is saved at: {model_path}")
    return
