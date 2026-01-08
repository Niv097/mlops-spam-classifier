import pandas as pd
import joblib
import warnings

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning

EXPERIMENT_NAME = "Spam_Classifier_MLOps"
THRESHOLD_MARGIN = 0.01
RANDOM_STATE = 42

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

mlflow.set_experiment(EXPERIMENT_NAME)


data = pd.read_csv("data.csv")

X = data["text"]
y = data["label"]


if y.nunique() < 2:
    raise ValueError("❌ Dataset must contain at least 2 classes")

if y.value_counts().min() < 2:
    raise ValueError("❌ Each class must have at least 2 samples")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train labels:")
print(y_train.value_counts())
print("\nTest labels:")
print(y_test.value_counts())


baseline_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

with mlflow.start_run(run_name="baseline_model"):
    baseline_pipeline.fit(X_train, y_train)

    baseline_preds = baseline_pipeline.predict(X_test)
    baseline_f1 = f1_score(y_test, baseline_preds, pos_label="spam")

    # Log params
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("ngram_range", "(1,1)")
    mlflow.log_param("train_ham", (y_train == "ham").sum())
    mlflow.log_param("train_spam", (y_train == "spam").sum())

    # Log metric
    mlflow.log_metric("f1_score", baseline_f1)

    # Log report
    baseline_report = classification_report(y_test, baseline_preds)
    with open("baseline_report.txt", "w") as f:
        f.write(baseline_report)
    mlflow.log_artifact("baseline_report.txt")

    # Log model
    mlflow.sklearn.log_model(baseline_pipeline, "baseline_model")

    print("Baseline F1 Score:", baseline_f1)


new_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_df=0.9)),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

with mlflow.start_run(run_name="new_model"):
    new_pipeline.fit(X_train, y_train)

    new_preds = new_pipeline.predict(X_test)
    new_f1 = f1_score(y_test, new_preds, pos_label="spam")

    # Log params
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_param("max_df", 0.9)

    # Log metric
    mlflow.log_metric("f1_score", new_f1)

    # Log report
    new_report = classification_report(y_test, new_preds)
    with open("new_report.txt", "w") as f:
        f.write(new_report)
    mlflow.log_artifact("new_report.txt")


    mlflow.sklearn.log_model(new_pipeline, "new_model")

    print("New F1 Score:", new_f1)


if new_f1 >= baseline_f1 + THRESHOLD_MARGIN:
    print("✅ New model approved for deployment")
    joblib.dump(new_pipeline, "model.pkl")
else:
    print("❌ New model rejected, baseline retained")
    joblib.dump(baseline_pipeline, "model.pkl")
