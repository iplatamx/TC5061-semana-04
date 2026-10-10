"""Entrena y evalúa el modelo baseline de churn del notebook."""

import argparse
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().with_name(
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Telco_Customer_Churn_Baseline"
TEST_SIZE = 0.20


def load_data(data_path: Path) -> pd.DataFrame:
    """Carga el dataset Telco Customer Churn desde el CSV."""
    return pd.read_csv(data_path)


def preprocess_data(data: pd.DataFrame):
    """Limpia TotalCharges y separa la variable objetivo de los predictores."""
    frame = data.copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame.loc[frame["TotalCharges"].isna(), "TotalCharges"] = 0

    target = frame[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    features = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    categorical_features = [
        column for column in features.columns if column not in NUMERIC_FEATURES
    ]
    return features, target, categorical_features


def build_model(
    categorical_features: list[str],
    c: float,
    solver: str,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """Construye el preprocesamiento y la regresión logística del notebook."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    solver=solver,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model(
    model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Ajusta el pipeline con los datos de entrenamiento."""
    return model.fit(x_train, y_train)


def evaluate_model(
    model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, object]:
    """Calcula las métricas, matriz de confusión y reporte del notebook."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(
            y_test, predictions, target_names=["No Churn", "Churn"]
        ),
    }


def parse_args() -> argparse.Namespace:
    """Lee los hiperparámetros y la semilla desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa el modelo baseline de churn."
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización (default: 1.0).",
    )
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
        default="lbfgs",
        help="Optimizador de LogisticRegression (default: lbfgs).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Máximo de iteraciones de LogisticRegression (default: 1000).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para la división de datos y el modelo (default: 42).",
    )
    args = parser.parse_args()
    if args.c <= 0:
        parser.error("--c debe ser mayor que cero.")
    if args.max_iter < 1:
        parser.error("--max-iter debe ser al menos 1.")
    return args


def main(args: argparse.Namespace) -> None:
    """Ejecuta carga, preparación, entrenamiento y evaluación del baseline."""
    data = load_data(DATA_PATH)
    features, target, categorical_features = preprocess_data(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=args.random_state,
        stratify=target,
    )

    model = build_model(
        categorical_features,
        c=args.c,
        solver=args.solver,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="LogisticRegression_Baseline"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", args.c)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("encoding", "OneHotEncoder")
        mlflow.log_param("scaling", "StandardScaler")

        train_model(model, x_train, y_train)
        results = evaluate_model(model, x_test, y_test)
        for metric_name in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
            mlflow.log_metric(metric_name, results[metric_name])

        with tempfile.TemporaryDirectory(prefix="telco-churn-") as temp_dir:
            artifact_path = Path(temp_dir) / "evaluation.txt"
            artifact_path.write_text(
                "Matriz de confusión (filas=real, columnas=predicción):\n"
                f"{results['confusion_matrix']}\n\n"
                f"Reporte de clasificación:\n{results['classification_report']}",
                encoding="utf-8",
            )
            mlflow.log_artifact(str(artifact_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="logistic_regression_baseline",
            input_example=x_test.iloc[[0]],
        )
        run_id = mlflow.active_run().info.run_id

    print("Métricas del modelo baseline:")
    for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"{name}: {results[name]:.4f}")
    print("\nMatriz de confusión:")
    print(results["confusion_matrix"])
    print("\nReporte de clasificación:")
    print(results["classification_report"])
    print(f"\nMLflow Run ID: {run_id}")


if __name__ == "__main__":
    main(parse_args())
