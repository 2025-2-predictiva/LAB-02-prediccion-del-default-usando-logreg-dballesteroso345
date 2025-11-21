from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import gzip
import pickle
import pandas as pd
import os
import json

# Constantes de rutas y nombres de columnas
DATA_DIR = "files/input"
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_PATH = "files/output/metrics.json"
TARGET_COL = "default"

CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
NUM_COLS = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


# %% [markdown]
# # 2. Carga y limpieza de datos

# %%
def load_zip_csv(filename: str) -> pd.DataFrame:
    """Lee un CSV comprimido (.csv.zip) desde files/input usando pandas."""
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path, compression="zip", index_col=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza según el enunciado: renombrar, quitar ID, filtrar y agrupar EDUCATION."""
    data = df.copy()

    # Renombrar columna objetivo y eliminar ID
    data = data.rename(columns={"default payment next month": TARGET_COL})
    data = data.drop(columns=["ID"])

    # Eliminar filas con NaN (por si acaso)
    data = data.dropna()

    # Filtrar EDUCATION y MARRIAGE donde 0 es N/A
    mask = (data["EDUCATION"] != 0) & (data["MARRIAGE"] != 0)
    data = data.loc[mask].copy()

    # EDUCATION > 4 -> 4 (others)
    data.loc[data["EDUCATION"] > 4, "EDUCATION"] = 4

    return data


def split_X_y(df: pd.DataFrame):
    """Separa características X y objetivo y."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


# %% [markdown]
# # 3. Construcción del pipeline de clasificación

# %%
def build_pipeline() -> Pipeline:
    """Pipeline: OHE para categóricas, escalado para numéricas, SelectKBest + Regresión Logística."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", MinMaxScaler(), NUM_COLS),
        ],
        remainder="passthrough",
    )

    selector = SelectKBest(score_func=f_classif)

    clf = LogisticRegression(
        max_iter=1000,
        solver="saga",
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("select", selector),
            ("clf", clf),
        ]
    )
    return pipe


def tune_hyperparameters(pipe: Pipeline, X_train, y_train, n_splits: int = 10, scoring: str = "balanced_accuracy"):
    """GridSearchCV para k, penalty y C de la regresión logística."""
    param_grid = {
        "select__k": range(1, 11),
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
    }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=n_splits,
        scoring=scoring,
        refit=True,
        verbose=0,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    return search


# %% [markdown]
# # 4. Cálculo de métricas y matrices de confusión

# %%
def make_metrics_record(dataset_name: str, y_true, y_pred) -> dict:
    """Diccionario con las métricas pedidas para train/test."""
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def make_confusion_record(dataset_name: str, y_true, y_pred) -> dict:
    """Diccionario con la matriz de confusión en el formato del enunciado."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(tn),
            "predicted_1": int(fp),
        },
        "true_1": {
            "predicted_0": int(fn),
            "predicted_1": int(tp),
        },
    }


# %% [markdown]
# # 5. Guardado de modelo y métricas

# %%
def save_model(model, path: str = MODEL_PATH) -> None:
    """Guarda el modelo comprimido con gzip + pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def save_metrics(records, path: str = METRICS_PATH) -> None:
    """Guarda métricas y matrices de confusión en JSON (una línea por registro)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# %% [markdown]
# # 6. Ejecución completa del pipeline

# %%
def main():
    # Paso 1: cargar datasets
    train_raw = load_zip_csv("train_data.csv.zip")
    test_raw = load_zip_csv("test_data.csv.zip")

    # Paso 2: limpiar
    train_clean = clean_data(train_raw)
    test_clean = clean_data(test_raw)

    # Paso 3: separar X e y
    X_train, y_train = split_X_y(train_clean)
    X_test, y_test = split_X_y(test_clean)

    # Paso 4: crear pipeline y optimizar hiperparámetros
    base_pipe = build_pipeline()
    best_model = tune_hyperparameters(base_pipe, X_train, y_train, n_splits=10, scoring="balanced_accuracy")

    # Paso 5: guardar modelo
    save_model(best_model)

    # Paso 6: predicciones
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Paso 7: métricas
    train_metrics = make_metrics_record("train", y_train, y_train_pred)
    test_metrics = make_metrics_record("test", y_test, y_test_pred)

    # Paso 8: matrices de confusión
    train_cm = make_confusion_record("train", y_train, y_train_pred)
    test_cm = make_confusion_record("test", y_test, y_test_pred)

    # Paso 9: guardar todo en metrics.json
    all_records = [train_metrics, test_metrics, train_cm, test_cm]
    save_metrics(all_records)


# %%
if __name__ == "__main__":
    main()