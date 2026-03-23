"""
model_evaluator.py
Constrói e avalia o pipeline sklearn com os hiperparâmetros decodificados do cromossomo.

O pipeline:
  Scaler → [PCA opcional] → LogisticRegression

Métricas retornadas (contexto médico feminino):
  - recall      : sensibilidade — minimiza falsos negativos (câncer não detectado)
  - specificity : minimiza alarmes falsos (benignos tratados como malignos)
  - f1_score    : equilíbrio precisão × recall
  - accuracy    : acurácia geral
"""

import os
import warnings
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report,
)

from .chromosome import decode

warnings.filterwarnings("ignore")

# Caminho para os dados da Fase 1
_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "breast_cancer_dataset.csv"
)

# Cache dos dados (evita recarregar a cada avaliação)
_DATA_CACHE: Dict = {}


# ---------------------------------------------------------------------------
# Carregamento e pré-processamento dos dados
# ---------------------------------------------------------------------------

def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega e prepara o dataset breast_cancer_dataset.csv da Fase 1.

    Returns:
        X (features), y (target: 0=benigno, 1=maligno)
    """
    if "X" in _DATA_CACHE:
        return _DATA_CACHE["X"], _DATA_CACHE["y"]

    df = pd.read_csv(_DATA_PATH)

    # Remove colunas irrelevantes (igual Fase 1)
    drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Codifica target
    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

    X = df.drop(columns=["diagnosis"]).values.astype(float)
    y = df["diagnosis"].values

    _DATA_CACHE["X"] = X
    _DATA_CACHE["y"] = y
    return X, y


# ---------------------------------------------------------------------------
# Construção do pipeline
# ---------------------------------------------------------------------------

def build_pipeline(hyperparams: Dict[str, Any]) -> Pipeline:
    """
    Constrói um Pipeline sklearn a partir dos hiperparâmetros decodificados.

    Args:
        hyperparams: dict retornado por chromosome.decode()

    Returns:
        sklearn Pipeline pronto para treino/avaliação
    """
    scaler_map = {
        "StandardScaler": StandardScaler(),
        "RobustScaler":   RobustScaler(),
        "MinMaxScaler":   MinMaxScaler(),
    }

    steps = [("scaler", scaler_map[hyperparams["scaler"]])]

    if hyperparams["use_pca"]:
        steps.append(("pca", PCA(
            n_components=hyperparams["pca_variance"],
            random_state=42,
        )))

    steps.append(("clf", LogisticRegression(
        C=hyperparams["C"],
        solver=hyperparams["solver"],
        max_iter=hyperparams["max_iter"],
        random_state=42,
        class_weight="balanced",   # lida com leve desbalanceamento
    )))

    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Avaliação por validação cruzada (usada pelo AG)
# ---------------------------------------------------------------------------

def evaluate_genes(
    genes,
    X: np.ndarray = None,
    y: np.ndarray = None,
    cv_folds: int = 5,
) -> Dict[str, float]:
    """
    Avalia um cromossomo usando validação cruzada estratificada.

    Args:
        genes:    vetor de genes do indivíduo
        X, y:     dados (carregados do CSV se None)
        cv_folds: número de folds para CV

    Returns:
        dict com métricas médias: recall, specificity, f1, accuracy, fitness
    """
    if X is None or y is None:
        X, y = load_data()

    hyperparams = decode(genes)

    try:
        pipeline = build_pipeline(hyperparams)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        scoring = {
            "recall":   "recall",
            "f1":       "f1",
            "accuracy": "accuracy",
            "specificity": "recall_macro",  # aproximação — calculada manualmente abaixo
        }

        # Calcula recall e f1 via cross_validate
        cv_results = cross_validate(
            pipeline, X, y, cv=cv,
            scoring={"recall": "recall", "f1": "f1", "accuracy": "accuracy"},
            return_train_score=False,
        )

        recall   = float(np.mean(cv_results["test_recall"]))
        f1       = float(np.mean(cv_results["test_f1"]))
        accuracy = float(np.mean(cv_results["test_accuracy"]))

        # Especificidade: média de TN/(TN+FP) em cada fold
        specificities = []
        for train_idx, val_idx in cv.split(X, y):
            pipe_fold = build_pipeline(hyperparams)
            pipe_fold.fit(X[train_idx], y[train_idx])
            y_pred = pipe_fold.predict(X[val_idx])
            tn, fp, fn, tp = confusion_matrix(y[val_idx], y_pred, labels=[0, 1]).ravel()
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        specificity = float(np.mean(specificities))

    except Exception:
        return {"recall": 0.0, "specificity": 0.0, "f1": 0.0, "accuracy": 0.0, "fitness": 0.0}

    fitness = compute_fitness(recall, specificity, f1)

    return {
        "recall":      round(recall, 6),
        "specificity": round(specificity, 6),
        "f1":          round(f1, 6),
        "accuracy":    round(accuracy, 6),
        "fitness":     round(fitness, 6),
    }


def compute_fitness(recall: float, specificity: float, f1: float) -> float:
    """
    Função de fitness composta — maximizar.

    Pesos alinhados com o contexto médico da Fase 2:
      50% recall      → prioridade máxima (falsos negativos = câncer não detectado)
      30% f1_score    → equilíbrio geral
      20% specificity → evitar alarmes falsos desnecessários
    """
    return 0.50 * recall + 0.30 * f1 + 0.20 * specificity


# ---------------------------------------------------------------------------
# Avaliação completa no conjunto de teste (para relatório final)
# ---------------------------------------------------------------------------

def evaluate_full(
    genes,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
) -> Dict[str, Any]:
    """
    Treina no conjunto completo de treino e avalia no teste.
    Usado para gerar o relatório comparativo final.
    """
    hyperparams = decode(genes)
    pipeline = build_pipeline(hyperparams)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = f1_score(y_test, y_pred)
    accuracy    = accuracy_score(y_test, y_pred)
    fitness     = compute_fitness(recall, specificity, f1)

    return {
        "hyperparams":   hyperparams,
        "recall":        round(recall, 4),
        "specificity":   round(specificity, 4),
        "f1":            round(f1, 4),
        "accuracy":      round(accuracy, 4),
        "fitness":       round(fitness, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "report":        classification_report(y_test, y_pred, target_names=["Benigno", "Maligno"]),
        "pipeline":      pipeline,
    }
