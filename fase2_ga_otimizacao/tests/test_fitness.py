"""
test_fitness.py
Testes unitários para avaliação do modelo e função de fitness.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from sklearn.model_selection import train_test_split

from fase2_ga_otimizacao.src.chromosome import baseline_genes, random_genes, decode
from fase2_ga_otimizacao.src.model_evaluator import (
    load_data, build_pipeline, evaluate_genes, evaluate_full, compute_fitness,
)


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def split_data(data):
    X, y = data
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


class TestLoadData:
    def test_shape(self, data):
        X, y = data
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 30
        assert set(y) == {0, 1}

    def test_no_nan(self, data):
        X, y = data
        assert not np.isnan(X).any()


class TestBuildPipeline:
    def test_baseline_pipeline(self):
        hp = decode(baseline_genes())
        pipe = build_pipeline(hp)
        assert "scaler" in pipe.named_steps
        assert "clf" in pipe.named_steps

    def test_no_pca_pipeline(self):
        genes = random_genes()
        genes[1] = 0.0   # use_pca = False
        hp = decode(genes)
        pipe = build_pipeline(hp)
        assert "pca" not in pipe.named_steps

    def test_with_pca_pipeline(self):
        genes = random_genes()
        genes[1] = 1.0   # use_pca = True
        hp = decode(genes)
        pipe = build_pipeline(hp)
        assert "pca" in pipe.named_steps


class TestEvaluateGenes:
    def test_baseline_recall_above_threshold(self, data):
        X, y = data
        metrics = evaluate_genes(baseline_genes(), X, y, cv_folds=3)
        assert metrics["recall"] >= 0.85, f"Recall baseline muito baixo: {metrics['recall']}"

    def test_metrics_in_range(self, data):
        X, y = data
        metrics = evaluate_genes(random_genes(), X, y, cv_folds=3)
        for key in ("recall", "specificity", "f1", "accuracy", "fitness"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key} fora de [0,1]: {metrics[key]}"

    def test_fitness_formula(self, data):
        X, y = data
        metrics = evaluate_genes(baseline_genes(), X, y, cv_folds=3)
        expected = compute_fitness(metrics["recall"], metrics["specificity"], metrics["f1"])
        assert abs(metrics["fitness"] - expected) < 1e-6


class TestEvaluateFull:
    def test_confusion_matrix_sums(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        result = evaluate_full(baseline_genes(), X_train, X_test, y_train, y_test)
        cm = result["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_test)

    def test_recall_above_baseline(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        result = evaluate_full(baseline_genes(), X_train, X_test, y_train, y_test)
        assert result["recall"] >= 0.90

    def test_report_is_string(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        result = evaluate_full(baseline_genes(), X_train, X_test, y_train, y_test)
        assert isinstance(result["report"], str)
        assert "Benigno" in result["report"]
        assert "Maligno" in result["report"]


class TestComputeFitness:
    def test_recall_weighted_most(self):
        f_high_recall = compute_fitness(recall=1.0, specificity=0.0, f1=0.0)
        f_high_spec   = compute_fitness(recall=0.0, specificity=1.0, f1=0.0)
        assert f_high_recall > f_high_spec

    def test_perfect_model(self):
        f = compute_fitness(recall=1.0, specificity=1.0, f1=1.0)
        assert abs(f - 1.0) < 1e-9

    def test_zero_model(self):
        f = compute_fitness(recall=0.0, specificity=0.0, f1=0.0)
        assert f == 0.0
