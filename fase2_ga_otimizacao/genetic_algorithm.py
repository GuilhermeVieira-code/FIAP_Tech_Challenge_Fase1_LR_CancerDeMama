"""
genetic_algorithm.py
====================
Tudo relacionado ao AG em um único arquivo:

  PARTE 1 — Cromossomo
    Representa um conjunto de hiperparâmetros como vetor real em [0, 1]^6.
    A função decode() converte os genes em hiperparâmetros concretos do sklearn.

  PARTE 2 — Avaliador de modelos
    Constrói o pipeline sklearn (Scaler → [PCA] → LogisticRegression),
    avalia com validação cruzada e calcula o fitness.

  PARTE 3 — Algoritmo Genético
    Seleção por torneio, crossover uniforme/aritmético,
    mutação gaussiana e elitismo.

Contexto: diagnóstico de câncer de mama (Fase 1 da FIAP Tech Challenge).
"""

import os
import copy
import json
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# PARTE 1 — CROMOSSOMO
# =============================================================================

SCALERS = ["StandardScaler", "RobustScaler", "MinMaxScaler"]
SOLVERS = ["lbfgs", "liblinear", "saga"]

PCA_VAR_MIN  = 0.80
PCA_VAR_MAX  = 0.99
C_LOG_MIN    = -3.0   # C = 10^-3 = 0.001
C_LOG_MAX    =  2.0   # C = 10^2  = 100
MAX_ITER_MIN = 200
MAX_ITER_MAX = 2000

N_GENES = 6


def decode(genes: List[float]) -> Dict[str, Any]:
    """
    Converte o vetor de genes [0,1]^6 em hiperparâmetros concretos.

    Gene 0 → tipo de scaler   (categorical)
    Gene 1 → usar PCA?        (boolean: >= 0.5 = True)
    Gene 2 → variância PCA    (float 0.80–0.99)
    Gene 3 → C do LogReg      (escala log: 10^-3 a 10^2)
    Gene 4 → solver LogReg    (categorical)
    Gene 5 → max_iter         (int 200–2000)
    """
    scaler_idx = min(int(genes[0] * len(SCALERS)), len(SCALERS) - 1)
    use_pca    = genes[1] >= 0.5
    pca_var    = PCA_VAR_MIN + genes[2] * (PCA_VAR_MAX - PCA_VAR_MIN)
    C          = 10 ** (C_LOG_MIN + genes[3] * (C_LOG_MAX - C_LOG_MIN))
    solver_idx = min(int(genes[4] * len(SOLVERS)), len(SOLVERS) - 1)
    max_iter   = int(MAX_ITER_MIN + genes[5] * (MAX_ITER_MAX - MAX_ITER_MIN))

    return {
        "scaler":       SCALERS[scaler_idx],
        "use_pca":      use_pca,
        "pca_variance": round(pca_var, 4),
        "C":            round(C, 6),
        "solver":       SOLVERS[solver_idx],
        "max_iter":     max_iter,
    }


def random_genes(rng: random.Random = None) -> List[float]:
    """Gera um cromossomo aleatório."""
    r = rng or random
    return [r.random() for _ in range(N_GENES)]


def baseline_genes() -> List[float]:
    """
    Genes que reproduzem o modelo vencedor da Fase 1:
    RobustScaler + PCA(0.90) + LogReg(C=1, solver=lbfgs, max_iter=1000)
    """
    return [
        SCALERS.index("RobustScaler") / len(SCALERS) + 0.01,   # gene 0
        0.9,                                                      # gene 1 → use_pca = True
        (0.90 - PCA_VAR_MIN) / (PCA_VAR_MAX - PCA_VAR_MIN),    # gene 2 → pca=0.90
        (0.0  - C_LOG_MIN)   / (C_LOG_MAX   - C_LOG_MIN),      # gene 3 → C=1 (log=0)
        SOLVERS.index("lbfgs") / len(SOLVERS) + 0.01,           # gene 4
        (1000 - MAX_ITER_MIN) / (MAX_ITER_MAX - MAX_ITER_MIN),  # gene 5
    ]


def clip_genes(genes: List[float]) -> List[float]:
    """Garante que todos os genes estejam em [0, 1]."""
    return [max(0.0, min(1.0, g)) for g in genes]


def genes_to_str(genes: List[float]) -> str:
    """Representação legível do cromossomo decodificado."""
    hp = decode(genes)
    pca_str = f"PCA({hp['pca_variance']:.2f})" if hp["use_pca"] else "sem PCA"
    return f"[{hp['scaler']} | {pca_str} | C={hp['C']:.4f} | {hp['solver']} | iter={hp['max_iter']}]"


# =============================================================================
# PARTE 2 — AVALIADOR DE MODELOS
# =============================================================================

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "breast_cancer_dataset.csv")
_DATA_CACHE: Dict = {}


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega o dataset breast_cancer_dataset.csv da Fase 1.
    Usa cache para evitar recarregamento a cada avaliação.
    """
    if "X" in _DATA_CACHE:
        return _DATA_CACHE["X"], _DATA_CACHE["y"]

    df = pd.read_csv(_DATA_PATH)
    drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

    X = df.drop(columns=["diagnosis"]).values.astype(float)
    y = df["diagnosis"].values

    _DATA_CACHE["X"] = X
    _DATA_CACHE["y"] = y
    return X, y


def build_pipeline(hyperparams: Dict[str, Any]) -> Pipeline:
    """Constrói um Pipeline sklearn a partir dos hiperparâmetros decodificados."""
    scaler_map = {
        "StandardScaler": StandardScaler(),
        "RobustScaler":   RobustScaler(),
        "MinMaxScaler":   MinMaxScaler(),
    }
    steps = [("scaler", scaler_map[hyperparams["scaler"]])]

    if hyperparams["use_pca"]:
        steps.append(("pca", PCA(n_components=hyperparams["pca_variance"], random_state=42)))

    steps.append(("clf", LogisticRegression(
        C=hyperparams["C"],
        solver=hyperparams["solver"],
        max_iter=hyperparams["max_iter"],
        random_state=42,
        class_weight="balanced",
    )))
    return Pipeline(steps)


def compute_fitness(recall: float, specificity: float, f1: float, equity: float = 1.0) -> float:
    """
    Função de fitness composta — maximizar.

    Pesos alinhados com o contexto médico feminino:
      45% recall      → falsos negativos = câncer não detectado (crítico)
      25% f1_score    → equilíbrio geral
      20% specificity → evitar alarmes falsos desnecessários
      10% equity      → consistência entre grupos demográficos (quartis)
    """
    return 0.45 * recall + 0.25 * f1 + 0.20 * specificity + 0.10 * equity


def evaluate_genes(
    genes: List[float],
    X: np.ndarray = None,
    y: np.ndarray = None,
    cv_folds: int = 5,
) -> Dict[str, float]:
    """
    Avalia um cromossomo usando validação cruzada estratificada.
    Usada pelo AG para calcular o fitness de cada indivíduo.
    """
    if X is None or y is None:
        X, y = load_data()

    hyperparams = decode(genes)
    try:
        pipeline = build_pipeline(hyperparams)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_results = cross_validate(
            pipeline, X, y, cv=cv,
            scoring={"recall": "recall", "f1": "f1", "accuracy": "accuracy"},
            return_train_score=False,
        )
        recall   = float(np.mean(cv_results["test_recall"]))
        f1       = float(np.mean(cv_results["test_f1"]))
        accuracy = float(np.mean(cv_results["test_accuracy"]))

        # Especificidade e equidade demográfica por fold
        specificities  = []
        equity_scores  = []
        for train_idx, val_idx in cv.split(X, y):
            pipe_fold = build_pipeline(hyperparams)
            pipe_fold.fit(X[train_idx], y[train_idx])
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            y_pred_fold = pipe_fold.predict(X_val_fold)

            # Especificidade
            tn, fp, fn, tp = confusion_matrix(y_val_fold, y_pred_fold, labels=[0, 1]).ravel()
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

            # Equidade: recall por quartil da feature 0 (mean radius)
            # Simula grupos demográficos com base na distribuição dos dados
            quartiles = np.percentile(X_val_fold[:, 0], [25, 50, 75])
            bounds = [
                (-np.inf,       quartiles[0]),
                (quartiles[0],  quartiles[1]),
                (quartiles[1],  quartiles[2]),
                (quartiles[2],  np.inf),
            ]
            group_recalls = []
            for lo, hi in bounds:
                mask = (X_val_fold[:, 0] > lo) & (X_val_fold[:, 0] <= hi)
                if mask.sum() >= 2 and y_val_fold[mask].sum() > 0:
                    from sklearn.metrics import recall_score as _recall_score
                    grp_r = _recall_score(y_val_fold[mask], y_pred_fold[mask], zero_division=0)
                    group_recalls.append(grp_r)
            equity_fold = max(0.0, 1.0 - float(np.std(group_recalls))) if len(group_recalls) >= 2 else 1.0
            equity_scores.append(equity_fold)

        specificity = float(np.mean(specificities))
        equity      = float(np.mean(equity_scores))

    except Exception:
        return {"recall": 0.0, "specificity": 0.0, "f1": 0.0, "accuracy": 0.0,
                "equity": 0.0, "fitness": 0.0}

    fitness = compute_fitness(recall, specificity, f1, equity)
    return {
        "recall":      round(recall, 6),
        "specificity": round(specificity, 6),
        "f1":          round(f1, 6),
        "accuracy":    round(accuracy, 6),
        "equity":      round(equity, 6),
        "fitness":     round(fitness, 6),
    }


def evaluate_full(
    genes: List[float],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
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

    # Equidade no conjunto de teste: recall por quartil de mean radius
    quartiles = np.percentile(X_test[:, 0], [25, 50, 75])
    bounds = [(-np.inf, quartiles[0]), (quartiles[0], quartiles[1]),
              (quartiles[1], quartiles[2]), (quartiles[2], np.inf)]
    group_recalls = []
    for lo, hi in bounds:
        mask = (X_test[:, 0] > lo) & (X_test[:, 0] <= hi)
        if mask.sum() >= 2 and y_test[mask].sum() > 0:
            from sklearn.metrics import recall_score as _recall_score
            group_recalls.append(_recall_score(y_test[mask], y_pred[mask], zero_division=0))
    equity = max(0.0, 1.0 - float(np.std(group_recalls))) if len(group_recalls) >= 2 else 1.0

    return {
        "hyperparams":      hyperparams,
        "recall":           round(recall, 4),
        "specificity":      round(specificity, 4),
        "f1":               round(f1, 4),
        "accuracy":         round(accuracy, 4),
        "equity":           round(equity, 4),
        "fitness":          round(compute_fitness(recall, specificity, f1, equity), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "report":           classification_report(y_test, y_pred, target_names=["Benigno", "Maligno"]),
        "pipeline":         pipeline,
    }


# =============================================================================
# PARTE 3 — ALGORITMO GENÉTICO
# =============================================================================

@dataclass
class Individual:
    genes:   List[float]
    fitness: float = -1.0

    def __lt__(self, other: "Individual") -> bool:
        return self.fitness > other.fitness  # ordenação decrescente (maximizar)

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness:.4f}, {genes_to_str(self.genes)})"


@dataclass
class GAConfig:
    population_size: int   = 50
    n_generations:   int   = 30
    mutation_rate:   float = 0.10
    mutation_sigma:  float = 0.15
    crossover_rate:  float = 0.85
    tournament_size: int   = 3
    elitism_size:    int   = 2
    cv_folds:        int   = 5
    seed:            int   = 42


@dataclass
class GAResult:
    best_individual:   Individual
    best_fitness_hist: List[float]
    avg_fitness_hist:  List[float]
    execution_time_s:  float
    config:            GAConfig

    def summary(self) -> str:
        hp = decode(self.best_individual.genes)
        pca_str = f"PCA({hp['pca_variance']:.2f})" if hp["use_pca"] else "sem PCA"
        lines = [
            "=" * 58,
            " RESULTADO DO ALGORITMO GENÉTICO",
            "=" * 58,
            f"  Pop={self.config.population_size} | Gerações={self.config.n_generations} | Mutação={self.config.mutation_rate}",
            f"  Melhor fitness : {self.best_individual.fitness:.4f}",
            f"  Fitness inicial: {self.best_fitness_hist[0]:.4f}",
            f"  Melhoria       : {(self.best_individual.fitness - self.best_fitness_hist[0]):.4f}",
            f"  Tempo          : {self.execution_time_s:.1f}s",
            f"  Hiperparâmetros ótimos:",
            f"    Scaler   : {hp['scaler']}",
            f"    PCA      : {pca_str}",
            f"    C        : {hp['C']:.4f}",
            f"    Solver   : {hp['solver']}",
            f"    max_iter : {hp['max_iter']}",
            "=" * 58,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------

def uniform_crossover(parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
    """
    Crossover uniforme: cada gene é herdado de um dos pais com prob 0.5.
    Adequado para cromossomos reais com genes independentes entre si.
    """
    child1, child2 = [], []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(g1); child2.append(g2)
        else:
            child1.append(g2); child2.append(g1)
    return child1, child2


def arithmetic_crossover(parent1: List[float], parent2: List[float], alpha: float = 0.5) -> Tuple[List[float], List[float]]:
    """
    Crossover aritmético: filhos são combinações lineares dos pais.
    Útil para genes contínuos como C e pca_variance.
    """
    child1 = [alpha * g1 + (1 - alpha) * g2 for g1, g2 in zip(parent1, parent2)]
    child2 = [alpha * g2 + (1 - alpha) * g1 for g1, g2 in zip(parent1, parent2)]
    return clip_genes(child1), clip_genes(child2)


def gaussian_mutation(genes: List[float], mutation_rate: float, sigma: float) -> List[float]:
    """
    Mutação gaussiana: perturba cada gene com probabilidade mutation_rate.
    Perturbação: N(0, sigma), depois clipa em [0, 1].
    """
    mutated = genes[:]
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] += random.gauss(0, sigma)
    return clip_genes(mutated)


def tournament_selection(population: List[Individual], tournament_size: int) -> Individual:
    """Seleciona o melhor indivíduo entre tournament_size candidatos aleatórios."""
    contestants = random.sample(population, min(tournament_size, len(population)))
    return max(contestants, key=lambda ind: ind.fitness)


# ---------------------------------------------------------------------------
# Loop principal do AG
# ---------------------------------------------------------------------------

def _eval(genes, X, y, cv_folds) -> Individual:
    metrics = evaluate_genes(genes, X, y, cv_folds)
    return Individual(genes=genes, fitness=metrics["fitness"])


def run_genetic_algorithm(
    config: GAConfig,
    X=None,
    y=None,
    verbose: bool = True,
    callback: Optional[Callable[[int, float], None]] = None,
) -> GAResult:
    """
    Executa o Algoritmo Genético para otimização de hiperparâmetros.

    Fluxo por geração:
      1. Elitismo → copia os melhores direto
      2. Seleção por torneio
      3. Crossover uniforme ou aritmético (alternado)
      4. Mutação gaussiana
      5. Avaliação → fitness
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    start = time.time()

    if X is None or y is None:
        X, y = load_data()

    if verbose:
        print(f"\n{'='*58}")
        print(f" AG — Pop={config.population_size} | Ger={config.n_generations} | Mut={config.mutation_rate}")
        print(f"{'='*58}")

    # Inicialização: 1 baseline + restante aleatório
    rng = random.Random(config.seed)
    population: List[Individual] = [_eval(baseline_genes(), X, y, config.cv_folds)]
    for _ in range(config.population_size - 1):
        population.append(_eval([rng.random() for _ in range(N_GENES)], X, y, config.cv_folds))
    population.sort()

    best_fitness_hist: List[float] = []
    avg_fitness_hist:  List[float] = []

    for gen in range(config.n_generations):
        # Elitismo
        new_pop = copy.deepcopy(population[:config.elitism_size])

        while len(new_pop) < config.population_size:
            p1 = tournament_selection(population, config.tournament_size)
            p2 = tournament_selection(population, config.tournament_size)

            if random.random() < config.crossover_rate:
                if random.random() < 0.5:
                    c1g, c2g = uniform_crossover(p1.genes, p2.genes)
                else:
                    c1g, c2g = arithmetic_crossover(p1.genes, p2.genes, random.uniform(0.3, 0.7))
            else:
                c1g, c2g = p1.genes[:], p2.genes[:]

            c1g = gaussian_mutation(c1g, config.mutation_rate, config.mutation_sigma)
            c2g = gaussian_mutation(c2g, config.mutation_rate, config.mutation_sigma)

            new_pop.append(_eval(c1g, X, y, config.cv_folds))
            if len(new_pop) < config.population_size:
                new_pop.append(_eval(c2g, X, y, config.cv_folds))

        population = sorted(new_pop)
        best_f = population[0].fitness
        avg_f  = sum(i.fitness for i in population) / len(population)
        best_fitness_hist.append(best_f)
        avg_fitness_hist.append(avg_f)

        if callback:
            callback(gen, best_f)

        if verbose and (gen % 5 == 0 or gen == config.n_generations - 1):
            print(f"  Gen {gen+1:>3}/{config.n_generations} | Melhor: {best_f:.4f} | Média: {avg_f:.4f} | {genes_to_str(population[0].genes)}")

    result = GAResult(
        best_individual=population[0],
        best_fitness_hist=best_fitness_hist,
        avg_fitness_hist=avg_fitness_hist,
        execution_time_s=round(time.time() - start, 2),
        config=config,
    )
    if verbose:
        print(result.summary())
    return result
