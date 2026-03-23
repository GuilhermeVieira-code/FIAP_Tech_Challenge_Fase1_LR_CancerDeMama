"""
test_ga.py
Testes unitários para o Algoritmo Genético.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import random

from fase2_ga_otimizacao.src.chromosome import (
    N_GENES, decode, random_genes, baseline_genes, clip_genes,
    SCALERS, SOLVERS, PCA_VAR_MIN, PCA_VAR_MAX,
)
from fase2_ga_otimizacao.src.genetic_algorithm import (
    Individual, GAConfig,
    uniform_crossover, arithmetic_crossover,
    gaussian_mutation, tournament_selection,
    run_genetic_algorithm,
)
from fase2_ga_otimizacao.src.model_evaluator import load_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    return load_data()


# ---------------------------------------------------------------------------
# Testes de cromossomo e decodificação
# ---------------------------------------------------------------------------

class TestChromosome:
    def test_random_genes_length(self):
        g = random_genes()
        assert len(g) == N_GENES

    def test_random_genes_in_range(self):
        for _ in range(20):
            g = random_genes()
            assert all(0.0 <= x <= 1.0 for x in g), "Genes fora de [0, 1]"

    def test_baseline_genes_decode(self):
        g = baseline_genes()
        hp = decode(g)
        assert hp["scaler"] == "RobustScaler"
        assert hp["use_pca"] is True
        assert abs(hp["pca_variance"] - 0.90) < 0.02
        assert hp["solver"] == "lbfgs"

    def test_decode_scaler_valid(self):
        for _ in range(30):
            g = random_genes()
            hp = decode(g)
            assert hp["scaler"] in SCALERS

    def test_decode_solver_valid(self):
        for _ in range(30):
            g = random_genes()
            hp = decode(g)
            assert hp["solver"] in SOLVERS

    def test_decode_pca_var_in_range(self):
        for _ in range(30):
            g = random_genes()
            hp = decode(g)
            assert PCA_VAR_MIN <= hp["pca_variance"] <= PCA_VAR_MAX

    def test_decode_C_positive(self):
        for _ in range(30):
            g = random_genes()
            hp = decode(g)
            assert hp["C"] > 0

    def test_clip_genes(self):
        g = [1.5, -0.3, 0.5, 1.1, 0.0, -0.1]
        clipped = clip_genes(g)
        assert all(0.0 <= x <= 1.0 for x in clipped)


# ---------------------------------------------------------------------------
# Testes de operadores
# ---------------------------------------------------------------------------

class TestCrossover:
    def test_uniform_length(self):
        p1 = random_genes()
        p2 = random_genes()
        c1, c2 = uniform_crossover(p1, p2)
        assert len(c1) == N_GENES
        assert len(c2) == N_GENES

    def test_uniform_values_from_parents(self):
        p1 = random_genes()
        p2 = random_genes()
        c1, c2 = uniform_crossover(p1, p2)
        for i in range(N_GENES):
            assert c1[i] in (p1[i], p2[i])
            assert c2[i] in (p1[i], p2[i])

    def test_arithmetic_length(self):
        p1 = random_genes()
        p2 = random_genes()
        c1, c2 = arithmetic_crossover(p1, p2)
        assert len(c1) == N_GENES
        assert len(c2) == N_GENES

    def test_arithmetic_in_range(self):
        for _ in range(20):
            p1, p2 = random_genes(), random_genes()
            c1, c2 = arithmetic_crossover(p1, p2)
            assert all(0.0 <= x <= 1.0 for x in c1)
            assert all(0.0 <= x <= 1.0 for x in c2)


class TestMutation:
    def test_gaussian_length(self):
        g = random_genes()
        m = gaussian_mutation(g, 1.0, 0.1)
        assert len(m) == N_GENES

    def test_gaussian_in_range(self):
        for _ in range(50):
            g = random_genes()
            m = gaussian_mutation(g, 1.0, 0.5)
            assert all(0.0 <= x <= 1.0 for x in m)

    def test_zero_rate_no_change(self):
        g = random_genes()
        m = gaussian_mutation(g[:], 0.0, 0.5)
        assert m == g


class TestTournament:
    def test_returns_individual(self):
        pop = [Individual(genes=random_genes(), fitness=random.random()) for _ in range(10)]
        winner = tournament_selection(pop, 3)
        assert isinstance(winner, Individual)

    def test_prefers_higher_fitness(self):
        best = Individual(genes=random_genes(), fitness=1.0)
        rest = [Individual(genes=random_genes(), fitness=0.1) for _ in range(9)]
        pop  = [best] + rest
        winners = [tournament_selection(pop, 5) for _ in range(30)]
        assert all(w.fitness >= 0.1 for w in winners)


# ---------------------------------------------------------------------------
# Testes de execução do AG
# ---------------------------------------------------------------------------

class TestRunGA:
    def test_result_improves_or_stays(self, data):
        X, y = data
        cfg = GAConfig(population_size=10, n_generations=3, cv_folds=3, seed=0)
        result = run_genetic_algorithm(cfg, X, y, verbose=False)
        assert result.best_individual.fitness >= result.best_fitness_hist[0] - 1e-6

    def test_history_length(self, data):
        X, y = data
        n_gen = 4
        cfg = GAConfig(population_size=8, n_generations=n_gen, cv_folds=3, seed=0)
        result = run_genetic_algorithm(cfg, X, y, verbose=False)
        assert len(result.best_fitness_hist) == n_gen
        assert len(result.avg_fitness_hist)  == n_gen

    def test_genes_in_range(self, data):
        X, y = data
        cfg = GAConfig(population_size=8, n_generations=3, cv_folds=3, seed=0)
        result = run_genetic_algorithm(cfg, X, y, verbose=False)
        assert all(0.0 <= g <= 1.0 for g in result.best_individual.genes)

    def test_fitness_positive(self, data):
        X, y = data
        cfg = GAConfig(population_size=8, n_generations=3, cv_folds=3, seed=0)
        result = run_genetic_algorithm(cfg, X, y, verbose=False)
        assert result.best_individual.fitness > 0
