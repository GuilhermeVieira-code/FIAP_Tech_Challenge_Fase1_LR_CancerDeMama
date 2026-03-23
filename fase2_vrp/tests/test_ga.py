"""
test_ga.py
Testes unitários para o Algoritmo Genético.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import random

from fase2_vrp.src.genetic_algorithm import (
    Individual,
    GAConfig,
    _ox_crossover,
    _swap_mutation,
    _inversion_mutation,
    mutate,
    tournament_selection,
    initialize_population,
    run_genetic_algorithm,
)
from fase2_vrp.src.data_generator import (
    generate_service_points,
    get_depot,
    build_distance_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_points():
    return generate_service_points(n_points=20, seed=0)


@pytest.fixture
def depot():
    return get_depot()


@pytest.fixture
def dist_matrix(small_points, depot):
    return build_distance_matrix(depot, small_points)


@pytest.fixture
def point_ids(small_points):
    return [p.id for p in small_points]


# ---------------------------------------------------------------------------
# Testes de operadores genéticos
# ---------------------------------------------------------------------------

class TestOXCrossover:
    def test_output_length(self, point_ids):
        random.seed(1)
        c1, c2 = _ox_crossover(point_ids, point_ids[::-1])
        assert len(c1) == len(point_ids)
        assert len(c2) == len(point_ids)

    def test_is_permutation(self, point_ids):
        random.seed(2)
        c1, c2 = _ox_crossover(point_ids, point_ids[::-1])
        assert sorted(c1) == sorted(point_ids), "Filho 1 não é permutação válida"
        assert sorted(c2) == sorted(point_ids), "Filho 2 não é permutação válida"

    def test_no_duplicates(self, point_ids):
        random.seed(3)
        for _ in range(20):
            c1, c2 = _ox_crossover(point_ids, point_ids[::-1])
            assert len(set(c1)) == len(point_ids)
            assert len(set(c2)) == len(point_ids)


class TestMutation:
    def test_swap_preserves_length(self, point_ids):
        result = _swap_mutation(point_ids)
        assert len(result) == len(point_ids)

    def test_swap_preserves_elements(self, point_ids):
        result = _swap_mutation(point_ids)
        assert sorted(result) == sorted(point_ids)

    def test_inversion_preserves_length(self, point_ids):
        result = _inversion_mutation(point_ids)
        assert len(result) == len(point_ids)

    def test_inversion_preserves_elements(self, point_ids):
        result = _inversion_mutation(point_ids)
        assert sorted(result) == sorted(point_ids)

    def test_mutate_zero_rate(self, point_ids):
        result = mutate(point_ids[:], mutation_rate=0.0)
        assert result == point_ids

    def test_mutate_full_rate(self, point_ids):
        random.seed(99)
        result = mutate(point_ids[:], mutation_rate=1.0)
        assert sorted(result) == sorted(point_ids)


# ---------------------------------------------------------------------------
# Testes de seleção por torneio
# ---------------------------------------------------------------------------

class TestTournamentSelection:
    def test_returns_individual(self, small_points, dist_matrix, point_ids):
        config = GAConfig(population_size=10, n_generations=1, seed=0)
        pop = initialize_population(config, point_ids, small_points, dist_matrix)
        winner = tournament_selection(pop, tournament_size=3)
        assert isinstance(winner, Individual)

    def test_winner_has_valid_route(self, small_points, dist_matrix, point_ids):
        config = GAConfig(population_size=10, n_generations=1, seed=0)
        pop = initialize_population(config, point_ids, small_points, dist_matrix)
        winner = tournament_selection(pop, tournament_size=3)
        assert sorted(winner.chromosome) == sorted(point_ids)

    def test_prefers_lower_fitness(self, small_points, dist_matrix, point_ids):
        """O torneio deve preferir indivíduos com fitness menor."""
        good = Individual(chromosome=point_ids[:], fitness=10.0)
        bad  = Individual(chromosome=point_ids[::-1], fitness=9999.0)
        # Com torneio de tamanho 2 incluindo o bom, o bom deve vencer sempre
        pop = [good, bad]
        winners = [tournament_selection(pop, tournament_size=2) for _ in range(50)]
        assert all(w.fitness == 10.0 for w in winners)


# ---------------------------------------------------------------------------
# Testes de execução completa
# ---------------------------------------------------------------------------

class TestRunGA:
    def test_result_is_valid_permutation(self, small_points, dist_matrix, point_ids):
        config = GAConfig(population_size=20, n_generations=5, seed=42)
        result = run_genetic_algorithm(small_points, dist_matrix, config, verbose=False)
        best_route = result.best_individual.chromosome
        assert sorted(best_route) == sorted(point_ids)

    def test_fitness_decreases_or_stays(self, small_points, dist_matrix):
        config = GAConfig(population_size=30, n_generations=15, seed=42)
        result = run_genetic_algorithm(small_points, dist_matrix, config, verbose=False)
        # Melhor fitness final deve ser <= melhor fitness inicial (elitismo garante isso)
        assert result.best_individual.fitness <= result.best_fitness_hist[0]

    def test_history_length(self, small_points, dist_matrix):
        n_gen = 10
        config = GAConfig(population_size=20, n_generations=n_gen, seed=42)
        result = run_genetic_algorithm(small_points, dist_matrix, config, verbose=False)
        assert len(result.best_fitness_hist) == n_gen
        assert len(result.avg_fitness_hist) == n_gen

    def test_execution_time_recorded(self, small_points, dist_matrix):
        config = GAConfig(population_size=20, n_generations=5, seed=42)
        result = run_genetic_algorithm(small_points, dist_matrix, config, verbose=False)
        assert result.execution_time_s > 0

    def test_three_experiments_configs(self, small_points, dist_matrix):
        """Verifica que os 3 experimentos rodam com as configurações corretas."""
        configs = [
            GAConfig(population_size=50,  n_generations=30, mutation_rate=0.10, seed=42),
            GAConfig(population_size=100, n_generations=50, mutation_rate=0.05, seed=42),
            GAConfig(population_size=200, n_generations=50, mutation_rate=0.15, seed=42),
        ]
        for cfg in configs:
            result = run_genetic_algorithm(small_points, dist_matrix, cfg, verbose=False)
            assert result.best_individual.fitness < float("inf")
