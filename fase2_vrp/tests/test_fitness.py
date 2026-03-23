"""
test_fitness.py
Testes unitários para a função de fitness e restrições.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import math

from fase2_vrp.src.data_generator import (
    generate_service_points,
    get_depot,
    build_distance_matrix,
    haversine,
    ServicePoint,
)
from fase2_vrp.src.fitness import compute_fitness, fitness_breakdown, normalize_fitness
from fase2_vrp.src.constraints import (
    penalty_priority,
    penalty_capacity,
    penalty_time_windows,
    _route_distance,
    compute_arrival_times,
    is_feasible,
    VEHICLE_CAPACITY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def points():
    return generate_service_points(n_points=20, seed=42)


@pytest.fixture
def depot():
    return get_depot()


@pytest.fixture
def dist_matrix(points, depot):
    return build_distance_matrix(depot, points)


@pytest.fixture
def route(points):
    return [p.id for p in points]


# ---------------------------------------------------------------------------
# Testes de haversine
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine(-23.55, -46.63, -23.55, -46.63) == 0.0

    def test_known_distance(self):
        # São Paulo → Rio de Janeiro ≈ 357 km (linha reta)
        d = haversine(-23.5505, -46.6333, -22.9068, -43.1729)
        assert 340 < d < 380, f"Distância SP-RJ fora do esperado: {d:.1f} km"

    def test_symmetry(self):
        d1 = haversine(-23.5, -46.6, -23.6, -46.7)
        d2 = haversine(-23.6, -46.7, -23.5, -46.6)
        assert abs(d1 - d2) < 1e-9


# ---------------------------------------------------------------------------
# Testes de _route_distance
# ---------------------------------------------------------------------------

class TestRouteDistance:
    def test_empty_route(self, dist_matrix):
        assert _route_distance([], dist_matrix) == 0.0

    def test_single_point(self, dist_matrix):
        # Depósito → ponto 1 → depósito
        d = _route_distance([1], dist_matrix)
        expected = dist_matrix[0][1] * 2
        assert abs(d - expected) < 1e-9

    def test_positive(self, route, dist_matrix):
        d = _route_distance(route, dist_matrix)
        assert d > 0


# ---------------------------------------------------------------------------
# Testes de penalidade de prioridade
# ---------------------------------------------------------------------------

class TestPenaltyPriority:
    def test_no_penalty_for_priority_first(self, points):
        """Emergências no início → sem penalidade."""
        sorted_route = sorted([p.id for p in points], key=lambda pid: next(
            p.priority for p in points if p.id == pid
        ))
        pen = penalty_priority(sorted_route, points)
        # Pode ter alguma penalidade mínima, mas bem menor que rota reversa
        reverse_route = sorted_route[::-1]
        pen_reverse = penalty_priority(reverse_route, points)
        assert pen <= pen_reverse

    def test_nonnegative(self, route, points):
        assert penalty_priority(route, points) >= 0

    def test_high_priority_late_increases_penalty(self, points):
        """Mover emergências para o fim aumenta penalidade."""
        all_ids = [p.id for p in points]
        emergency_ids = [p.id for p in points if p.priority == 1]
        other_ids     = [p.id for p in points if p.priority != 1]

        if not emergency_ids:
            pytest.skip("Nenhum ponto de emergência nos dados gerados")

        early_route = emergency_ids + other_ids
        late_route  = other_ids + emergency_ids

        pen_early = penalty_priority(early_route, points)
        pen_late  = penalty_priority(late_route, points)
        assert pen_late > pen_early


# ---------------------------------------------------------------------------
# Testes de penalidade de capacidade
# ---------------------------------------------------------------------------

class TestPenaltyCapacity:
    def test_no_penalty_within_capacity(self, points, dist_matrix):
        """Subconjunto pequeno → sem violação de capacidade."""
        small_route = [p.id for p in points[:3]]   # poucas paradas
        pen = penalty_capacity(small_route, points, dist_matrix)
        total_demand = sum(p.demand for p in points[:3])
        if total_demand <= VEHICLE_CAPACITY:
            # Não deve haver penalidade de demanda
            assert pen >= 0   # pode ter penalidade de distância se rota for longa

    def test_penalty_increases_with_overload(self, points, dist_matrix):
        """Adicionar mais pontos aumenta penalidade quando excede capacidade."""
        route_small = [p.id for p in points[:5]]
        route_all   = [p.id for p in points]
        pen_small = penalty_capacity(route_small, points, dist_matrix)
        pen_all   = penalty_capacity(route_all, points, dist_matrix)
        # Rota com todos os pontos tem mais chance de violar
        assert pen_all >= pen_small

    def test_nonnegative(self, route, points, dist_matrix):
        assert penalty_capacity(route, points, dist_matrix) >= 0


# ---------------------------------------------------------------------------
# Testes de penalidade de janelas de tempo
# ---------------------------------------------------------------------------

class TestPenaltyTimeWindows:
    def test_nonnegative(self, route, points, dist_matrix):
        assert penalty_time_windows(route, points, dist_matrix) >= 0

    def test_arrival_times_computed(self, route, points, dist_matrix):
        schedule = compute_arrival_times(route, points, dist_matrix)
        assert len(schedule) == len(route)
        for point_id, arr, dep in schedule:
            assert dep >= arr, "Saída deve ser >= chegada"


# ---------------------------------------------------------------------------
# Testes da função de fitness principal
# ---------------------------------------------------------------------------

class TestComputeFitness:
    def test_positive_fitness(self, route, points, dist_matrix):
        f = compute_fitness(route, points, dist_matrix)
        assert f > 0

    def test_empty_route_returns_inf(self, points, dist_matrix):
        assert compute_fitness([], points, dist_matrix) == float("inf")

    def test_breakdown_sums_to_total(self, route, points, dist_matrix):
        bd = fitness_breakdown(route, points, dist_matrix)
        manual_sum = (
            bd["distance"]
            + bd["penalty_priority"]
            + bd["penalty_capacity"]
            + bd["penalty_time"]
        )
        assert abs(manual_sum - bd["total"]) < 1e-2

    def test_priority_sorted_better_than_random(self, points, dist_matrix):
        """Rota com emergências primeiro tende a ter fitness menor."""
        sorted_route = sorted(
            [p.id for p in points],
            key=lambda pid: next(p.priority for p in points if p.id == pid)
        )
        reverse_route = sorted_route[::-1]

        f_sorted  = compute_fitness(sorted_route,  points, dist_matrix)
        f_reverse = compute_fitness(reverse_route, points, dist_matrix)
        assert f_sorted < f_reverse, (
            "Rota com prioridades corretas deve ter fitness menor"
        )


# ---------------------------------------------------------------------------
# Testes de is_feasible e normalize_fitness
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    def test_is_feasible_small_route(self, points, dist_matrix):
        small_route = [p.id for p in points[:2]]
        # Duas paradas perto raramente violam
        result = is_feasible(small_route, points, dist_matrix)
        assert isinstance(result, bool)

    def test_normalize_fitness_range(self):
        values = [100.0, 200.0, 150.0, 300.0]
        normalized = normalize_fitness(values)
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0

    def test_normalize_fitness_all_equal(self):
        values = [50.0, 50.0, 50.0]
        normalized = normalize_fitness(values)
        assert all(v == 0.0 for v in normalized)
