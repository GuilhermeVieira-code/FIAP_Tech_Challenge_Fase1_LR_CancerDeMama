"""
fitness.py
Função de fitness para o Algoritmo Genético do VRP de saúde da mulher.

fitness = distância_total + penalidade_prioridade
                          + penalidade_capacidade
                          + penalidade_janela_tempo

Minimizar (menor = melhor rota).
"""

from typing import List, Dict, Any
from .data_generator import ServicePoint, Depot
from .constraints import (
    penalty_priority,
    penalty_capacity,
    penalty_time_windows,
    _route_distance,
)


# ---------------------------------------------------------------------------
# Pesos relativos das componentes do fitness
# ---------------------------------------------------------------------------

WEIGHT_DISTANCE  = 1.0   # distância base (km)
WEIGHT_PRIORITY  = 1.0   # prioridade de atendimento
WEIGHT_CAPACITY  = 1.0   # capacidade do veículo
WEIGHT_TIME      = 1.0   # janelas de tempo


# ---------------------------------------------------------------------------
# Função principal de fitness
# ---------------------------------------------------------------------------

def compute_fitness(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
) -> float:
    """
    Calcula o fitness de uma rota (cromossomo).

    Args:
        route:       lista de IDs dos pontos na ordem de visita (sem depósito)
        points:      lista de ServicePoint
        dist_matrix: matriz de distâncias [n+1 x n+1], índice 0 = depósito

    Returns:
        Valor de fitness (float). Quanto menor, melhor.
    """
    if not route:
        return float("inf")

    # Componente 1: distância total (km)
    dist = _route_distance(route, dist_matrix)

    # Componente 2: penalidade de prioridade
    pen_prio = penalty_priority(route, points)

    # Componente 3: penalidade de capacidade + distância máxima
    pen_cap = penalty_capacity(route, points, dist_matrix)

    # Componente 4: penalidade de janelas de tempo
    pen_time = penalty_time_windows(route, points, dist_matrix)

    fitness = (
        WEIGHT_DISTANCE * dist
        + WEIGHT_PRIORITY * pen_prio
        + WEIGHT_CAPACITY * pen_cap
        + WEIGHT_TIME    * pen_time
    )

    return fitness


def fitness_breakdown(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
) -> Dict[str, float]:
    """
    Retorna o detalhamento de cada componente do fitness.
    Útil para análise e relatórios.

    Returns:
        Dicionário com chaves: 'distance', 'penalty_priority',
        'penalty_capacity', 'penalty_time', 'total'
    """
    dist     = _route_distance(route, dist_matrix)
    pen_prio = penalty_priority(route, points)
    pen_cap  = penalty_capacity(route, points, dist_matrix)
    pen_time = penalty_time_windows(route, points, dist_matrix)
    total    = compute_fitness(route, points, dist_matrix)

    return {
        "distance":         round(dist, 3),
        "penalty_priority": round(pen_prio, 3),
        "penalty_capacity": round(pen_cap, 3),
        "penalty_time":     round(pen_time, 3),
        "total":            round(total, 3),
    }


# ---------------------------------------------------------------------------
# Utilitário: normalização (para comparação entre experimentos)
# ---------------------------------------------------------------------------

def normalize_fitness(
    fitness_values: List[float],
) -> List[float]:
    """
    Normaliza valores de fitness para [0, 1] para comparação.

    Args:
        fitness_values: lista de valores brutos

    Returns:
        Lista normalizada
    """
    min_f = min(fitness_values)
    max_f = max(fitness_values)
    span = max_f - min_f
    if span == 0:
        return [0.0] * len(fitness_values)
    return [(f - min_f) / span for f in fitness_values]
