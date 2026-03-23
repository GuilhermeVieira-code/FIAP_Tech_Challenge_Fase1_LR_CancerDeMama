"""
constraints.py
Verificação e cálculo de penalidades para as restrições do VRP de saúde da mulher.

Restrições implementadas:
  1. Prioridade de atendimento  — emergências devem ser visitadas cedo na rota
  2. Capacidade do veículo       — limite de suprimentos e distância máxima
  3. Janelas de tempo            — cada ponto deve ser visitado dentro do horário seguro
"""

from typing import List, Tuple
from .data_generator import ServicePoint, Depot, haversine

# ---------------------------------------------------------------------------
# Parâmetros globais de restrição
# ---------------------------------------------------------------------------

VEHICLE_CAPACITY    = 120    # unidades de suprimento máximas por veículo
MAX_ROUTE_DISTANCE  = 200.0  # km máximos por rota
AVG_SPEED_KMH       = 30.0   # velocidade média em São Paulo (tráfego urbano)
DEPOT_OPEN_HOUR     = 7.0    # hora de abertura do depósito
DEPOT_CLOSE_HOUR    = 19.0   # hora de fechamento do depósito

# Pesos das penalidades (ajuste fino via experimentação)
W_PRIORITY   = 50.0   # penalidade por ponto prioritário visitado tarde
W_CAPACITY   = 500.0  # penalidade por exceder capacidade
W_DISTANCE   = 300.0  # penalidade por exceder distância máxima
W_TIME_EARLY = 20.0   # penalidade por chegar muito cedo (espera)
W_TIME_LATE  = 80.0   # penalidade por chegar fora da janela


# ---------------------------------------------------------------------------
# 1. Restrição de prioridade
# ---------------------------------------------------------------------------

def penalty_priority(
    route: List[int],
    points: List[ServicePoint],
) -> float:
    """
    Penaliza rotas em que pontos de alta prioridade aparecem tarde na sequência.

    Lógica: para cada ponto de prioridade 1 (emergência obstétrica) ou 2
    (violência doméstica), a posição na rota não deveria ultrapassar um limiar.
    Quanto mais tarde na rota, maior a penalidade multiplicada pela urgência.

    Args:
        route: lista de índices dos pontos (ordem de visita), sem o depósito
        points: lista completa de ServicePoint (índice 0 = points[0], id=1)

    Returns:
        Valor de penalidade (float, ≥ 0)
    """
    point_map = {p.id: p for p in points}
    n = len(route)
    penalty = 0.0

    for position, point_id in enumerate(route):
        pt = point_map[point_id]
        relative_position = position / max(n - 1, 1)  # 0.0 (início) a 1.0 (fim)

        if pt.priority == 1:
            # Emergência obstétrica: deve estar no primeiro terço da rota
            threshold = 0.33
            if relative_position > threshold:
                penalty += W_PRIORITY * (relative_position - threshold) * (5 - pt.priority)
        elif pt.priority == 2:
            # Violência doméstica: deve estar na primeira metade
            threshold = 0.50
            if relative_position > threshold:
                penalty += W_PRIORITY * (relative_position - threshold) * (5 - pt.priority)

    return penalty


# ---------------------------------------------------------------------------
# 2. Restrição de capacidade
# ---------------------------------------------------------------------------

def penalty_capacity(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
) -> float:
    """
    Penaliza rotas que excedem a capacidade do veículo ou a distância máxima.

    Args:
        route: lista de índices dos pontos (sem o depósito)
        points: lista de ServicePoint
        dist_matrix: matriz de distâncias (índice 0 = depósito)

    Returns:
        Valor de penalidade (float, ≥ 0)
    """
    point_map = {p.id: p for p in points}
    penalty = 0.0

    # --- capacidade ---
    total_demand = sum(point_map[pid].demand for pid in route)
    if total_demand > VEHICLE_CAPACITY:
        excess = total_demand - VEHICLE_CAPACITY
        penalty += W_CAPACITY * excess

    # --- distância máxima ---
    total_dist = _route_distance(route, dist_matrix)
    if total_dist > MAX_ROUTE_DISTANCE:
        excess_dist = total_dist - MAX_ROUTE_DISTANCE
        penalty += W_DISTANCE * excess_dist

    return penalty


# ---------------------------------------------------------------------------
# 3. Restrição de janelas de tempo
# ---------------------------------------------------------------------------

def penalty_time_windows(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
) -> float:
    """
    Penaliza chegadas fora das janelas de tempo de cada ponto.

    Simulação de horário:
      - Saída do depósito às DEPOT_OPEN_HOUR
      - Tempo de viagem calculado com AVG_SPEED_KMH
      - Ao chegar antes da janela, aguarda (sem penalidade excessiva)
      - Ao chegar depois da janela, aplica penalidade proporcional ao atraso

    Args:
        route: lista de índices dos pontos (sem o depósito)
        points: lista de ServicePoint
        dist_matrix: matriz de distâncias

    Returns:
        Valor de penalidade (float, ≥ 0)
    """
    point_map = {p.id: p for p in points}
    penalty = 0.0
    current_time = DEPOT_OPEN_HOUR   # hora decimal
    current_node = 0                  # depósito = índice 0 na matriz

    for point_id in route:
        pt = point_map[point_id]
        pt_matrix_idx = point_id      # id do ponto == índice na matriz (1-based)

        # Tempo de viagem até este ponto
        dist_km = dist_matrix[current_node][pt_matrix_idx]
        travel_time_h = dist_km / AVG_SPEED_KMH

        arrival = current_time + travel_time_h
        tw_start, tw_end = pt.time_window

        if arrival < tw_start:
            # Chega cedo — espera até a janela abrir (pequena penalidade por tempo ocioso)
            wait = tw_start - arrival
            penalty += W_TIME_EARLY * wait
            current_time = tw_start
        elif arrival > tw_end:
            # Chega tarde — violação de janela
            late = arrival - tw_end
            # Emergências recebem penalidade extra
            urgency_mult = 5 - pt.priority   # 4 para prio=1, 1 para prio=4
            penalty += W_TIME_LATE * late * urgency_mult
            current_time = arrival
        else:
            current_time = arrival

        # Adiciona tempo de serviço no ponto
        current_time += pt.service_time / 60.0
        current_node = pt_matrix_idx

    # Verifica se retorna ao depósito antes do fechamento
    dist_back = dist_matrix[current_node][0]
    return_time = current_time + dist_back / AVG_SPEED_KMH
    if return_time > DEPOT_CLOSE_HOUR:
        penalty += W_TIME_LATE * (return_time - DEPOT_CLOSE_HOUR) * 2

    return penalty


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _route_distance(
    route: List[int],
    dist_matrix: List[List[float]],
) -> float:
    """
    Calcula a distância total da rota (depósito → pontos → depósito).

    Args:
        route: lista de índices dos pontos (sem o depósito, índice 0)
        dist_matrix: matriz completa de distâncias

    Returns:
        Distância total em km
    """
    if not route:
        return 0.0

    total = dist_matrix[0][route[0]]           # depósito → primeiro ponto
    for i in range(len(route) - 1):
        total += dist_matrix[route[i]][route[i + 1]]
    total += dist_matrix[route[-1]][0]         # último ponto → depósito
    return total


def compute_arrival_times(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
) -> List[Tuple[int, float, float]]:
    """
    Retorna o horário simulado de chegada e saída de cada ponto na rota.

    Returns:
        Lista de (point_id, hora_chegada, hora_saida) em horas decimais.
    """
    point_map = {p.id: p for p in points}
    schedule = []
    current_time = DEPOT_OPEN_HOUR
    current_node = 0

    for point_id in route:
        pt = point_map[point_id]
        dist_km = dist_matrix[current_node][point_id]
        travel_time_h = dist_km / AVG_SPEED_KMH
        arrival = current_time + travel_time_h
        tw_start, _ = pt.time_window

        # Espera se chegou antes da janela
        actual_start = max(arrival, tw_start)
        departure = actual_start + pt.service_time / 60.0

        schedule.append((point_id, round(arrival, 3), round(departure, 3)))
        current_time = departure
        current_node = point_id

    return schedule


def is_feasible(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
) -> bool:
    """
    Verifica se uma rota é feasível (sem violações de capacidade/distância).
    Não considera penalidades de tempo (aceita esperas).
    """
    point_map = {p.id: p for p in points}
    total_demand = sum(point_map[pid].demand for pid in route)
    total_dist = _route_distance(route, dist_matrix)
    return total_demand <= VEHICLE_CAPACITY and total_dist <= MAX_ROUTE_DISTANCE
