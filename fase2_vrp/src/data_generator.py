"""
data_generator.py
Geração de dados sintéticos para o VRP de saúde da mulher em São Paulo.

Cada ponto representa um local de atendimento especializado à mulher:
  - emergencia_obstetrica  (prioridade 1 — mais urgente)
  - violencia_domestica    (prioridade 2)
  - medicamento_hormonal   (prioridade 3)
  - pos_parto              (prioridade 4)
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import random
import math


# ---------------------------------------------------------------------------
# Estrutura de dados
# ---------------------------------------------------------------------------

@dataclass
class ServicePoint:
    """Representa um ponto de atendimento."""
    id: int
    name: str
    type: str                          # tipo de atendimento
    priority: int                      # 1 (maior) a 4 (menor)
    lat: float
    lon: float
    time_window: Tuple[float, float]   # (hora_inicio, hora_fim) em horas decimais
    demand: int                        # unidades de suprimentos necessárias
    service_time: float                # tempo de atendimento em minutos

    def __repr__(self):
        return (f"ServicePoint(id={self.id}, name='{self.name}', "
                f"type='{self.type}', priority={self.priority})")


@dataclass
class Depot:
    """Ponto de partida e chegada da rota."""
    id: int = 0
    name: str = "Almoxarifado Central – Saúde da Mulher"
    lat: float = -23.5489
    lon: float = -46.6388   # Próximo à Av. Paulista, SP
    time_window: Tuple[float, float] = field(default=(7.0, 19.0))


# ---------------------------------------------------------------------------
# Dados base: bairros reais de São Paulo com coordenadas
# ---------------------------------------------------------------------------

NEIGHBORHOODS = [
    ("UBS Pinheiros",         -23.5660, -46.6986, "pos_parto",             (8.0, 17.0), 15, 20),
    ("UBS Vila Madalena",     -23.5536, -46.6917, "medicamento_hormonal",  (8.0, 16.0), 20, 25),
    ("CAPS Lapa",             -23.5102, -46.7072, "violencia_domestica",   (7.0, 13.0), 10, 30),
    ("Maternidade Moema",     -23.6062, -46.6641, "emergencia_obstetrica", (6.0, 18.0), 30, 45),
    ("UBS Perdizes",          -23.5368, -46.6641, "medicamento_hormonal",  (9.0, 17.0), 15, 20),
    ("UBS Ipiranga",          -23.5896, -46.6083, "pos_parto",             (8.0, 16.0), 20, 25),
    ("Casa da Mulher – Sé",   -23.5476, -46.6311, "violencia_domestica",   (7.0, 12.0), 10, 30),
    ("Maternidade Santana",   -23.5024, -46.6272, "emergencia_obstetrica", (6.0, 18.0), 35, 45),
    ("UBS Tatuapé",           -23.5452, -46.5744, "medicamento_hormonal",  (8.0, 16.0), 20, 25),
    ("UBS Vila Prudente",     -23.5891, -46.5643, "pos_parto",             (9.0, 17.0), 15, 20),
    ("CAPS Butantã",          -23.5586, -46.7230, "violencia_domestica",   (7.0, 13.0), 10, 30),
    ("Maternidade Campo Limpo",-23.6534,-46.7792, "emergencia_obstetrica", (6.0, 18.0), 40, 45),
    ("UBS Penha",             -23.5279, -46.5389, "medicamento_hormonal",  (8.0, 16.0), 20, 25),
    ("UBS Ermelino",          -23.4862, -46.4881, "pos_parto",             (9.0, 17.0), 15, 20),
    ("Casa da Mulher – Itaim",-23.5871, -46.6771, "violencia_domestica",   (7.0, 12.0), 10, 30),
    ("UBS Jabaquara",         -23.6469, -46.6489, "pos_parto",             (8.0, 16.0), 25, 20),
    ("Maternidade Grajaú",    -23.7029, -46.6941, "emergencia_obstetrica", (6.0, 18.0), 40, 45),
    ("UBS Santo André",       -23.6731, -46.5346, "medicamento_hormonal",  (8.0, 16.0), 20, 25),
    ("UBS Osasco",            -23.5329, -46.7917, "pos_parto",             (9.0, 17.0), 15, 20),
    ("CAPS São Mateus",       -23.6126, -46.4906, "violencia_domestica",   (7.0, 13.0), 10, 30),
    ("UBS Carapicuíba",       -23.5231, -46.8401, "medicamento_hormonal",  (8.0, 16.0), 20, 25),
    ("Maternidade Guarulhos",  -23.4648, -46.5337, "emergencia_obstetrica",(6.0, 18.0), 35, 45),
    ("UBS Diadema",           -23.6808, -46.6228, "pos_parto",             (9.0, 17.0), 15, 20),
    ("UBS Mauá",              -23.6680, -46.4620, "medicamento_hormonal",  (8.0, 16.0), 20, 25),
    ("Casa da Mulher – ABC",  -23.6897, -46.5648, "violencia_domestica",   (7.0, 12.0), 10, 30),
]

PRIORITY_MAP = {
    "emergencia_obstetrica": 1,
    "violencia_domestica":   2,
    "medicamento_hormonal":  3,
    "pos_parto":             4,
}

TYPE_LABELS = {
    "emergencia_obstetrica": "Emergência Obstétrica",
    "violencia_domestica":   "Violência Doméstica",
    "medicamento_hormonal":  "Medicamento Hormonal",
    "pos_parto":             "Pós-Parto",
}


# ---------------------------------------------------------------------------
# Geração de dados
# ---------------------------------------------------------------------------

def generate_service_points(
    n_points: int = 20,
    seed: int = 42
) -> List[ServicePoint]:
    """
    Gera uma lista de pontos de atendimento sintéticos.

    Args:
        n_points: número de pontos (entre 20 e 25)
        seed: semente para reprodutibilidade

    Returns:
        Lista de ServicePoint
    """
    random.seed(seed)
    n_points = max(20, min(n_points, len(NEIGHBORHOODS)))

    selected = random.sample(NEIGHBORHOODS, n_points)
    points: List[ServicePoint] = []

    for idx, (name, lat, lon, stype, tw, demand, svc_time) in enumerate(selected, start=1):
        # Adiciona leve variação nas coordenadas para realismo
        lat_jitter = random.uniform(-0.003, 0.003)
        lon_jitter = random.uniform(-0.003, 0.003)
        # Variação de demanda
        demand_var = demand + random.randint(-3, 5)

        points.append(ServicePoint(
            id=idx,
            name=name,
            type=stype,
            priority=PRIORITY_MAP[stype],
            lat=round(lat + lat_jitter, 6),
            lon=round(lon + lon_jitter, 6),
            time_window=tw,
            demand=max(5, demand_var),
            service_time=float(svc_time),
        ))

    # Garante pelo menos uma emergência obstétrica
    types_present = {p.type for p in points}
    if "emergencia_obstetrica" not in types_present:
        points[0] = ServicePoint(
            id=points[0].id,
            name="Maternidade Emergência (extra)",
            type="emergencia_obstetrica",
            priority=1,
            lat=-23.5600,
            lon=-46.6500,
            time_window=(6.0, 18.0),
            demand=35,
            service_time=45.0,
        )

    return points


def get_depot() -> Depot:
    """Retorna o depósito central."""
    return Depot()


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula a distância em km entre dois pontos (lat/lon) usando a fórmula de Haversine.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def build_distance_matrix(
    depot: Depot,
    points: List[ServicePoint]
) -> List[List[float]]:
    """
    Constrói a matriz de distâncias (km) entre todos os nós.
    Índice 0 = depósito; índices 1..n = pontos de atendimento.

    Returns:
        Matriz n+1 x n+1 de floats.
    """
    all_nodes = [(depot.lat, depot.lon)] + [(p.lat, p.lon) for p in points]
    n = len(all_nodes)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(all_nodes[i][0], all_nodes[i][1],
                          all_nodes[j][0], all_nodes[j][1])
            matrix[i][j] = d
            matrix[j][i] = d

    return matrix


def summarize_points(points: List[ServicePoint]) -> None:
    """Imprime um resumo dos pontos gerados."""
    from collections import Counter
    type_count = Counter(p.type for p in points)
    total_demand = sum(p.demand for p in points)

    print(f"\n{'='*60}")
    print(f" PONTOS DE ATENDIMENTO GERADOS: {len(points)}")
    print(f"{'='*60}")
    for stype, count in sorted(type_count.items(), key=lambda x: PRIORITY_MAP[x[0]]):
        label = TYPE_LABELS[stype]
        prio = PRIORITY_MAP[stype]
        print(f"  Prioridade {prio} — {label}: {count} pontos")
    print(f"\n  Demanda total de suprimentos: {total_demand} unidades")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    depot = get_depot()
    points = generate_service_points(n_points=20)
    summarize_points(points)

    dist_matrix = build_distance_matrix(depot, points)
    print(f"Matriz de distâncias: {len(dist_matrix)}x{len(dist_matrix[0])} nós")
    print(f"Exemplo: depósito → ponto 1 = {dist_matrix[0][1]:.2f} km")
