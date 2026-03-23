"""
route_visualizer.py
Visualização interativa de rotas com Folium.

Gera um mapa HTML com:
  - Marcadores coloridos por tipo de atendimento
  - Linha da rota otimizada
  - Popups com informações de cada parada (tipo, prioridade, horário)
  - Legenda com categorias de atendimento
"""

import os
from typing import List, Optional, Tuple
import folium
from folium import plugins

from .data_generator import ServicePoint, Depot, TYPE_LABELS, PRIORITY_MAP
from .constraints import compute_arrival_times, _route_distance


# ---------------------------------------------------------------------------
# Paleta de cores por tipo de atendimento
# ---------------------------------------------------------------------------

TYPE_COLORS = {
    "emergencia_obstetrica": "#E63946",   # vermelho — máxima urgência
    "violencia_domestica":   "#FF7F50",   # coral — alta urgência
    "medicamento_hormonal":  "#2196F3",   # azul — moderado
    "pos_parto":             "#4CAF50",   # verde — rotineiro
}

PRIORITY_ICONS = {
    1: "exclamation-triangle",   # emergência
    2: "shield-alt",             # violência
    3: "capsules",               # medicamento
    4: "baby",                   # pós-parto
}

DEPOT_COLOR = "#9C27B0"  # roxo


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _decimal_to_hhmm(hour_decimal: float) -> str:
    """Converte hora decimal para string 'HH:MM'."""
    h = int(hour_decimal)
    m = int((hour_decimal - h) * 60)
    return f"{h:02d}:{m:02d}"


def _make_popup_html(
    pt: ServicePoint,
    arrival: Optional[float],
    departure: Optional[float],
    stop_number: int,
) -> str:
    """Gera HTML do popup para um ponto de atendimento."""
    color = TYPE_COLORS[pt.type]
    label = TYPE_LABELS[pt.type]
    arrival_str   = _decimal_to_hhmm(arrival)   if arrival   else "--"
    departure_str = _decimal_to_hhmm(departure) if departure else "--"
    tw_start, tw_end = pt.time_window

    return f"""
    <div style="font-family: Arial, sans-serif; min-width: 220px;">
        <div style="background:{color}; color:white; padding:6px 10px;
                    border-radius:4px 4px 0 0; font-weight:bold;">
            Parada {stop_number} — {label}
        </div>
        <div style="padding:8px 10px; border:1px solid {color};
                    border-top:none; border-radius:0 0 4px 4px;">
            <b>{pt.name}</b><br>
            <hr style="margin:4px 0;">
            <span style="color:#555;">Prioridade:</span> {pt.priority} / 4<br>
            <span style="color:#555;">Janela de tempo:</span>
                {_decimal_to_hhmm(tw_start)} – {_decimal_to_hhmm(tw_end)}<br>
            <span style="color:#555;">Chegada prevista:</span> {arrival_str}<br>
            <span style="color:#555;">Saída prevista:</span> {departure_str}<br>
            <span style="color:#555;">Demanda:</span> {pt.demand} unid.<br>
            <span style="color:#555;">Tempo de atendimento:</span>
                {int(pt.service_time)} min
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Função principal de visualização
# ---------------------------------------------------------------------------

def create_route_map(
    depot: Depot,
    points: List[ServicePoint],
    route: List[int],
    dist_matrix: List[List[float]],
    output_path: str = "results/route_map.html",
    title: str = "Rota Otimizada — Saúde da Mulher",
) -> str:
    """
    Cria o mapa interativo com a rota otimizada.

    Args:
        depot:        ponto de partida/chegada
        points:       lista de ServicePoint
        route:        ordem de visita (lista de IDs)
        dist_matrix:  matriz de distâncias
        output_path:  caminho para salvar o arquivo HTML
        title:        título exibido no mapa

    Returns:
        Caminho absoluto do arquivo HTML gerado.
    """
    point_map = {p.id: p for p in points}

    # Calcula horários simulados
    schedule = compute_arrival_times(route, points, dist_matrix)
    schedule_map = {pid: (arr, dep) for pid, arr, dep in schedule}

    # Centro do mapa: média das coordenadas
    all_lats = [depot.lat] + [p.lat for p in points]
    all_lons = [depot.lon] + [p.lon for p in points]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    # Mapa base
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    # --- Depósito ---
    folium.Marker(
        location=[depot.lat, depot.lon],
        popup=folium.Popup(
            f"<b>DEPÓSITO</b><br>{depot.name}<br>"
            f"Saída: {_decimal_to_hhmm(depot.time_window[0])}",
            max_width=250,
        ),
        tooltip="Depósito Central",
        icon=folium.Icon(color="purple", icon="home", prefix="fa"),
    ).add_to(m)

    # --- Pontos de atendimento ---
    for stop_num, pid in enumerate(route, start=1):
        pt = point_map[pid]
        color = TYPE_COLORS[pt.type]
        arrival, departure = schedule_map.get(pid, (None, None))

        popup_html = _make_popup_html(pt, arrival, departure, stop_num)
        tooltip_text = f"{stop_num}. {pt.name} ({TYPE_LABELS[pt.type]})"

        # Círculo colorido (mais visível que ícone padrão)
        folium.CircleMarker(
            location=[pt.lat, pt.lon],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=tooltip_text,
        ).add_to(m)

        # Número da parada
        folium.Marker(
            location=[pt.lat, pt.lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:9px; font-weight:bold; '
                     f'color:white; text-align:center; '
                     f'margin-top:3px;">{stop_num}</div>',
                icon_size=(20, 20),
                icon_anchor=(10, 10),
            ),
        ).add_to(m)

    # --- Linha da rota ---
    route_coords = (
        [(depot.lat, depot.lon)]
        + [(point_map[pid].lat, point_map[pid].lon) for pid in route]
        + [(depot.lat, depot.lon)]
    )

    folium.PolyLine(
        locations=route_coords,
        color="#333333",
        weight=2.5,
        opacity=0.7,
        dash_array="5,8",
        tooltip="Rota otimizada",
    ).add_to(m)

    # Setas de direção (mini-marcadores nos segmentos)
    for i in range(len(route_coords) - 1):
        mid_lat = (route_coords[i][0] + route_coords[i+1][0]) / 2
        mid_lon = (route_coords[i][1] + route_coords[i+1][1]) / 2
        folium.Marker(
            location=[mid_lat, mid_lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px; color:#555;">→</div>',
                icon_size=(14, 14),
                icon_anchor=(7, 7),
            ),
        ).add_to(m)

    # --- Legenda ---
    total_dist = _route_distance(route, dist_matrix)
    legend_html = _build_legend_html(route, points, dist_matrix, total_dist, title)
    m.get_root().html.add_child(folium.Element(legend_html))

    # --- Salvar ---
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    m.save(output_path)
    abs_path = os.path.abspath(output_path)
    print(f"\n[route_visualizer] Mapa salvo em: {abs_path}")
    return abs_path


def _build_legend_html(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
    total_dist: float,
    title: str,
) -> str:
    """Gera HTML da legenda sobreposta ao mapa."""
    from collections import Counter
    point_map = {p.id: p for p in points}
    type_count = Counter(point_map[pid].type for pid in route)

    rows = ""
    for stype, color in TYPE_COLORS.items():
        count = type_count.get(stype, 0)
        if count > 0:
            label = TYPE_LABELS[stype]
            rows += (
                f'<tr><td><span style="color:{color}; font-size:16px;">&#9679;</span></td>'
                f'<td style="padding:2px 6px;">{label}</td>'
                f'<td style="text-align:right;">{count}</td></tr>'
            )

    return f"""
    <div style="position:fixed; bottom:30px; left:20px; z-index:1000;
                background:white; padding:12px 16px; border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.3);
                font-family:Arial, sans-serif; font-size:13px; max-width:260px;">
        <b style="font-size:14px;">{title}</b><hr style="margin:6px 0;">
        <table style="width:100%; border-collapse:collapse;">
            {rows}
        </table>
        <hr style="margin:6px 0;">
        <div style="color:#555;">
            <b>Paradas:</b> {len(route)}<br>
            <b>Distância total:</b> {total_dist:.1f} km
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Comparação de múltiplas rotas (para notebooks)
# ---------------------------------------------------------------------------

def create_comparison_map(
    depot: Depot,
    points: List[ServicePoint],
    routes: List[Tuple[str, List[int], float]],  # (label, route, fitness)
    dist_matrix: List[List[float]],
    output_path: str = "results/comparison_map.html",
) -> str:
    """
    Cria mapa com múltiplas rotas sobrepostas (para comparar experimentos).

    Args:
        routes: lista de (label, route, fitness_value)
    """
    all_lats = [depot.lat] + [p.lat for p in points]
    all_lons = [depot.lon] + [p.lon for p in points]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles="CartoDB positron")

    route_colors = ["#E63946", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    point_map = {p.id: p for p in points}

    for idx, (label, route, fitness) in enumerate(routes):
        color = route_colors[idx % len(route_colors)]
        coords = (
            [(depot.lat, depot.lon)]
            + [(point_map[pid].lat, point_map[pid].lon) for pid in route]
            + [(depot.lat, depot.lon)]
        )
        dist = _route_distance(route, dist_matrix)
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=3,
            opacity=0.8,
            tooltip=f"{label} | fitness={fitness:.1f} | dist={dist:.1f}km",
        ).add_to(m)

    # Pontos
    for pt in points:
        folium.CircleMarker(
            location=[pt.lat, pt.lon],
            radius=6,
            color=TYPE_COLORS[pt.type],
            fill=True,
            fill_color=TYPE_COLORS[pt.type],
            fill_opacity=0.9,
            tooltip=f"{pt.name} ({TYPE_LABELS[pt.type]})",
        ).add_to(m)

    folium.Marker(
        location=[depot.lat, depot.lon],
        icon=folium.Icon(color="purple", icon="home", prefix="fa"),
        tooltip="Depósito Central",
    ).add_to(m)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    m.save(output_path)
    return os.path.abspath(output_path)
