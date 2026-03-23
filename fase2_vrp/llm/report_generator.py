"""
report_generator.py
Geração de documentos via Google Gemini API.

Outputs:
  1. Manual de instruções operacionais para a equipe de transporte
  2. Roteiro narrativo de visitas (lista técnica → briefing humanizado)
  3. Análise comparativa dos experimentos do AG

Configuração: defina GEMINI_API_KEY no arquivo .env ou como variável de ambiente.
"""

import os
import json
from typing import List, Optional, Dict, Any

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .prompts import (
    SYSTEM_CONTEXT,
    MANUAL_PROMPT,
    ROTEIRO_PROMPT,
    EXPERIMENT_ANALYSIS_PROMPT,
)
from ..src.data_generator import ServicePoint, TYPE_LABELS
from ..src.constraints import _route_distance, compute_arrival_times


# ---------------------------------------------------------------------------
# Inicialização do cliente Gemini
# ---------------------------------------------------------------------------

def _get_gemini_model(model_name: str = "gemini-2.0-flash"):
    """
    Inicializa e retorna o modelo Gemini.

    Lê a API key de GEMINI_API_KEY (env var ou .env).
    """
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "google-generativeai não está instalado. "
            "Execute: pip install google-generativeai"
        )

    # Tenta carregar .env se python-dotenv estiver disponível
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY não encontrada. "
            "Defina no arquivo .env ou como variável de ambiente."
        )

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


# ---------------------------------------------------------------------------
# Utilitários de formatação
# ---------------------------------------------------------------------------

def _decimal_to_hhmm(hour: float) -> str:
    h = int(hour)
    m = int((hour - h) * 60)
    return f"{h:02d}:{m:02d}"


def _build_route_summary(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
) -> str:
    """Monta string de resumo da rota para inserção nos prompts."""
    schedule = compute_arrival_times(route, points, dist_matrix)
    schedule_map = {pid: (arr, dep) for pid, arr, dep in schedule}
    point_map = {p.id: p for p in points}
    total_dist = _route_distance(route, dist_matrix)
    total_demand = sum(point_map[pid].demand for pid in route)

    lines = [
        f"Total de paradas: {len(route)}",
        f"Distância total: {total_dist:.1f} km",
        f"Demanda total de suprimentos: {total_demand} unidades",
        "",
    ]

    for i, pid in enumerate(route, 1):
        pt = point_map[pid]
        arr, dep = schedule_map.get(pid, (None, None))
        arr_str = _decimal_to_hhmm(arr) if arr else "--"
        dep_str = _decimal_to_hhmm(dep) if dep else "--"
        tw = f"{int(pt.time_window[0]):02d}:00–{int(pt.time_window[1]):02d}:00"
        label = TYPE_LABELS[pt.type]
        urgency = " ⚡" if pt.priority <= 2 else ""
        lines.append(
            f"  {i:>2}. [{label}{urgency}] {pt.name}"
            f"\n      Chegada: {arr_str} | Saída: {dep_str}"
            f" | Janela: {tw} | Demanda: {pt.demand} un."
        )

    return "\n".join(lines)


def _build_stop_list(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
) -> str:
    """Monta lista de paradas para o roteiro narrativo."""
    schedule = compute_arrival_times(route, points, dist_matrix)
    schedule_map = {pid: (arr, dep) for pid, arr, dep in schedule}
    point_map = {p.id: p for p in points}

    lines = []
    for i, pid in enumerate(route, 1):
        pt = point_map[pid]
        arr, dep = schedule_map.get(pid, (None, None))
        arr_str = _decimal_to_hhmm(arr) if arr else "a confirmar"
        dep_str = _decimal_to_hhmm(dep) if dep else "a confirmar"
        label = TYPE_LABELS[pt.type]
        urgency = " [URGENTE]" if pt.priority <= 2 else ""

        lines.append(
            f"Parada {i}: {pt.name}\n"
            f"  Tipo: {label}{urgency}\n"
            f"  Chegada prevista: {arr_str} | Saída: {dep_str}\n"
            f"  Tempo de serviço: {int(pt.service_time)} min\n"
            f"  Janela permitida: {int(pt.time_window[0]):02d}:00 – "
            f"{int(pt.time_window[1]):02d}:00\n"
        )

    return "\n".join(lines)


def _call_gemini(model, prompt: str) -> str:
    """
    Chama a API do Gemini e retorna o texto gerado.
    Em caso de erro, retorna mensagem amigável.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Erro ao chamar a API Gemini: {e}]"


# ---------------------------------------------------------------------------
# 1. Geração do Manual de Instruções
# ---------------------------------------------------------------------------

def generate_operations_manual(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
) -> str:
    """
    Gera o Manual de Instruções Operacionais para a equipe de transporte.

    Args:
        route:        ordem de visita (IDs)
        points:       lista de ServicePoint
        dist_matrix:  matriz de distâncias
        output_path:  se fornecido, salva o manual em arquivo .txt/.md
        model_name:   versão do Gemini a usar

    Returns:
        Texto do manual gerado.
    """
    model = _get_gemini_model(model_name)
    route_summary = _build_route_summary(route, points, dist_matrix)

    prompt = MANUAL_PROMPT.substitute(
        system_context=SYSTEM_CONTEXT,
        route_summary=route_summary,
    )

    print("[report_generator] Gerando manual de instruções via Gemini...")
    manual_text = _call_gemini(model, prompt)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(manual_text)
        print(f"[report_generator] Manual salvo em: {output_path}")

    return manual_text


# ---------------------------------------------------------------------------
# 2. Geração do Roteiro Narrativo
# ---------------------------------------------------------------------------

def generate_visit_itinerary(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
) -> str:
    """
    Transforma a sequência técnica de paradas em um roteiro narrativo legível.

    Args:
        route:        ordem de visita (IDs)
        points:       lista de ServicePoint
        dist_matrix:  matriz de distâncias
        output_path:  se fornecido, salva em arquivo

    Returns:
        Texto do roteiro gerado.
    """
    model = _get_gemini_model(model_name)
    point_map = {p.id: p for p in points}
    stop_list = _build_stop_list(route, points, dist_matrix)
    total_dist = _route_distance(route, dist_matrix)
    total_demand = sum(point_map[pid].demand for pid in route)

    prompt = ROTEIRO_PROMPT.substitute(
        system_context=SYSTEM_CONTEXT,
        stop_list=stop_list,
        n_stops=len(route),
        total_dist=f"{total_dist:.1f}",
        total_demand=total_demand,
    )

    print("[report_generator] Gerando roteiro narrativo via Gemini...")
    itinerary_text = _call_gemini(model, prompt)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(itinerary_text)
        print(f"[report_generator] Roteiro salvo em: {output_path}")

    return itinerary_text


# ---------------------------------------------------------------------------
# 3. Análise comparativa dos experimentos
# ---------------------------------------------------------------------------

def generate_experiment_analysis(
    experiments: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
) -> str:
    """
    Gera análise comparativa dos 3 experimentos do AG via Gemini.

    Args:
        experiments: lista de dicionários com chaves:
            - config: GAConfig
            - best_fitness: float
            - initial_fitness: float
            - improvement_pct: float
            - execution_time_s: float
            - best_route_dist: float
        output_path: se fornecido, salva em arquivo

    Returns:
        Texto da análise.
    """
    model = _get_gemini_model(model_name)

    def _fmt_exp(exp: Dict[str, Any]) -> str:
        cfg = exp["config"]
        return (
            f"  Melhor fitness: {exp['best_fitness']:.2f}\n"
            f"  Fitness inicial: {exp['initial_fitness']:.2f}\n"
            f"  Melhoria: {exp['improvement_pct']:.1f}%\n"
            f"  Distância da melhor rota: {exp['best_route_dist']:.1f} km\n"
            f"  Tempo de execução: {exp['execution_time_s']:.2f}s"
        )

    exps = experiments + [{}] * (3 - len(experiments))  # pad para 3

    prompt = EXPERIMENT_ANALYSIS_PROMPT.substitute(
        system_context=SYSTEM_CONTEXT,
        exp1_results=_fmt_exp(exps[0]) if exps[0] else "Não executado",
        exp2_results=_fmt_exp(exps[1]) if len(exps) > 1 and exps[1] else "Não executado",
        exp3_results=_fmt_exp(exps[2]) if len(exps) > 2 and exps[2] else "Não executado",
    )

    print("[report_generator] Gerando análise comparativa via Gemini...")
    analysis_text = _call_gemini(model, prompt)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(analysis_text)
        print(f"[report_generator] Análise salva em: {output_path}")

    return analysis_text
