"""
main.py
Pipeline principal da Fase 2 — VRP para Saúde da Mulher.

Fluxo:
  1. Gera dados sintéticos (pontos de atendimento em São Paulo)
  2. Executa 3 experimentos do Algoritmo Genético
  3. Plota o gráfico de convergência
  4. Gera mapa interativo Folium da melhor rota
  5. Gera documentos via Gemini (manual + roteiro)
  6. Executa perguntas de demonstração ao chat

Uso:
    python main.py                          # executa tudo (LLM opcional)
    python main.py --skip-llm              # pula etapas que chamam a API Gemini
    python main.py --n-points 25           # número de pontos de atendimento
    python main.py --seed 123              # semente para reprodutibilidade
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # sem GUI — funciona em ambientes sem display
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Garantir imports relativos funcionem ao executar como script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from fase2_vrp.src.data_generator import (
    generate_service_points,
    get_depot,
    build_distance_matrix,
    summarize_points,
)
from fase2_vrp.src.genetic_algorithm import GAConfig, run_genetic_algorithm
from fase2_vrp.src.fitness import fitness_breakdown
from fase2_vrp.src.constraints import _route_distance
from fase2_vrp.src.route_visualizer import create_route_map, create_comparison_map


# ---------------------------------------------------------------------------
# Configurações dos 3 experimentos obrigatórios
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    GAConfig(population_size=50,  n_generations=30, mutation_rate=0.10, seed=42, elitism_size=2),
    GAConfig(population_size=100, n_generations=50, mutation_rate=0.05, seed=42, elitism_size=2),
    GAConfig(population_size=200, n_generations=50, mutation_rate=0.15, seed=42, elitism_size=2),
]

EXPERIMENT_LABELS = [
    "Exp 1 — Pop=50  | Mut=0.10 | 30 gen",
    "Exp 2 — Pop=100 | Mut=0.05 | 50 gen",
    "Exp 3 — Pop=200 | Mut=0.15 | 50 gen",
]

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _ensure_results_dir():
    RESULTS_DIR.mkdir(exist_ok=True)


def plot_convergence(results, labels, output_path: str):
    """Plota as curvas de convergência dos 3 experimentos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#E63946", "#2196F3", "#4CAF50"]
    linestyles = ["-", "--", "-."]

    # Painel esquerdo: melhor fitness por geração
    for res, label, color, ls in zip(results, labels, colors, linestyles):
        axes[0].plot(res.best_fitness_hist, label=label, color=color,
                     linestyle=ls, linewidth=2)
    axes[0].set_title("Convergência — Melhor Fitness por Geração", fontsize=12)
    axes[0].set_xlabel("Geração")
    axes[0].set_ylabel("Fitness (minimizar)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Painel direito: fitness médio por geração
    for res, label, color, ls in zip(results, labels, colors, linestyles):
        axes[1].plot(res.avg_fitness_hist, label=label, color=color,
                     linestyle=ls, linewidth=2, alpha=0.8)
    axes[1].set_title("Convergência — Fitness Médio por Geração", fontsize=12)
    axes[1].set_xlabel("Geração")
    axes[1].set_ylabel("Fitness médio")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Algoritmo Genético — Otimização de Rotas de Saúde da Mulher",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[main] Gráfico de convergência salvo em: {output_path}")


def save_results_json(results, points, dist_matrix, labels, output_path: str):
    """Salva os resultados dos experimentos em JSON."""
    data = []
    for res, label in zip(results, labels):
        route = res.best_individual.chromosome
        breakdown = fitness_breakdown(route, points, dist_matrix)
        exp_data = {
            "label": label,
            "config": {
                "population_size": res.config.population_size,
                "n_generations":   res.config.n_generations,
                "mutation_rate":   res.config.mutation_rate,
            },
            "best_fitness":    round(res.best_individual.fitness, 4),
            "initial_fitness": round(res.best_fitness_hist[0], 4),
            "improvement_pct": round(
                (1 - res.best_individual.fitness / res.best_fitness_hist[0]) * 100, 2
            ),
            "execution_time_s": res.execution_time_s,
            "best_route":       route,
            "fitness_breakdown": breakdown,
        }
        data.append(exp_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[main] Resultados JSON salvos em: {output_path}")
    return data


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main(n_points: int = 20, skip_llm: bool = False, seed: int = 42):
    _ensure_results_dir()

    print("\n" + "=" * 60)
    print(" TECH CHALLENGE FASE 2 — VRP SAÚDE DA MULHER")
    print(" Algoritmo Genético + LLM (Gemini)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # ETAPA 1: Geração de dados
    # ------------------------------------------------------------------
    print("\n[1/5] Gerando dados sintéticos...")
    depot  = get_depot()
    points = generate_service_points(n_points=n_points, seed=seed)
    summarize_points(points)
    dist_matrix = build_distance_matrix(depot, points)

    # ------------------------------------------------------------------
    # ETAPA 2: Execução dos 3 experimentos do AG
    # ------------------------------------------------------------------
    print("\n[2/5] Executando experimentos do Algoritmo Genético...")
    ga_results = []
    for i, (config, label) in enumerate(zip(EXPERIMENTS, EXPERIMENT_LABELS), 1):
        print(f"\n  ► Experimento {i}: {label}")
        result = run_genetic_algorithm(points, dist_matrix, config, verbose=True)
        ga_results.append(result)

    # Identifica melhor resultado geral
    best_result = min(ga_results, key=lambda r: r.best_individual.fitness)
    best_route  = best_result.best_individual.chromosome
    best_label  = EXPERIMENT_LABELS[ga_results.index(best_result)]
    print(f"\n  ★ Melhor solução: {best_label}")
    print(f"    Fitness: {best_result.best_individual.fitness:.2f}")
    print(f"    Distância: {_route_distance(best_route, dist_matrix):.1f} km")

    # ------------------------------------------------------------------
    # ETAPA 3: Gráfico de convergência
    # ------------------------------------------------------------------
    print("\n[3/5] Gerando gráfico de convergência...")
    convergence_path = str(RESULTS_DIR / "convergence.png")
    plot_convergence(ga_results, EXPERIMENT_LABELS, convergence_path)

    # Salva resultados em JSON
    json_path = str(RESULTS_DIR / "experiment_results.json")
    exp_data  = save_results_json(ga_results, points, dist_matrix, EXPERIMENT_LABELS, json_path)

    # ------------------------------------------------------------------
    # ETAPA 4: Mapas interativos
    # ------------------------------------------------------------------
    print("\n[4/5] Gerando mapas interativos (Folium)...")

    # Mapa da melhor rota
    map_path = str(RESULTS_DIR / "route_map.html")
    create_route_map(depot, points, best_route, dist_matrix, output_path=map_path)

    # Mapa de comparação dos 3 experimentos
    comparison_data = [
        (EXPERIMENT_LABELS[i], r.best_individual.chromosome, r.best_individual.fitness)
        for i, r in enumerate(ga_results)
    ]
    comparison_path = str(RESULTS_DIR / "comparison_map.html")
    create_comparison_map(depot, points, comparison_data, dist_matrix, output_path=comparison_path)

    # ------------------------------------------------------------------
    # ETAPA 5: Geração de documentos via Gemini (LLM)
    # ------------------------------------------------------------------
    if skip_llm:
        print("\n[5/5] Etapa LLM ignorada (--skip-llm ativo)")
    else:
        print("\n[5/5] Gerando documentos e chat via Gemini...")
        try:
            from fase2_vrp.llm.report_generator import (
                generate_operations_manual,
                generate_visit_itinerary,
                generate_experiment_analysis,
            )
            from fase2_vrp.llm.route_chat import run_demo_questions

            # Manual de instruções
            generate_operations_manual(
                best_route, points, dist_matrix,
                output_path=str(RESULTS_DIR / "manual_instrucoes.txt"),
            )

            # Roteiro narrativo
            generate_visit_itinerary(
                best_route, points, dist_matrix,
                output_path=str(RESULTS_DIR / "roteiro_visitas.txt"),
            )

            # Análise dos experimentos
            exp_summary = [
                {
                    "config": r.config,
                    "best_fitness":     r.best_individual.fitness,
                    "initial_fitness":  r.best_fitness_hist[0],
                    "improvement_pct":  (1 - r.best_individual.fitness / r.best_fitness_hist[0]) * 100,
                    "execution_time_s": r.execution_time_s,
                    "best_route_dist":  _route_distance(r.best_individual.chromosome, dist_matrix),
                }
                for r in ga_results
            ]
            generate_experiment_analysis(
                exp_summary,
                output_path=str(RESULTS_DIR / "analise_experimentos.txt"),
            )

            # Demo do chat
            run_demo_questions(best_route, points, dist_matrix)

        except Exception as e:
            print(f"[LLM] Erro: {e}")
            print("[LLM] Dica: configure GEMINI_API_KEY no .env ou use --skip-llm")

    # ------------------------------------------------------------------
    # Resumo final
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" EXECUÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"  Resultados em: {RESULTS_DIR}/")
    print(f"  - convergence.png          (gráfico de convergência)")
    print(f"  - experiment_results.json  (métricas dos experimentos)")
    print(f"  - route_map.html           (mapa interativo da melhor rota)")
    print(f"  - comparison_map.html      (comparação dos 3 experimentos)")
    if not skip_llm:
        print(f"  - manual_instrucoes.txt    (manual da equipe — Gemini)")
        print(f"  - roteiro_visitas.txt      (roteiro narrativo — Gemini)")
        print(f"  - analise_experimentos.txt (análise dos experimentos — Gemini)")
    print("=" * 60 + "\n")

    return {
        "best_route":    best_route,
        "best_fitness":  best_result.best_individual.fitness,
        "ga_results":    ga_results,
        "points":        points,
        "dist_matrix":   dist_matrix,
        "depot":         depot,
    }


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VRP Saúde da Mulher — Algoritmo Genético + Gemini"
    )
    parser.add_argument("--n-points", type=int,   default=20,    help="Número de pontos de atendimento (20-25)")
    parser.add_argument("--skip-llm", action="store_true",        help="Pula as etapas que chamam a API Gemini")
    parser.add_argument("--seed",     type=int,   default=42,    help="Semente para reprodutibilidade")
    args = parser.parse_args()

    main(n_points=args.n_points, skip_llm=args.skip_llm, seed=args.seed)
