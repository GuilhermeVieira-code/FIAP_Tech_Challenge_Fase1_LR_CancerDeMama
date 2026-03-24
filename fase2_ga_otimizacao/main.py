"""
main.py
=======
Pipeline principal da Fase 2 — Otimização de Hiperparâmetros via AG + LLM.

Fluxo:
  1. Carrega os dados da Fase 1 (breast_cancer_dataset.csv)
  2. Avalia o modelo baseline (Fase 1)
  3. Executa 3 experimentos do Algoritmo Genético
  4. Avalia o melhor modelo encontrado no conjunto de teste
  5. Gera gráficos e JSON de resultados
  6. Gera relatórios em linguagem natural via LLM local (opcional)

Uso:
    python -m fase2_ga_otimizacao.main
    python -m fase2_ga_otimizacao.main --skip-llm
    python -m fase2_ga_otimizacao.main --cv-folds 3   # mais rápido para testes
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from fase2_ga_otimizacao.genetic_algorithm import (
    GAConfig, GAResult,
    load_data, evaluate_full, evaluate_genes,
    baseline_genes, decode,
    run_genetic_algorithm,
)

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Configurações dos 3 experimentos
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    GAConfig(population_size=30, n_generations=20, mutation_rate=0.15, mutation_sigma=0.15, cv_folds=5, seed=42),
    GAConfig(population_size=50, n_generations=30, mutation_rate=0.10, mutation_sigma=0.12, cv_folds=5, seed=42),
    GAConfig(population_size=80, n_generations=30, mutation_rate=0.20, mutation_sigma=0.18, cv_folds=5, seed=42),
]
EXPERIMENT_LABELS = [
    "Exp 1 — Pop=30  | Mut=0.15 | 20 gen",
    "Exp 2 — Pop=50  | Mut=0.10 | 30 gen",
    "Exp 3 — Pop=80  | Mut=0.20 | 30 gen",
]


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def plot_convergence(results, labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#E63946", "#2196F3", "#4CAF50"]
    lss    = ["-", "--", "-."]

    for res, label, color, ls in zip(results, labels, colors, lss):
        axes[0].plot(res.best_fitness_hist, label=label, color=color, ls=ls, lw=2)
        axes[1].plot(res.avg_fitness_hist,  label=label, color=color, ls=ls, lw=2, alpha=0.8)

    for ax, title in zip(axes, ["Melhor Fitness", "Fitness Médio"]):
        ax.set_title(f"Convergência — {title}", fontsize=12)
        ax.set_xlabel("Geração")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("AG — Otimização de Hiperparâmetros (Diagnóstico Câncer de Mama)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = str(RESULTS_DIR / "convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[main] Gráfico salvo: {path}")


def plot_metrics_comparison(baseline, best_opt, best_label):
    metrics  = ["Recall", "Especificidade", "F1-score", "Acurácia"]
    b_vals   = [baseline["recall"], baseline["specificity"], baseline["f1"], baseline["accuracy"]]
    opt_vals = [best_opt["recall"], best_opt["specificity"], best_opt["f1"], best_opt["accuracy"]]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, b_vals,   w, label="Baseline Fase 1",         color="#9E9E9E", alpha=0.85)
    ax.bar(x + w/2, opt_vals, w, label=f"Otimizado AG\n({best_label})", color="#2196F3", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor da métrica")
    ax.set_title("Baseline vs. Modelo Otimizado — Diagnóstico Câncer de Mama", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for rect in ax.patches:
        ax.annotate(f"{rect.get_height():.3f}",
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = str(RESULTS_DIR / "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[main] Gráfico salvo: {path}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main(skip_llm: bool = False, cv_folds: int = 5):
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print(" TECH CHALLENGE FASE 2 — Projeto 1")
    print(" Otimização de Hiperparâmetros via AG + LLM")
    print("=" * 60)

    # 1. Dados
    print("\n[1/5] Carregando dados da Fase 1...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 2. Baseline
    print("\n[2/5] Avaliando modelo baseline (Fase 1)...")
    bl_genes = baseline_genes()
    baseline = evaluate_full(bl_genes, X_train, X_test, y_train, y_test)
    print(f"      Recall={baseline['recall']} | F1={baseline['f1']} | "
          f"Acc={baseline['accuracy']} | Fitness={baseline['fitness']}")
    print(f"\n{baseline['report']}")

    # 3. Experimentos do AG
    print("\n[3/5] Executando 3 experimentos do AG...")
    ga_results = []
    for i, (cfg, label) in enumerate(zip(EXPERIMENTS, EXPERIMENT_LABELS), 1):
        cfg.cv_folds = cv_folds
        print(f"\n  ► Experimento {i}: {label}")
        result = run_genetic_algorithm(cfg, X_train, y_train, verbose=True)
        ga_results.append(result)

    best_ga    = max(ga_results, key=lambda r: r.best_individual.fitness)
    best_label = EXPERIMENT_LABELS[ga_results.index(best_ga)]
    print(f"\n  ★ Melhor solução global: {best_label}")

    # 4. Avaliação final no conjunto de teste
    print("\n[4/5] Avaliando melhor modelo no conjunto de teste...")
    best_opt = evaluate_full(best_ga.best_individual.genes, X_train, X_test, y_train, y_test)
    print(f"      Recall={best_opt['recall']} | F1={best_opt['f1']} | "
          f"Acc={best_opt['accuracy']} | Fitness={best_opt['fitness']}")
    print(f"\n{best_opt['report']}")

    plot_convergence(ga_results, EXPERIMENT_LABELS)
    plot_metrics_comparison(baseline, best_opt, best_label)

    # Salvar JSON
    results_data = {
        "baseline":       {k: v for k, v in baseline.items() if k not in ("pipeline", "report")},
        "best_optimized": {k: v for k, v in best_opt.items() if k not in ("pipeline", "report")},
        "experiments":    [],
    }
    for res, label in zip(ga_results, EXPERIMENT_LABELS):
        cv_metrics = evaluate_genes(res.best_individual.genes, X_train, y_train, cv_folds)
        results_data["experiments"].append({
            "label":            label,
            "config":           {"population_size": res.config.population_size,
                                 "n_generations":   res.config.n_generations,
                                 "mutation_rate":   res.config.mutation_rate},
            "best_fitness":     res.best_individual.fitness,
            "best_hyperparams": decode(res.best_individual.genes),
            "cv_metrics":       cv_metrics,
            "execution_time_s": res.execution_time_s,
        })

    json_path = str(RESULTS_DIR / "experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n[main] Resultados JSON: {json_path}")

    # 5. LLM (Hugging Face local)
    if skip_llm:
        print("\n[5/5] Etapa LLM ignorada (--skip-llm)")
    else:
        print("\n[5/5] Gerando relatórios via LLM local (Hugging Face)...")
        try:
            from fase2_ga_otimizacao.llm import explain_diagnosis, compare_models, analyze_experiments

            # Explicação de diagnóstico para o primeiro paciente do teste
            pipeline = best_opt["pipeline"]
            sample   = X_test[0:1]
            pred     = int(pipeline.predict(sample)[0])
            proba    = pipeline.predict_proba(sample)[0][1]

            clf = pipeline.named_steps["clf"]
            if "pca" in pipeline.named_steps:
                feature_names = [f"PC{i+1}" for i in range(clf.coef_.shape[1])]
            else:
                import pandas as pd
                df_tmp = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "breast_cancer_dataset.csv"))
                feature_names = [c for c in df_tmp.columns if c not in ("id", "Unnamed: 32", "diagnosis")]

            top_features = {name: float(coef) for name, coef in zip(feature_names, clf.coef_[0])}

            explain_diagnosis(
                classification=pred, probability_malignant=float(proba),
                recall=best_opt["recall"], specificity=best_opt["specificity"],
                f1_score=best_opt["f1"], top_features=top_features,
                output_path=str(RESULTS_DIR / "explicacao_diagnostico.txt"),
            )

            compare_models(
                baseline_metrics=baseline,
                optimized_metrics=best_opt,
                output_path=str(RESULTS_DIR / "comparacao_modelos.txt"),
            )

            exp_summary = [
                {
                    "config":           r.config,
                    "best_fitness":     r.best_individual.fitness,
                    **evaluate_genes(r.best_individual.genes, X_train, y_train),
                    "execution_time_s": r.execution_time_s,
                }
                for r in ga_results
            ]
            analyze_experiments(exp_summary, output_path=str(RESULTS_DIR / "analise_experimentos.txt"))

        except Exception as e:
            print(f"[LLM] Erro: {e}")
            print("[LLM] Verifique se 'transformers torch sentencepiece' estão instalados ou use --skip-llm")

    # Resumo final
    print("\n" + "=" * 60)
    print(" EXECUÇÃO CONCLUÍDA")
    print("=" * 60)
    delta_recall = best_opt["recall"] - baseline["recall"]
    delta_f1     = best_opt["f1"]     - baseline["f1"]
    print(f"  Baseline  → Recall: {baseline['recall']} | F1: {baseline['f1']}")
    print(f"  Otimizado → Recall: {best_opt['recall']} | F1: {best_opt['f1']}")
    print(f"  Melhoria  → ΔRecall: {delta_recall:+.4f} | ΔF1: {delta_f1:+.4f}")
    print(f"\n  Resultados em: {RESULTS_DIR}/")
    print(f"  - convergence.png         (curvas de convergência)")
    print(f"  - metrics_comparison.png  (baseline vs. otimizado)")
    print(f"  - experiment_results.json (métricas completas)")
    if not skip_llm:
        print(f"  - explicacao_diagnostico.txt")
        print(f"  - comparacao_modelos.txt")
        print(f"  - analise_experimentos.txt")
        print(f"  - llm_responses/          (JSONs para Fase 3)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 2 — Otimização de Hiperparâmetros via AG")
    parser.add_argument("--skip-llm", action="store_true", help="Pula a etapa LLM local")
    parser.add_argument("--cv-folds", type=int, default=5, help="Folds de validação cruzada (use 3 para testes rápidos)")
    args = parser.parse_args()
    main(skip_llm=args.skip_llm, cv_folds=args.cv_folds)
