"""
report_generator.py
Geração de explicações e relatórios via Google Gemini.

Outputs:
  1. Explicação clínica de um resultado de diagnóstico
  2. Análise comparativa baseline vs. modelo otimizado
  3. Análise dos experimentos do AG
  4. Respostas salvas em JSON para uso futuro na Fase 3 (fine-tuning)
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .prompts import (
    SYSTEM_CONTEXT,
    DIAGNOSIS_EXPLANATION_PROMPT,
    MODEL_COMPARISON_PROMPT,
    GA_ANALYSIS_PROMPT,
)

# Pasta para salvar respostas do LLM (Fase 3)
_RESPONSES_DIR = os.path.join(os.path.dirname(__file__), "llm_responses")


# ---------------------------------------------------------------------------
# Inicialização do Gemini
# ---------------------------------------------------------------------------

def _get_model(model_name: str = "gemini-2.0-flash"):
    if not GEMINI_AVAILABLE:
        raise ImportError("Instale: pip install google-generativeai")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY não encontrada. "
            "Configure no .env ou como variável de ambiente."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def _call_llm(model, prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Erro na API Gemini: {e}]"


# ---------------------------------------------------------------------------
# Salvar respostas para Fase 3
# ---------------------------------------------------------------------------

def _save_response(response_type: str, prompt: str, response: str, metadata: Dict = None):
    """
    Salva a resposta do LLM em JSON para uso na Fase 3 (fine-tuning).
    Formato: {prompt, response, type, timestamp, metadata}
    """
    os.makedirs(_RESPONSES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{response_type}_{timestamp}.json"

    record = {
        "type":      response_type,
        "timestamp": datetime.now().isoformat(),
        "prompt":    prompt,
        "response":  response,
        "metadata":  metadata or {},
    }

    path = os.path.join(_RESPONSES_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"[LLM] Resposta salva em: {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Explicação clínica de diagnóstico
# ---------------------------------------------------------------------------

def explain_diagnosis(
    classification: int,
    probability_malignant: float,
    recall: float,
    specificity: float,
    f1_score: float,
    top_features: Dict[str, float],
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
) -> str:
    """
    Gera explicação clínica para o resultado de um diagnóstico.

    Args:
        classification:        0 (benigno) ou 1 (maligno)
        probability_malignant: probabilidade estimada de malignidade (0-1)
        recall, specificity, f1_score: métricas do modelo
        top_features:          dict {feature_name: importance_value}
        output_path:           salvar em arquivo .txt
        model_name:            versão do Gemini

    Returns:
        Texto da explicação gerada.
    """
    model = _get_model(model_name)

    classification_str = "MALIGNO (M)" if classification == 1 else "BENIGNO (B)"
    prob_pct = round(probability_malignant * 100, 1)

    features_str = "\n".join(
        f"  - {name}: {value:.4f}"
        for name, value in sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    )

    prompt = DIAGNOSIS_EXPLANATION_PROMPT.substitute(
        system_context=SYSTEM_CONTEXT,
        classification=classification_str,
        probability_malignant=prob_pct,
        recall=round(recall, 4),
        specificity=round(specificity, 4),
        f1_score=round(f1_score, 4),
        top_features=features_str,
    )

    print("[LLM] Gerando explicação clínica via Gemini...")
    response = _call_llm(model, prompt)

    _save_response("diagnosis_explanation", prompt, response, {
        "classification": classification_str,
        "probability_malignant": prob_pct,
    })

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"[LLM] Explicação salva em: {output_path}")

    return response


# ---------------------------------------------------------------------------
# 2. Comparação baseline vs. otimizado
# ---------------------------------------------------------------------------

def compare_models(
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, Any],
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
) -> str:
    """
    Gera análise comparativa entre o modelo base (Fase 1) e o otimizado pelo AG.
    """
    model = _get_model(model_name)

    hp = optimized_metrics.get("hyperparams", {})
    hp_str = (
        f"{hp.get('scaler', '?')} | "
        f"{'PCA(' + str(hp.get('pca_variance', '')) + ')' if hp.get('use_pca') else 'sem PCA'} | "
        f"C={hp.get('C', '?'):.4f} | {hp.get('solver', '?')}"
    )

    prompt = MODEL_COMPARISON_PROMPT.substitute(
        system_context=SYSTEM_CONTEXT,
        baseline_recall=baseline_metrics.get("recall", "N/A"),
        baseline_specificity=baseline_metrics.get("specificity", "N/A"),
        baseline_f1=baseline_metrics.get("f1", "N/A"),
        baseline_accuracy=baseline_metrics.get("accuracy", "N/A"),
        optimized_hyperparams=hp_str,
        optimized_recall=optimized_metrics.get("recall", "N/A"),
        optimized_specificity=optimized_metrics.get("specificity", "N/A"),
        optimized_f1=optimized_metrics.get("f1", "N/A"),
        optimized_accuracy=optimized_metrics.get("accuracy", "N/A"),
    )

    print("[LLM] Gerando comparação de modelos via Gemini...")
    response = _call_llm(model, prompt)

    _save_response("model_comparison", prompt, response, {
        "baseline": baseline_metrics,
        "optimized": {k: v for k, v in optimized_metrics.items() if k != "pipeline"},
    })

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"[LLM] Comparação salva em: {output_path}")

    return response


# ---------------------------------------------------------------------------
# 3. Análise dos experimentos do AG
# ---------------------------------------------------------------------------

def analyze_experiments(
    experiments: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
) -> str:
    """
    Gera análise técnica dos 3 experimentos do AG.

    Args:
        experiments: lista com dicts {config, best_fitness, recall, f1, specificity, time}
    """
    model = _get_model(model_name)

    def fmt(exp):
        return (
            f"  Melhor fitness : {exp.get('best_fitness', 'N/A'):.4f}\n"
            f"  Recall         : {exp.get('recall', 'N/A'):.4f}\n"
            f"  Especificidade : {exp.get('specificity', 'N/A'):.4f}\n"
            f"  F1-score       : {exp.get('f1', 'N/A'):.4f}\n"
            f"  Tempo          : {exp.get('execution_time_s', 'N/A'):.1f}s"
        )

    exps = experiments + [{}] * (3 - len(experiments))
    cfgs = [e.get("config", GAConfig_placeholder()) for e in exps]

    prompt = GA_ANALYSIS_PROMPT.substitute(
        system_context=SYSTEM_CONTEXT,
        pop1=getattr(cfgs[0], 'population_size', '?'),
        mut1=getattr(cfgs[0], 'mutation_rate', '?'),
        gen1=getattr(cfgs[0], 'n_generations', '?'),
        exp1_results=fmt(exps[0]) if exps[0] else "Não executado",
        pop2=getattr(cfgs[1], 'population_size', '?') if len(cfgs) > 1 else '?',
        mut2=getattr(cfgs[1], 'mutation_rate', '?') if len(cfgs) > 1 else '?',
        gen2=getattr(cfgs[1], 'n_generations', '?') if len(cfgs) > 1 else '?',
        exp2_results=fmt(exps[1]) if len(exps) > 1 and exps[1] else "Não executado",
        pop3=getattr(cfgs[2], 'population_size', '?') if len(cfgs) > 2 else '?',
        mut3=getattr(cfgs[2], 'mutation_rate', '?') if len(cfgs) > 2 else '?',
        gen3=getattr(cfgs[2], 'n_generations', '?') if len(cfgs) > 2 else '?',
        exp3_results=fmt(exps[2]) if len(exps) > 2 and exps[2] else "Não executado",
    )

    print("[LLM] Gerando análise dos experimentos via Gemini...")
    response = _call_llm(model, prompt)

    _save_response("ga_analysis", prompt, response)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)

    return response


class GAConfig_placeholder:
    population_size = "?"
    mutation_rate   = "?"
    n_generations   = "?"
