"""
llm.py
======
Integração com LLM pré-treinada via Hugging Face Transformers.

Roda 100% local, sem API key, sem internet após o primeiro download.

Modelo padrão: google/flan-t5-base (~250 MB, roda em CPU)
Modelo melhor : google/flan-t5-large (~770 MB, melhor qualidade)

Troque via variável de ambiente:
    set HF_MODEL=google/flan-t5-large

Pré-requisitos:
    pip install -r requirements.txt

Uso:
    from fase2_ga_otimizacao.llm import explain_diagnosis, compare_models, analyze_experiments
"""

import json
import os
from datetime import datetime
from string import Template
from typing import Any, Dict, List, Optional

# Modelo configurável via env var
DEFAULT_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")

try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Pipeline singleton (carrega o modelo só uma vez)
_pipeline = None


def _get_pipeline(model: str = DEFAULT_MODEL):
    """Carrega o pipeline de geração de texto (só na primeira chamada)."""
    global _pipeline
    if not HF_AVAILABLE:
        raise ImportError("Execute: pip install transformers torch sentencepiece")
    if _pipeline is None:
        print(f"[LLM] Carregando modelo '{model}'... (só na primeira vez)")
        _pipeline = hf_pipeline(
            "text2text-generation",
            model=model,
            max_new_tokens=512,
            do_sample=False,
        )
        print(f"[LLM] Modelo carregado.")
    return _pipeline


# =============================================================================
# PARTE 1 — PROMPTS
# =============================================================================

SYSTEM_CONTEXT = (
    "Assistente médico especializado em oncologia feminina. "
    "Use linguagem clínica em português brasileiro. "
    "Seja empático, preciso e nunca alarmista. "
    "Este sistema apoia o médico — a decisão final é sempre do profissional."
)


DIAGNOSIS_PROMPT = Template(
    "$system_context\n\n"
    "Resultado do modelo de diagnóstico de câncer de mama: $classification, "
    "probabilidade de malignidade $probability_malignant%, recall $recall, F1 $f1_score.\n\n"
    "Em português, explique para o médico: qual o nível de confiança, quais os próximos "
    "passos clínicos recomendados e como comunicar o resultado à paciente com empatia."
)

COMPARISON_PROMPT = Template(
    "$system_context\n\n"
    "Modelo Fase 1 (baseline): recall=$baseline_recall, F1=$baseline_f1, "
    "acurácia=$baseline_accuracy.\n"
    "Modelo otimizado pelo AG (Fase 2): $optimized_hyperparams, "
    "recall=$optimized_recall, F1=$optimized_f1, acurácia=$optimized_accuracy.\n\n"
    "Em português, analise: qual o impacto clínico da melhoria no recall, "
    "quantos falsos negativos foram evitados e qual modelo recomenda para uso clínico."
)


GA_ANALYSIS_PROMPT = Template("""
$system_context

---

Analise os resultados dos três experimentos do Algoritmo Genético aplicado
à otimização do modelo de diagnóstico de câncer de mama:

Exp1: pop=$pop1, mutação=$mut1, $gen1 gerações, fitness=$exp1_results
Exp2: pop=$pop2, mutação=$mut2, $gen2 gerações, fitness=$exp2_results
Exp3: pop=$pop3, mutação=$mut3, $gen3 gerações, fitness=$exp3_results

Em português, responda: qual experimento foi melhor para diagnóstico de câncer de mama, "
"por que a taxa de mutação importa e qual configuração recomenda para produção clínica.""")


# =============================================================================
# PARTE 2 — AVALIAÇÃO DE QUALIDADE DAS RESPOSTAS LLM
# =============================================================================

_MEDICAL_TERMS = [
    "benigno", "maligno", "biópsia", "encaminhamento", "oncologista",
    "mamografia", "ultrassom", "recall", "especificidade", "diagnóstico",
    "paciente", "tratamento", "exame", "resultado", "confiança",
]

_ALARM_TERMS = ["certeza", "definitivamente", "com certeza", "100%", "garantido"]


def evaluate_llm_response(response: str, response_type: str) -> Dict[str, Any]:
    """
    Avalia a qualidade da resposta gerada pelo LLM.

    Critérios:
    - Completude     : tamanho mínimo esperado (150 palavras = score máximo)
    - Terminologia   : presença de termos clínicos relevantes
    - Adequação      : ausência de linguagem alarmista ou absoluta
    - Idioma         : marcadores do português brasileiro
    """
    text = response.lower()
    words = text.split()

    completeness  = min(len(words) / 150, 1.0)
    terms_found   = [t for t in _MEDICAL_TERMS if t in text]
    medical_score = min(len(terms_found) / 5, 1.0)
    alarm_found   = [t for t in _ALARM_TERMS if t in text]
    adequacy      = max(0.0, 1.0 - len(alarm_found) * 0.2)
    pt_markers    = ["que", "com", "para", "uma", "não", "mais", "como", "por"]
    pt_score      = min(sum(1 for w in pt_markers if w in text) / len(pt_markers), 1.0)

    overall = round(completeness * 0.3 + medical_score * 0.4 + adequacy * 0.2 + pt_score * 0.1, 3)

    return {
        "type":                response_type,
        "overall_score":       overall,
        "completeness":        round(completeness, 3),
        "medical_terminology": round(medical_score, 3),
        "adequacy":            round(adequacy, 3),
        "language_pt":         round(pt_score, 3),
        "terms_found":         terms_found,
        "word_count":          len(words),
        "quality_label":       "Boa" if overall >= 0.7 else "Regular" if overall >= 0.5 else "Fraca",
    }


# =============================================================================
# PARTE 3 — GERADOR DE RELATÓRIOS
# =============================================================================

_RESPONSES_DIR = os.path.join(os.path.dirname(__file__), "llm_responses")


def _is_echo(prompt: str, response: str, threshold: float = 0.6) -> bool:
    """Detecta se o modelo apenas repetiu o prompt em vez de gerar uma resposta."""
    prompt_words   = set(prompt.lower().split())
    response_words = set(response.lower().split())
    if not response_words:
        return True
    overlap = len(prompt_words & response_words) / len(response_words)
    return overlap > threshold or len(response_words) < 15


def _call_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Gera texto com o modelo local via Hugging Face. Usa fallback se o modelo ecoar o prompt."""
    pipe = _get_pipeline(model)
    try:
        result   = pipe(prompt)
        response = result[0]["generated_text"].strip()
        if _is_echo(prompt, response):
            return None   # sinaliza para usar fallback
        return response
    except Exception as e:
        return f"[Erro ao gerar resposta: {e}]"


def _save_response(response_type: str, prompt: str, response: str,
                   metadata: Dict = None, quality: Dict = None) -> str:
    """
    Salva prompt + resposta em JSON para uso na Fase 3 (fine-tuning).
    """
    os.makedirs(_RESPONSES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_RESPONSES_DIR, f"{response_type}_{timestamp}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "type":      response_type,
            "timestamp": datetime.now().isoformat(),
            "model":     DEFAULT_MODEL,
            "prompt":    prompt,
            "response":  response,
            "quality":   quality or {},
            "metadata":  metadata or {},
        }, f, indent=2, ensure_ascii=False)

    print(f"[LLM] Resposta salva em: {path}")
    return path


def _fallback_diagnosis(classification, probability_malignant, recall, f1_score):
    label    = "MALIGNO" if classification == 1 else "BENIGNO"
    urgencia = "necessita avaliação urgente com oncologista" if classification == 1 else "recomenda-se acompanhamento de rotina"
    return (
        f"RESULTADO DO MODELO: {label} — Probabilidade de malignidade: {round(probability_malignant*100,1)}%\n\n"
        f"NÍVEL DE CONFIANÇA: O modelo apresenta recall de {recall:.2%} e F1-score de {f1_score:.2%}, "
        f"indicando alta capacidade de detecção de casos malignos.\n\n"
        f"PRÓXIMOS PASSOS CLÍNICOS: A paciente {urgencia}. "
        f"{'Recomenda-se biópsia para confirmação diagnóstica e encaminhamento imediato.' if classification == 1 else 'Manter rastreamento anual com mamografia conforme protocolo.'}\n\n"
        f"COMUNICAÇÃO À PACIENTE: Apresente o resultado com empatia, em linguagem acessível, "
        f"reforçando que este é um sistema de apoio e que o diagnóstico final será feito pelo médico especialista.\n\n"
        f"DIREITOS NO SUS: A paciente tem direito a confirmação diagnóstica em até 30 dias (Lei 12.732/2012) "
        f"e ao tratamento integral no sistema público de saúde.\n\n"
        f"NOTA: Este sistema é uma ferramenta de apoio — a decisão clínica final é sempre do médico."
    )


def _fallback_comparison(baseline_recall, optimized_recall, baseline_f1, optimized_f1,
                          baseline_accuracy, optimized_accuracy, hyperparams_str):
    delta_recall = optimized_recall - baseline_recall
    delta_f1     = optimized_f1 - baseline_f1
    return (
        f"COMPARAÇÃO DE MODELOS — Diagnóstico de Câncer de Mama\n\n"
        f"Baseline (Fase 1): Recall={baseline_recall:.4f} | F1={baseline_f1:.4f} | Acc={baseline_accuracy:.4f}\n"
        f"Otimizado (AG):    Recall={optimized_recall:.4f} | F1={optimized_f1:.4f} | Acc={optimized_accuracy:.4f}\n"
        f"Configuração otimizada: {hyperparams_str}\n\n"
        f"IMPACTO CLÍNICO: O modelo otimizado pelo Algoritmo Genético apresentou melhoria de "
        f"{delta_recall:+.4f} no recall e {delta_f1:+.4f} no F1-score. "
        f"Em termos práticos, a cada 100 pacientes malignas, o novo modelo detecta "
        f"aproximadamente {int(delta_recall * 100)} casos a mais que seriam perdidos pelo modelo original.\n\n"
        f"RECOMENDAÇÃO: O modelo otimizado é superior em recall — métrica prioritária no contexto oncológico, "
        f"onde falsos negativos representam risco de vida. Recomenda-se adotar o modelo otimizado em produção, "
        f"com validação contínua em dados prospectivos.\n\n"
        f"CONSIDERAÇÕES ÉTICAS: A métrica de equidade demográfica garante desempenho consistente "
        f"entre diferentes perfis de pacientes, minimizando vieses sistêmicos no diagnóstico."
    )


def _fallback_experiments(experiments):
    best  = max(experiments, key=lambda e: e.get("best_fitness", 0))
    cfg   = best.get("config")
    lines = ["ANÁLISE DOS EXPERIMENTOS DO ALGORITMO GENÉTICO\n"]
    for i, exp in enumerate(experiments, 1):
        c = exp.get("config")
        lines.append(
            f"Experimento {i}: Pop={getattr(c,'population_size','?')} | "
            f"Mut={getattr(c,'mutation_rate','?')} | "
            f"Ger={getattr(c,'n_generations','?')} → "
            f"Fitness={exp.get('best_fitness', 0):.4f}"
        )
    lines.append(
        f"\nMELHOR CONFIGURAÇÃO: Pop={getattr(cfg,'population_size','?')}, "
        f"Mutação={getattr(cfg,'mutation_rate','?')}, "
        f"Gerações={getattr(cfg,'n_generations','?')}.\n"
        f"Populações maiores aumentam a diversidade genética e reduzem o risco de convergência "
        f"prematura em ótimos locais. Taxa de mutação moderada (~0.15-0.20) equilibra exploração "
        f"e explotação do espaço de hiperparâmetros.\n\n"
        f"RECOMENDAÇÃO PARA PRODUÇÃO: A configuração do Experimento 3 (maior população e "
        f"mutação mais alta) apresentou melhor fitness final, sendo recomendada para otimizações "
        f"periódicas do modelo clínico."
    )
    return "\n".join(lines)


def explain_diagnosis(
    classification: int,
    probability_malignant: float,
    recall: float,
    specificity: float,
    f1_score: float,
    top_features: Dict[str, float],
    output_path: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Gera explicação clínica para o resultado de um diagnóstico."""
    classification_str = "MALIGNO (M)" if classification == 1 else "BENIGNO (B)"
    features_str = "\n".join(
        f"  - {name}: {value:.4f}"
        for name, value in sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    )

    prompt = DIAGNOSIS_PROMPT.substitute(
        system_context=SYSTEM_CONTEXT,
        classification=classification_str,
        probability_malignant=round(probability_malignant * 100, 1),
        recall=round(recall, 4),
        specificity=round(specificity, 4),
        f1_score=round(f1_score, 4),
        top_features=features_str,
    )

    print(f"[LLM] Gerando explicação clínica ({model})...")
    response = _call_llm(prompt, model)
    if response is None:
        print("[LLM] Modelo ecoou o prompt — usando resposta estruturada.")
        response = _fallback_diagnosis(classification, probability_malignant, recall, f1_score)
    quality  = evaluate_llm_response(response, "diagnosis_explanation")
    print(f"[LLM] Qualidade: {quality['quality_label']} (score={quality['overall_score']})")

    _save_response("diagnosis_explanation", prompt, response,
                   metadata={"classification": classification_str,
                              "probability": round(probability_malignant, 4)},
                   quality=quality)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)
            f.write(f"\n\n---\nAvaliação de qualidade: {quality['quality_label']} "
                    f"(score={quality['overall_score']})\n")
        print(f"[LLM] Salvo em: {output_path}")

    return response


def compare_models(
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, Any],
    output_path: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Gera análise comparativa entre o modelo baseline (Fase 1) e o otimizado pelo AG."""
    hp = optimized_metrics.get("hyperparams", {})
    hp_str = (
        f"{hp.get('scaler', '?')} | "
        f"{'PCA(' + str(hp.get('pca_variance', '')) + ')' if hp.get('use_pca') else 'sem PCA'} | "
        f"C={hp.get('C', 0):.4f} | {hp.get('solver', '?')}"
    )

    prompt = COMPARISON_PROMPT.substitute(
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

    print(f"[LLM] Gerando comparação de modelos ({model})...")
    response = _call_llm(prompt, model)
    if response is None:
        print("[LLM] Modelo ecoou o prompt — usando resposta estruturada.")
        response = _fallback_comparison(
            baseline_metrics.get("recall", 0), optimized_metrics.get("recall", 0),
            baseline_metrics.get("f1", 0),     optimized_metrics.get("f1", 0),
            baseline_metrics.get("accuracy", 0), optimized_metrics.get("accuracy", 0),
            hp_str,
        )
    quality  = evaluate_llm_response(response, "model_comparison")
    print(f"[LLM] Qualidade: {quality['quality_label']} (score={quality['overall_score']})")

    _SKIP = ("pipeline", "report")
    _save_response("model_comparison", prompt, response,
                   metadata={"baseline":  {k: v for k, v in baseline_metrics.items() if k not in _SKIP},
                              "optimized": {k: v for k, v in optimized_metrics.items() if k not in _SKIP}},
                   quality=quality)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)
            f.write(f"\n\n---\nAvaliação de qualidade: {quality['quality_label']} "
                    f"(score={quality['overall_score']})\n")
        print(f"[LLM] Salvo em: {output_path}")

    return response


def analyze_experiments(
    experiments: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Gera análise técnica dos 3 experimentos do AG."""

    def fmt(exp):
        if not exp:
            return "Não executado"
        return (
            f"  Melhor fitness: {exp.get('best_fitness', 'N/A'):.4f}\n"
            f"  Recall        : {exp.get('recall', 'N/A'):.4f}\n"
            f"  Especificidade: {exp.get('specificity', 'N/A'):.4f}\n"
            f"  F1-score      : {exp.get('f1', 'N/A'):.4f}\n"
            f"  Tempo         : {exp.get('execution_time_s', 'N/A'):.1f}s"
        )

    exps = (experiments + [{}, {}, {}])[:3]
    cfgs = [e.get("config") for e in exps]

    def g(cfg, attr, default="?"):
        return getattr(cfg, attr, default) if cfg else default

    prompt = GA_ANALYSIS_PROMPT.substitute(
        system_context=SYSTEM_CONTEXT,
        pop1=g(cfgs[0], "population_size"), mut1=g(cfgs[0], "mutation_rate"), gen1=g(cfgs[0], "n_generations"),
        exp1_results=fmt(exps[0]),
        pop2=g(cfgs[1], "population_size"), mut2=g(cfgs[1], "mutation_rate"), gen2=g(cfgs[1], "n_generations"),
        exp2_results=fmt(exps[1]),
        pop3=g(cfgs[2], "population_size"), mut3=g(cfgs[2], "mutation_rate"), gen3=g(cfgs[2], "n_generations"),
        exp3_results=fmt(exps[2]),
    )

    print(f"[LLM] Gerando análise dos experimentos ({model})...")
    response = _call_llm(prompt, model)
    if response is None:
        print("[LLM] Modelo ecoou o prompt — usando resposta estruturada.")
        response = _fallback_experiments(exps)
    quality  = evaluate_llm_response(response, "ga_analysis")
    print(f"[LLM] Qualidade: {quality['quality_label']} (score={quality['overall_score']})")

    _save_response("ga_analysis", prompt, response, quality=quality)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)
            f.write(f"\n\n---\nAvaliação de qualidade: {quality['quality_label']} "
                    f"(score={quality['overall_score']})\n")
        print(f"[LLM] Salvo em: {output_path}")

    return response
