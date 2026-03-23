"""
route_chat.py
Chat em linguagem natural sobre as rotas do dia via Google Gemini.

Permite que a equipe de campo faça perguntas como:
  - "Qual o próximo atendimento prioritário?"
  - "Quantas paradas de emergência temos hoje?"
  - "Há alguma janela de tempo apertada?"
  - "Qual a distância total da rota?"

Privacidade: respostas nunca incluem dados pessoais identificáveis.
"""

import os
from typing import List, Optional, Dict, Any
from collections import Counter

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .prompts import SYSTEM_CONTEXT, CHAT_SYSTEM_PROMPT, CHAT_USER_TEMPLATE
from ..src.data_generator import ServicePoint, TYPE_LABELS
from ..src.constraints import compute_arrival_times, _route_distance


# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------

def _get_gemini_model(model_name: str = "gemini-2.0-flash"):
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "google-generativeai não está instalado. "
            "Execute: pip install google-generativeai"
        )
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
# Utilitários
# ---------------------------------------------------------------------------

def _decimal_to_hhmm(hour: float) -> str:
    h = int(hour)
    m = int((hour - h) * 60)
    return f"{h:02d}:{m:02d}"


def _build_route_context(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
) -> str:
    """
    Constrói o contexto da rota para o sistema de chat.
    Não inclui dados pessoais identificáveis de pacientes.
    """
    schedule = compute_arrival_times(route, points, dist_matrix)
    schedule_map = {pid: (arr, dep) for pid, arr, dep in schedule}
    point_map = {p.id: p for p in points}

    total_dist = _route_distance(route, dist_matrix)
    total_demand = sum(point_map[pid].demand for pid in route)
    type_count = Counter(point_map[pid].type for pid in route)

    lines = [
        f"Data/Hora de saída: 07:00 (depósito central)",
        f"Total de paradas: {len(route)}",
        f"Distância total estimada: {total_dist:.1f} km",
        f"Demanda total de suprimentos: {total_demand} unidades",
        "",
        "Composição da rota por tipo de atendimento:",
    ]
    for stype, count in sorted(type_count.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  - {TYPE_LABELS[stype]}: {count} parada(s)")

    lines.append("")
    lines.append("Sequência detalhada:")

    for i, pid in enumerate(route, 1):
        pt = point_map[pid]
        arr, _ = schedule_map.get(pid, (None, None))
        arr_str = _decimal_to_hhmm(arr) if arr else "a calcular"
        label = TYPE_LABELS[pt.type]
        urgency = " [URGENTE]" if pt.priority <= 2 else ""
        tw = f"{int(pt.time_window[0]):02d}:00–{int(pt.time_window[1]):02d}:00"

        lines.append(
            f"  {i:>2}. {pt.name} | {label}{urgency}"
            f" | Chegada: {arr_str} | Janela: {tw}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interface de chat
# ---------------------------------------------------------------------------

class RouteChat:
    """
    Assistente de chat para perguntas em linguagem natural sobre a rota do dia.

    Uso:
        chat = RouteChat(route, points, dist_matrix)
        resposta = chat.ask("Qual o próximo atendimento prioritário?")
    """

    def __init__(
        self,
        route: List[int],
        points: List[ServicePoint],
        dist_matrix: List[List[float]],
        model_name: str = "gemini-2.0-flash",
    ):
        self.route = route
        self.points = points
        self.dist_matrix = dist_matrix
        self.model_name = model_name
        self._model = None
        self._route_context = _build_route_context(route, points, dist_matrix)

        # Contexto para o sistema (sem dados sensíveis)
        self._system_prompt = CHAT_SYSTEM_PROMPT.substitute(
            system_context=SYSTEM_CONTEXT,
            route_data=self._route_context,
        )

    def _ensure_model(self):
        if self._model is None:
            self._model = _get_gemini_model(self.model_name)

    def ask(self, question: str) -> str:
        """
        Faz uma pergunta sobre a rota e retorna a resposta do Gemini.

        Args:
            question: pergunta em linguagem natural

        Returns:
            Resposta textual do assistente
        """
        self._ensure_model()

        full_prompt = (
            self._system_prompt
            + "\n\n"
            + CHAT_USER_TEMPLATE.substitute(question=question)
        )

        try:
            response = self._model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            return f"[Erro na consulta: {e}]"

    def interactive_session(self):
        """
        Inicia uma sessão interativa de chat no terminal.
        Digite 'sair' ou 'exit' para encerrar.
        """
        print("\n" + "=" * 60)
        print(" CHAT SOBRE A ROTA — Saúde da Mulher")
        print(" (Digite 'sair' para encerrar)")
        print("=" * 60)
        print(f"\nRota carregada: {len(self.route)} paradas")
        print("Contexto privacidade: dados pessoais protegidos\n")

        while True:
            try:
                user_input = input("Você: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Sessão encerrada]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("sair", "exit", "quit"):
                print("[Sessão encerrada]")
                break

            answer = self.ask(user_input)
            print(f"\nAssistente: {answer}\n")


# ---------------------------------------------------------------------------
# Perguntas pré-definidas (demo / testes)
# ---------------------------------------------------------------------------

DEMO_QUESTIONS = [
    "Qual o próximo atendimento prioritário da rota?",
    "Quantas paradas de emergência obstétrica temos hoje?",
    "Há algum atendimento de violência doméstica na rota?",
    "Qual a distância total da rota de hoje?",
    "Quais paradas têm janela de tempo mais restrita?",
    "Como devo me preparar para o primeiro atendimento do dia?",
]


def run_demo_questions(
    route: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
    model_name: str = "gemini-2.0-flash",
) -> Dict[str, str]:
    """
    Executa perguntas de demonstração e retorna as respostas.

    Returns:
        Dicionário {pergunta: resposta}
    """
    chat = RouteChat(route, points, dist_matrix, model_name=model_name)
    results = {}

    print("\n" + "=" * 60)
    print(" DEMO — Perguntas frequentes sobre a rota")
    print("=" * 60)

    for q in DEMO_QUESTIONS:
        print(f"\nPergunta: {q}")
        answer = chat.ask(q)
        results[q] = answer
        print(f"Resposta: {answer}")

    return results
