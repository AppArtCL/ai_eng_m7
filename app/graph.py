from langgraph.graph import StateGraph, START, END
from typing import Literal
import logging

import config
from state import AgentState
from agents.supervisor import nodo_supervisor
from agents.supervisor import nodo_sintesis
from agents.analyst_agent import nodo_analista
from agents.research_agent import nodo_investigador

logger = logging.getLogger(__name__)


def enrutar(state: AgentState) -> Literal["investigador", "analista", "sintesis"]:
    """Traduce la decisión del supervisor al nombre del próximo nodo."""
    siguiente = state.get("next_agent")
    if siguiente in ("investigador", "analista"):
        return siguiente
    return "sintesis"


grafo = StateGraph(AgentState)
grafo.add_node("investigador", nodo_investigador)
grafo.add_node("analista", nodo_analista)
grafo.add_node("sintesis", nodo_sintesis)
grafo.add_node("supervisor", nodo_supervisor)
grafo.add_edge(START, "supervisor")
grafo.add_conditional_edges("supervisor", enrutar)
grafo.add_edge("investigador", "supervisor")
grafo.add_edge("analista", "supervisor")
grafo.add_edge("sintesis", END)

app = grafo.compile()


def exportar_diagrama():
    """Exporta el diagrama del grafo a un archivo visualizable."""
    with open(config.BASE_DIR / "diagrama.md", "w", encoding="utf-8") as archivo:
        archivo.write(f"```mermaid\n{app.get_graph().draw_mermaid()}```\n")
    try:
        app.get_graph().draw_mermaid_png(output_file_path=str(config.BASE_DIR / "diagrama.png"))
    except Exception as e:
        logger.warning("No se pudo generar el PNG (requiere internet): %s", e)


if __name__ == "__main__":
    exportar_diagrama()
