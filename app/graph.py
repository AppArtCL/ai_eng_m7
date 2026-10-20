from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing import Literal
import logging

from app import config
from app.state import AgentState
from app.agents.supervisor import nodo_supervisor
from app.agents.supervisor import nodo_sintesis
from app.agents.analyst_agent import nodo_analista
from app.agents.research_agent import nodo_investigador

logger = logging.getLogger(__name__)


def enrutar(state: AgentState) -> Literal["investigador", "analista", "sintesis"]:
    """Traduce la decisión del supervisor al nombre del próximo nodo."""
    siguiente = state.get("next_agent")
    if siguiente in ("investigador", "analista"):
        return siguiente
    return "sintesis"


def construir_grafo(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Arma el orquestador y lo compila con el checkpointer recibido (Redis en la API)."""
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
    return grafo.compile(checkpointer=checkpointer)


def exportar_diagrama():
    """Exporta el diagrama del grafo a un archivo visualizable."""
    dibujo = construir_grafo().get_graph()
    with open(config.BASE_DIR / "diagrama.md", "w", encoding="utf-8") as archivo:
        archivo.write(f"```mermaid\n{dibujo.draw_mermaid()}```\n")
    try:
        dibujo.draw_mermaid_png(output_file_path=str(config.BASE_DIR / "diagrama.png"))
    except Exception as e:
        logger.warning("No se pudo generar el PNG (requiere internet): %s", e)


if __name__ == "__main__":
    exportar_diagrama()
