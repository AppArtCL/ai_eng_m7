from langgraph.checkpoint.memory import InMemorySaver

from app.graph import construir_grafo


def test_construir_grafo_contiene_nodos_del_m6():
    nodos = construir_grafo(InMemorySaver()).get_graph().nodes
    assert {"supervisor", "investigador", "analista", "sintesis"} <= set(nodos)
