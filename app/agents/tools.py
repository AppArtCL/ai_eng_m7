import ast
import operator as op
from functools import lru_cache
from langchain_core.tools import tool
from rag_system import RAGSystem

_OPERADORES = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.USub: op.neg, ast.UAdd: op.pos,
}
_MAX_EXPONENTE = 100

@lru_cache(maxsize=1)
def get_rag() -> RAGSystem:
    """Construye el RAGSystem una sola vez y lo reutiliza en cada búsqueda."""
    return RAGSystem()


@tool
def buscar_en_reglamento(pregunta: str) -> str:
    """Busca información en el Reglamento de Copropiedad del Edificio Riesco 6540.

    Úsala para cualquier pregunta sobre normas, prohibiciones, horarios, montos o
    plazos del edificio: mascotas, ruidos, asados y parrillas, piscina, salón multiuso,
    mudanzas, obras, estacionamientos, multas (UTM) y gastos comunes.

    Ejemplo de entrada: "horario permitido para mudanzas"
    """
    documentos = get_rag().retrieve(pregunta)
    if not documentos:
        return "No se encontró información relevante en el reglamento."
    return "\n\n".join(
        f"[Fuente: {d.metadata.get('source', '?')} | Sección: {d.metadata.get('seccion', '?')}]\n{d.page_content}"
        for d in documentos
    )

def _evaluar_nodo(nodo):
    """Recorre el árbol de la expresión y evalúa solo nodos permitidos."""
    if isinstance(nodo, ast.Constant) and type(nodo.value) in (int, float):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES:
        izquierda, derecha = _evaluar_nodo(nodo.left), _evaluar_nodo(nodo.right)
        if isinstance(nodo.op, ast.Pow) and abs(derecha) > _MAX_EXPONENTE:
            raise ValueError(f"exponente mayor a {_MAX_EXPONENTE}")
        return _OPERADORES[type(nodo.op)](izquierda, derecha)
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES:
        return _OPERADORES[type(nodo.op)](_evaluar_nodo(nodo.operand))
    raise ValueError(f"operación no permitida: {type(nodo).__name__}")

@tool
def calculadora(expresion: str) -> str:
    """Evalúa una expresión aritmética y devuelve el resultado exacto.

    Operadores permitidos: + - * / // % ** y paréntesis. Solo números, sin variables
    ni unidades. Úsala para cualquier cálculo en vez de hacerlo mentalmente.

    Ejemplo de entrada: "5 * (17.5 - 7) + (14 - 9)"
    """
    try:
        resultado = _evaluar_nodo(ast.parse(expresion, mode="eval").body)
        return f"{expresion} = {resultado}"
    except ZeroDivisionError:
        return f"Error al calcular '{expresion}': división por cero."
    except (ValueError, SyntaxError) as e:
        return f"Error al calcular '{expresion}': {e}"

