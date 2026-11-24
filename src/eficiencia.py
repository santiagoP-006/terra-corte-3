"""
eficiencia.py
=============
Medir la eficiencia de los programas Terra CONTANDO OPERACIONES.

No se mide el tiempo con el reloj (eso necesitaria el modulo time de
Python): se cuenta el trabajo que hace el programa. Contar tiene una
ventaja: da siempre el mismo resultado, en cualquier computador, y deja
ver como crece el trabajo cuando crecen los datos (la complejidad).

Como se cuenta
--------------
Cada vez que el evaluador va a evaluar un nodo del arbol de analisis, lo
pasa primero por Contador.registrar(). Segun el tipo de nodo se suma en
una categoria:

    nodos            todos los nodos evaluados (el trabajo total)
    aritmeticas      + - * / % ^ y el signo -
    comparaciones    == != > < >= <= y 'en'
    logicas          && || !
    asignaciones     fijo, var y cambios de valor
    decisiones       cada 'si' que se evalua
    iteraciones      cada vuelta de un 'mientras' o un 'para'
    llamadas         cada llamada a una funcion
    operaciones de datos   cada paso |> de una tuberia

Una funcion incorporada (por ejemplo knn o ordenar) cuenta como UNA
llamada: lo que se mide es el trabajo del programa escrito en Terra.

Funciones del lenguaje
----------------------
    iniciar_conteo()                 pone los contadores en cero
    reporte_conteo()                 tabla con lo contado desde el inicio
    contar_operaciones(f [, args])   cuenta lo que hace UNA llamada a f
    crecimiento(f, [n1, n2, ...])    llama f(n) para cada n y cuenta: sirve
                                     para graficar como crece el trabajo
    memoria_total()                  bytes estimados de todas las variables

Por consola: python src/terra.py programa.terra --eficiencia
imprime el reporte al terminar el programa.
"""

from src.biblioteca import _registrar, llamar_funcion, es_funcion
from src.memoria import estimar_bytes
from src.tabla import Tabla, ErrorDatos

CATEGORIAS = ["nodos", "aritmeticas", "comparaciones", "logicas",
              "asignaciones", "decisiones", "iteraciones", "llamadas",
              "operaciones de datos"]


class Contador:
    """Lleva la cuenta de las operaciones, por categoria."""

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.cuentas = {c: 0 for c in CATEGORIAS}

    def copia(self) -> dict:
        return dict(self.cuentas)

    def registrar(self, nodo) -> None:
        """Suma lo que corresponde al tipo de nodo que se va a evaluar."""
        self.cuentas["nodos"] += 1
        tipo = type(nodo).__name__
        regla = _REGLAS.get(tipo)
        if regla:
            categoria, cantidad = regla(nodo)
            if cantidad:
                self.cuentas[categoria] += cantidad


def _unario(nodo):
    if nodo.postfijo() is not None:
        return "aritmeticas", 0
    if nodo.getChild(0).getText() == "!":
        return "logicas", 1
    return "aritmeticas", 1


def _bloque(nodo):
    """Un bloque cuyo padre es un ciclo es una vuelta del ciclo."""
    padre = type(nodo.parentCtx).__name__ if nodo.parentCtx else ""
    if padre in ("Mientras_stmtContext", "Para_stmtContext"):
        return "iteraciones", 1
    return "iteraciones", 0


# tipo de nodo -> funcion que dice (categoria, cuantas operaciones)
_REGLAS = {
    "SumaContext":        lambda n: ("aritmeticas", len(n.termino()) - 1),
    "TerminoContext":     lambda n: ("aritmeticas", len(n.potencia()) - 1),
    "PotenciaContext":    lambda n: ("aritmeticas",
                                     1 if n.potencia() is not None else 0),
    "UnarioContext":      _unario,
    "IgualdadContext":    lambda n: ("comparaciones",
                                     len(n.relacional()) - 1),
    "RelacionalContext":  lambda n: ("comparaciones",
                                     len(n.pertenencia()) - 1),
    "PertenenciaContext": lambda n: ("comparaciones",
                                     1 if n.EN() is not None else 0),
    "O_logicoContext":    lambda n: ("logicas", len(n.y_logico()) - 1),
    "Y_logicoContext":    lambda n: ("logicas", len(n.igualdad()) - 1),
    "Declaracion_variableContext": lambda n: ("asignaciones", 1),
    "AsignacionContext":  lambda n: ("asignaciones", 1),
    "Si_stmtContext":     lambda n: ("decisiones", 1),
    "BloqueContext":      _bloque,
    "LlamadaContext":     lambda n: ("llamadas", 1),
    "TuberiaContext":     lambda n: ("operaciones de datos",
                                     len(n.operacion())),
}


# =====================================================================
# REPORTES
# =====================================================================

def _diferencia(despues: dict, antes: dict) -> dict:
    return {c: despues[c] - antes[c] for c in CATEGORIAS}


def _total(cuentas: dict) -> int:
    """Operaciones: todo menos 'nodos', que es otra forma de medir."""
    return sum(v for c, v in cuentas.items() if c != "nodos")


def tabla_de_cuentas(cuentas: dict) -> Tabla:
    filas = [[c, cuentas[c]] for c in CATEGORIAS]
    filas.append(["total de operaciones", _total(cuentas)])
    return Tabla(["categoria", "cantidad"], filas)


def memoria_total(evaluador) -> int:
    """Bytes estimados de todas las variables del programa."""
    return sum(estimar_bytes(dato["valor"])
               for dato in evaluador.memoria.values())


def reporte_texto(evaluador) -> str:
    """Reporte que imprime la consola con --eficiencia."""
    cuentas = evaluador.contador.copia()
    lineas = ["", "Eficiencia del programa (operaciones contadas)",
              "-" * 46]
    for c in CATEGORIAS:
        lineas.append(f"  {c:<22}{cuentas[c]:>12,}".replace(",", "."))
    lineas.append(f"  {'total de operaciones':<22}"
                  f"{_total(cuentas):>12,}".replace(",", "."))
    lineas.append(f"  {'memoria de variables':<22}"
                  f"{memoria_total(evaluador):>12,} bytes"
                  .replace(",", "."))
    return "\n".join(lineas)


# =====================================================================
# FUNCIONES DEL LENGUAJE
# =====================================================================

def funciones(evaluador) -> dict:
    """Recibe el evaluador porque necesita su contador y su memoria."""
    marca = {"inicio": evaluador.contador.copia()}

    def iniciar_conteo():
        marca["inicio"] = evaluador.contador.copia()
        return None

    def reporte_conteo():
        return tabla_de_cuentas(_diferencia(evaluador.contador.copia(),
                                            marca["inicio"]))

    def _medir(f, argumentos, quien):
        if not es_funcion(f):
            raise ErrorDatos(f"{quien}(): el primer argumento debe ser una "
                             f"funcion, por ejemplo funcion(n) => ... o el "
                             f"nombre de una funcion")
        antes = evaluador.contador.copia()
        llamar_funcion(evaluador, f, argumentos, 0)
        return _diferencia(evaluador.contador.copia(), antes)

    def contar_operaciones(f, argumentos=None):
        if argumentos is None:
            argumentos = []
        if not isinstance(argumentos, list):
            raise ErrorDatos("contar_operaciones(): los argumentos de la "
                             "funcion van en una lista, por ejemplo [10]")
        return tabla_de_cuentas(_medir(f, argumentos, "contar_operaciones"))

    def crecimiento(f, entradas):
        """Tabla n | operaciones | nodos, para ver la complejidad."""
        if not isinstance(entradas, list) or not entradas:
            raise ErrorDatos("crecimiento(): las entradas van en una lista, "
                             "por ejemplo [10, 20, 40, 80]")
        filas = []
        for n in entradas:
            cuentas = _medir(f, [n], "crecimiento")
            filas.append([n, _total(cuentas), cuentas["nodos"]])
        return Tabla(["n", "operaciones", "nodos"], filas)

    def memoria_total_():
        return memoria_total(evaluador)

    return _registrar([
        ("iniciar_conteo",     iniciar_conteo,     0),
        ("reporte_conteo",     reporte_conteo,     0),
        ("contar_operaciones", contar_operaciones, 1, 2),
        ("crecimiento",        crecimiento,        2),
        ("memoria_total",      memoria_total_,     0),
    ])
