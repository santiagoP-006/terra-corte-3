"""
test_conversiones.py
====================
Pruebas de tablas <-> matrices <-> arrays y de los operadores + - * /
entre ellos (src/conversiones.py).

Ejecucion:
    pytest tests/test_conversiones.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.terra_numpy import Array
from src.terra_pandas import DataFrame
from src.tabla import Tabla
from src.visitante_evaluador import ErrorSemantico
from tests.test_terra import evaluar

TABLA = 'fijo t = dataframe(["pais", "a", "b"], [["CO", "PE"], [1, 3], [2, 4]])\n'
MATRIZ = "fijo A = array([[1.0, 2.0], [3.0, 4.0]])\n"


# =====================================================================
# CONVERSIONES
# =====================================================================

def test_tabla_a_array_toma_las_columnas_numericas():
    r = evaluar(TABLA + "fijo r = a_array(t)\n")
    assert isinstance(r, Array)
    assert r.forma == (2, 2)
    assert r.como_listas() == [[1.0, 2.0], [3.0, 4.0]]


def test_tabla_a_array_con_columnas_elegidas():
    r = evaluar(TABLA + 'fijo r = a_array(t, ["b"])\n')
    assert r.como_listas() == [[2.0], [4.0]]


def test_array_a_tabla_con_nombres():
    r = evaluar(MATRIZ + 'fijo r = a_tabla(A, ["x", "y"])\n')
    assert isinstance(r, DataFrame)
    assert r.columnas == ["x", "y"]
    assert r.filas == [[1.0, 2.0], [3.0, 4.0]]


def test_lista_a_tabla_con_nombres_automaticos():
    r = evaluar("fijo r = a_tabla([[1, 2], [3, 4]])\n")
    assert r.columnas == ["c1", "c2"]


def test_ida_y_vuelta_tabla_array_tabla():
    r = evaluar(TABLA + 'fijo r = a_tabla(a_array(t), ["a", "b"])\n')
    assert r.filas == [[1.0, 2.0], [3.0, 4.0]]


def test_a_listas():
    assert evaluar(MATRIZ + "fijo r = a_listas(A)\n") == [[1.0, 2.0],
                                                           [3.0, 4.0]]


def test_la_tabla_resultante_sirve_en_tuberias():
    r = evaluar(MATRIZ + 'fijo r = a_tabla(A, ["x", "y"]) '
                         '|> filtrar donde x > 2\n')
    assert r.num_filas == 1


# =====================================================================
# OPERADORES CON ARRAYS
# =====================================================================

@pytest.mark.parametrize("expresion, esperado", [
    ("A + A", [[2, 4], [6, 8]]),
    ("A - A", [[0, 0], [0, 0]]),
    ("A * A", [[1, 4], [9, 16]]),        # elemento a elemento
    ("A / A", [[1, 1], [1, 1]]),
    ("A * 10", [[10, 20], [30, 40]]),
    ("10 - A", [[9, 8], [7, 6]]),
    ("-A", [[-1, -2], [-3, -4]]),
    ("A - array([1.0, 2.0])", [[0, 0], [2, 2]]),   # fila contra matriz
])
def test_operadores_con_arrays(expresion, esperado):
    r = evaluar(MATRIZ + f"fijo r = {expresion}\n")
    assert r.como_listas() == esperado


def test_formas_incompatibles():
    with pytest.raises(ErrorSemantico, match="misma forma"):
        evaluar(MATRIZ + "fijo r = A + array([1.0, 2.0, 3.0])\n")


def test_division_por_cero_en_array():
    with pytest.raises(ErrorSemantico, match="division por cero"):
        evaluar(MATRIZ + "fijo r = A / 0\n")


def test_modulo_no_se_permite():
    with pytest.raises(ErrorSemantico, match="solo \\+ - \\* /"):
        evaluar(MATRIZ + "fijo r = A % 2\n")


# =====================================================================
# OPERADORES CON TABLAS
# =====================================================================

def test_tabla_por_numero_solo_cambia_columnas_numericas():
    r = evaluar(TABLA + "fijo r = t * 10\n")
    assert isinstance(r, DataFrame)
    assert r.filas == [["CO", 10, 20], ["PE", 30, 40]]


def test_tabla_mas_tabla():
    r = evaluar(TABLA + "fijo r = t + t\n")
    assert r.filas == [["CO", 2, 4], ["PE", 6, 8]]


def test_tabla_menos_array():
    r = evaluar(TABLA + MATRIZ + "fijo r = t - A\n")
    assert r.filas == [["CO", 0.0, 0.0], ["PE", 0.0, 0.0]]


def test_negativo_de_tabla():
    r = evaluar(TABLA + "fijo r = -t\n")
    assert r.filas == [["CO", -1, -2], ["PE", -3, -4]]


def test_los_nulos_siguen_nulos():
    r = evaluar('fijo t = dataframe(["x"], [[1, nulo]])\nfijo r = t + 1\n')
    assert r.filas == [[2], [None]]


def test_la_tabla_original_no_cambia():
    codigo = TABLA + "fijo r = t * 10\n"
    from tests.test_terra import Terra
    interprete = Terra()
    arbol, _, _, _ = interprete.analizar(codigo)
    interprete.visitante.visit(arbol)
    original = interprete.visitante.memoria["t"]["valor"]
    assert original.filas == [["CO", 1, 2], ["PE", 3, 4]]


def test_tablas_con_columnas_distintas():
    with pytest.raises(ErrorSemantico, match="mismas columnas"):
        evaluar(TABLA + 'fijo u = dataframe(["z"], [[1, 2]])\n'
                        "fijo r = t + u\n")


def test_texto_distinto_no_se_puede_operar():
    with pytest.raises(ErrorSemantico, match="no es numerica"):
        evaluar(TABLA +
                'fijo u = dataframe(["pais", "a", "b"], '
                '[["CO", "EC"], [1, 3], [2, 4]])\n'
                "fijo r = t + u\n")


def test_lista_de_listas_sigue_uniendo_listas():
    assert evaluar("fijo r = [[1, 2]] + [[3, 4]]\n") == [[1, 2], [3, 4]]


def test_lista_con_array_pide_convertir():
    with pytest.raises(ErrorSemantico, match="conviertela con array"):
        evaluar(MATRIZ + "fijo r = A + [[1, 2], [3, 4]]\n")


# =====================================================================
# PRODUCTO DE MATRICES
# =====================================================================

@pytest.mark.parametrize("a, b", [
    ("A", "A"),
    ("[[1, 2], [3, 4]]", "A"),
    ("a_tabla(A)", "[[1, 2], [3, 4]]"),
])
def test_producto_mat_con_cualquier_tipo(a, b):
    r = evaluar(MATRIZ + f"fijo r = producto_mat({a}, {b})\n")
    assert r.como_listas() == [[7.0, 10.0], [15.0, 22.0]]


def test_tabla_con_nulos_no_se_convierte():
    with pytest.raises(ErrorSemantico, match="rellenar nulos"):
        evaluar('fijo t = dataframe(["x"], [[1, nulo]])\n'
                "fijo r = a_array(t)\n")
