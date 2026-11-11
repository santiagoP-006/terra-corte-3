"""
test_arbol_decision.py
======================
Pruebas del modulo de arbol de decision de Terra.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_terra import evaluar, ejecutar
from src.visitante_evaluador import ErrorSemantico


DATOS = (
    'fijo X = [[1.0, 0.0], [1.1, 0.0], [3.0, 1.0], [3.1, 1.0]]\n'
    'fijo y = ["bajo", "bajo", "alto", "alto"]\n'
)


# --- Entrenamiento ---

def test_arbol_devuelve_modelo():
    m = evaluar(DATOS + 'fijo m = arbol_decision(X, y)\n')
    assert isinstance(m, dict)
    assert m["tipo"] == "arbol_decision"


def test_arbol_clases_correctas():
    m = evaluar(DATOS + 'fijo m = arbol_decision(X, y)\n')
    assert m["clases"] == ["alto", "bajo"]


def test_arbol_n_correcto():
    m = evaluar(DATOS + 'fijo m = arbol_decision(X, y)\n')
    assert m["n"] == 4


def test_arbol_p_correcto():
    m = evaluar(DATOS + 'fijo m = arbol_decision(X, y)\n')
    assert m["p"] == 2


def test_arbol_respeta_max_profundidad():
    from src import arbol_decision as ad
    m = evaluar(DATOS + 'fijo m = arbol_decision(X, y, 1)\n')
    assert ad._profundidad_real(m["raiz"]) <= 1


def test_arbol_X_vacio():
    with pytest.raises(ErrorSemantico, match="vacios"):
        evaluar('fijo m = arbol_decision([], [])\n')


def test_arbol_X_e_y_distintas_longitudes():
    with pytest.raises(ErrorSemantico):
        evaluar('fijo m = arbol_decision([[1.0]], ["a", "b"])\n')


def test_arbol_X_no_es_lista_de_listas():
    with pytest.raises(ErrorSemantico, match="lista de listas"):
        evaluar('fijo m = arbol_decision([1.0, 2.0], ["a", "b"])\n')


def test_arbol_max_profundidad_invalida():
    with pytest.raises(ErrorSemantico, match="max_profundidad"):
        evaluar(DATOS + 'fijo m = arbol_decision(X, y, 0)\n')


# --- Clasificacion ---

def test_clasificar_clase_correcta():
    resultado = evaluar(
        DATOS +
        'fijo m = arbol_decision(X, y)\n'
        'fijo c = clasificar(m, [1.0, 0.0])\n')
    assert resultado == "bajo"


def test_clasificar_clase_alta():
    resultado = evaluar(
        DATOS +
        'fijo m = arbol_decision(X, y)\n'
        'fijo c = clasificar(m, [3.0, 1.0])\n')
    assert resultado == "alto"


def test_clasificar_numero_incorrecto_de_predictores():
    with pytest.raises(ErrorSemantico, match="2 predictores"):
        evaluar(
            DATOS +
            'fijo m = arbol_decision(X, y)\n'
            'fijo c = clasificar(m, [1.0])\n')


def test_clasificar_sin_modelo_valido():
    with pytest.raises(ErrorSemantico):
        evaluar('fijo c = clasificar(5, [1.0, 0.0])\n')


# --- Clasificacion en lote ---

def test_clasificar_lote_devuelve_lista():
    resultado = evaluar(
        DATOS +
        'fijo m = arbol_decision(X, y)\n'
        'fijo ps = clasificar_lote(m, [[1.0, 0.0], [3.0, 1.0]])\n')
    assert resultado == ["bajo", "alto"]


def test_clasificar_lote_longitud_correcta():
    resultado = evaluar(
        DATOS +
        'fijo m = arbol_decision(X, y)\n'
        'fijo ps = clasificar_lote(m, X)\n')
    assert len(resultado) == 4


# --- Evaluacion ---

def test_evaluar_arbol_exactitud_perfecta():
    resultado = evaluar(
        DATOS +
        'fijo m = arbol_decision(X, y)\n'
        'fijo e = evaluar_arbol(m, X, y)\n'
        'fijo a = exactitud(e)\n')
    assert resultado == 1.0


def test_evaluar_arbol_correctas():
    resultado = evaluar(
        DATOS +
        'fijo m = arbol_decision(X, y)\n'
        'fijo e = evaluar_arbol(m, X, y)\n')
    assert resultado["correctas"] == 4
    assert resultado["total"]     == 4


def test_evaluar_arbol_longitudes_distintas():
    with pytest.raises(ErrorSemantico):
        evaluar(
            DATOS +
            'fijo m = arbol_decision(X, y)\n'
            'fijo e = evaluar_arbol(m, X, ["bajo"])\n')


def test_exactitud_sobre_valor_invalido():
    with pytest.raises(ErrorSemantico):
        evaluar('fijo a = exactitud(5)\n')


# --- mostrar_arbol ---

def test_mostrar_arbol_no_lanza_error(capsys):
    ejecutar(
        DATOS +
        'fijo m = arbol_decision(X, y)\n'
        'mostrar_arbol(m, ["feat1", "feat2"])\n', capsys)


def test_mostrar_arbol_sin_nombres(capsys):
    ejecutar(
        DATOS +
        'fijo m = arbol_decision(X, y)\n'
        'mostrar_arbol(m)\n', capsys)


# --- Caso ambiental ---

def test_clasificacion_emisiones():
    codigo = (
        'fijo X = [[1.68, 26.4], [1.91, 27.1], '
        '[4.41, 31.2], [4.72, 34.5], '
        '[2.38, 22.1], [2.42, 23.4]]\n'
        'fijo y = ["moderada", "moderada", '
        '"alta", "alta", "media", "media"]\n'
        'fijo m = arbol_decision(X, y, 3)\n'
        'fijo c = clasificar(m, [1.75, 27.0])\n')
    assert evaluar(codigo) == "moderada"