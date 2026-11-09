"""
test_regresion.py
=================
Pruebas de las funciones de regresion lineal de Terra.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_terra import evaluar
from src.visitante_evaluador import ErrorSemantico


# --- Simple ---

def test_regresion_simple_coeficientes():
    # y = 1 + 2x  exacto
    resultado = evaluar(
        'fijo m = regresion_lineal([1, 2, 3], [3, 5, 7])\n'
        'fijo c = coeficientes(m)\n')
    assert abs(resultado[0] - 1.0) < 1e-9
    assert abs(resultado[1] - 2.0) < 1e-9


def test_regresion_simple_r_cuadrado_perfecto():
    resultado = evaluar(
        'fijo m = regresion_lineal([1, 2, 3], [3, 5, 7])\n'
        'fijo r = r_cuadrado(m)\n')
    assert abs(resultado - 1.0) < 1e-9


def test_predecir_simple():
    resultado = evaluar(
        'fijo m = regresion_lineal([1, 2, 3], [3, 5, 7])\n'
        'fijo p = predecir(m, 4)\n')
    assert abs(resultado - 9.0) < 1e-6


def test_regresion_simple_necesita_dos_obs():
    with pytest.raises(ErrorSemantico, match="2 observaciones"):
        evaluar('fijo m = regresion_lineal([1], [2])\n')


def test_regresion_simple_listas_distintas_longitudes():
    with pytest.raises(ErrorSemantico):
        evaluar('fijo m = regresion_lineal([1, 2], [3])\n')


def test_regresion_simple_x_constante():
    with pytest.raises(ErrorSemantico, match="iguales"):
        evaluar('fijo m = regresion_lineal([5, 5, 5], [1, 2, 3])\n')


# --- Multiple ---

def test_regresion_multiple_coeficientes():
    # y = 1 + 2*x1 + 3*x2  exacto con 3 puntos
    resultado = evaluar(
        'fijo X = [[1, 1], [2, 1], [1, 2]]\n'
        'fijo y = [6, 8, 9]\n'
        'fijo m = regresion_multiple(X, y)\n'
        'fijo c = coeficientes(m)\n')
    assert abs(resultado[0] - 1.0) < 1e-6
    assert abs(resultado[1] - 2.0) < 1e-6
    assert abs(resultado[2] - 3.0) < 1e-6


def test_predecir_multiple():
    resultado = evaluar(
        'fijo X = [[1, 1], [2, 1], [1, 2]]\n'
        'fijo y = [6, 8, 9]\n'
        'fijo m = regresion_multiple(X, y)\n'
        'fijo p = predecir_multiple(m, [3, 1])\n')
    assert abs(resultado - 10.0) < 1e-6


def test_r_cuadrado_sobre_modelo_invalido():
    with pytest.raises(ErrorSemantico):
        evaluar('fijo r = r_cuadrado(5)\n')