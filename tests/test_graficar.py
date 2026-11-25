"""
test_graficar.py
================
Pruebas de la sentencia 'graficar' del lenguaje (visitGraficar_stmt y
src/visualizacion.py).

Ejecucion:
    pytest tests/test_graficar.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import png
from src.visitante_evaluador import ErrorSemantico
from tests.test_terra import evaluar, ejecutar

CARGA = 'fijo d = cargar "datos/emisiones_ejemplo.csv"\n'


def programa(tipo, opciones, ruta=None):
    """Arma un programa con una sola sentencia graficar."""
    texto = CARGA + f"graficar {tipo} d\n"
    for opcion in opciones:
        texto += f"    {opcion}\n"
    if ruta is not None:
        texto += f'    guardar como "{ruta}"\n'
    return texto


def es_png_valido(ruta) -> bool:
    ancho, alto, _ = png.leer_png(str(ruta))
    return ancho > 0 and alto > 0


# =====================================================================
# LOS CINCO TIPOS
# =====================================================================

@pytest.mark.parametrize("tipo, opciones", [
    ("barras", ["eje_x pais", "eje_y co2"]),
    ("lineas", ["eje_x anio", "eje_y co2", 'leyenda "pais"']),
    ("histograma", ["eje_x co2"]),
    ("dispersion", ["eje_x poblacion", "eje_y co2"]),
    ("caja", ["eje_x pais", "eje_y co2"]),
])
def test_cada_tipo_genera_un_png(tipo, opciones, tmp_path, capsys):
    ruta = tmp_path / f"{tipo}.png"
    salida = ejecutar(programa(tipo, opciones + ['titulo "Prueba"'], ruta),
                      capsys)
    assert f"Grafica de {tipo} guardada" in salida
    assert es_png_valido(ruta)


def test_opciones_en_cualquier_orden(tmp_path, capsys):
    ruta = tmp_path / "orden.png"
    ejecutar(programa("barras", ['titulo "T"', "eje_y co2", "eje_x pais"],
                      ruta), capsys)
    assert es_png_valido(ruta)


def test_sin_guardar_dibuja_en_la_terminal(capsys):
    salida = ejecutar(programa("barras", ["eje_x pais", "eje_y co2",
                                          'titulo "En consola"']), capsys)
    assert "En consola" in salida
    assert "█" in salida


def test_graficar_un_resultado_de_tuberia(tmp_path, capsys):
    ruta = tmp_path / "resumen.png"
    codigo = (CARGA +
              "fijo r = d |> agrupar por [pais] |> resumir total = suma(co2)\n"
              "graficar barras r\n"
              "    eje_x pais\n"
              "    eje_y total\n"
              f'    guardar como "{ruta}"\n')
    ejecutar(codigo, capsys)
    assert es_png_valido(ruta)


# =====================================================================
# ERRORES
# =====================================================================

@pytest.mark.parametrize("tipo, opciones, mensaje", [
    ("barras", ["eje_x pais"], "necesita la opcion 'eje_y'"),
    ("barras", ["eje_x pais", "eje_y continente"], "no existe"),
    ("barras", ["eje_x pais", "eje_y pais"], "debe ser numerica"),
    ("lineas", ["eje_x anio", "eje_y co2"], "valores repetidos"),
    ("barras", ["eje_x pais", "eje_x anio", "eje_y co2"], "repetida"),
    ("barras", ["eje_x pais", "eje_y co2", "titulo 5"], "debe ser texto"),
])
def test_errores_de_opciones(tipo, opciones, mensaje):
    with pytest.raises(ErrorSemantico, match=mensaje):
        evaluar(programa(tipo, opciones))


def test_ruta_debe_ser_png(tmp_path):
    with pytest.raises(ErrorSemantico, match="formato PNG"):
        evaluar(programa("barras", ["eje_x pais", "eje_y co2"],
                         tmp_path / "grafica.jpg"))


def test_graficar_algo_que_no_es_tabla():
    with pytest.raises(ErrorSemantico, match="necesita una tabla"):
        evaluar("fijo x = 5\ngraficar barras x\n    eje_x a\n")


def test_el_error_dice_la_linea():
    with pytest.raises(ErrorSemantico, match=r"\[linea 2\]"):
        evaluar(programa("caja", []))

