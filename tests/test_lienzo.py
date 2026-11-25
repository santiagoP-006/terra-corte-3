"""
test_lienzo.py
==============
Pruebas del lienzo de pixeles (src/lienzo.py).

Ejecucion:
    pytest tests/test_lienzo.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import png
from src.lienzo import Lienzo, FUENTE, BLANCO, NEGRO

ROJO = (255, 0, 0)


def test_lienzo_nuevo_tiene_el_fondo():
    lienzo = Lienzo(4, 3, fondo=(10, 20, 30))
    assert len(lienzo.datos) == 4 * 3 * 3
    assert lienzo.color_en(3, 2) == (10, 20, 30)


def test_pixel_y_fuera_de_rango():
    lienzo = Lienzo(5, 5)
    lienzo.pixel(2, 3, ROJO)
    lienzo.pixel(-1, 0, ROJO)        # fuera: no debe fallar
    lienzo.pixel(5, 5, ROJO)
    assert lienzo.color_en(2, 3) == ROJO
    assert lienzo.color_en(0, 0) == BLANCO


def test_rectangulo_relleno_incluye_las_esquinas():
    lienzo = Lienzo(10, 10)
    lienzo.rectangulo(2, 3, 5, 6, ROJO)
    assert lienzo.color_en(2, 3) == ROJO
    assert lienzo.color_en(5, 6) == ROJO
    assert lienzo.color_en(6, 6) == BLANCO
    pintados = sum(1 for y in range(10) for x in range(10)
                   if lienzo.color_en(x, y) == ROJO)
    assert pintados == 4 * 4


def test_rectangulo_que_se_sale_se_recorta():
    lienzo = Lienzo(4, 4)
    lienzo.rectangulo(-5, -5, 100, 1, ROJO)
    assert lienzo.color_en(3, 1) == ROJO
    assert lienzo.color_en(3, 2) == BLANCO


def test_linea_horizontal_vertical_y_diagonal():
    lienzo = Lienzo(10, 10)
    lienzo.linea(0, 0, 9, 0, ROJO)
    lienzo.linea(0, 0, 0, 9, ROJO)
    lienzo.linea(0, 0, 9, 9, ROJO)
    for i in range(10):
        assert lienzo.color_en(i, 0) == ROJO
        assert lienzo.color_en(0, i) == ROJO
        assert lienzo.color_en(i, i) == ROJO


def test_linea_en_cualquier_direccion_llega_al_final():
    lienzo = Lienzo(20, 20)
    lienzo.linea(15, 2, 3, 17, ROJO)          # hacia la izquierda y abajo
    assert lienzo.color_en(15, 2) == ROJO
    assert lienzo.color_en(3, 17) == ROJO


def test_circulo_relleno():
    lienzo = Lienzo(11, 11)
    lienzo.circulo(5, 5, 3, ROJO)
    assert lienzo.color_en(5, 5) == ROJO
    assert lienzo.color_en(8, 5) == ROJO      # justo en el borde
    assert lienzo.color_en(8, 8) == BLANCO    # esquina: fuera del circulo


def test_fuente_completa_y_bien_formada():
    for letra, filas in FUENTE.items():
        assert len(filas) == 7, letra
        for fila in filas:
            assert len(fila) == 5 and set(fila) <= {"#", "."}, letra


def test_ancho_del_texto():
    assert Lienzo.ancho_texto("AB") == 11          # 5 + 1 + 5
    assert Lienzo.ancho_texto("AB", escala=2) == 22
    assert Lienzo.ancho_texto("") == 0


def test_texto_pinta_la_letra():
    lienzo = Lienzo(20, 10)
    lienzo.texto(0, 0, "I", NEGRO)
    # La I es ".###." en la primera fila
    assert lienzo.color_en(0, 0) == BLANCO
    assert lienzo.color_en(1, 0) == NEGRO
    assert lienzo.color_en(2, 3) == NEGRO     # el palo del medio


def test_minusculas_y_tildes_usan_la_misma_letra():
    a = Lienzo(10, 10)
    b = Lienzo(10, 10)
    a.texto(0, 0, "á")
    b.texto(0, 0, "A")
    assert a.datos == b.datos


def test_guardar_png_valido(tmp_path):
    lienzo = Lienzo(30, 20)
    lienzo.rectangulo(0, 0, 9, 9, ROJO)
    ruta = str(tmp_path / "lienzo.png")
    lienzo.guardar_png(ruta)
    ancho, alto, datos = png.leer_png(ruta)
    assert (ancho, alto) == (30, 20)
    assert datos == lienzo.datos
