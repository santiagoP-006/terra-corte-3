"""
test_graficas.py
================
Pruebas de las cinco graficas en PNG (src/graficas.py).

Cada grafica se guarda en una carpeta temporal y se vuelve a leer con
png.leer_png, que verifica la firma y las sumas de verificacion. Ademas
se revisan los calculos que hay detras: marcas de los ejes, conteos del
histograma y cuartiles de la caja.

Ejecucion:
    pytest tests/test_graficas.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import graficas, png
from src.lienzo import PALETA
from src.tabla import ErrorDatos

CO2_CO = [86.2, 92.5, 79.8, 88.1]
CO2_PE = [52.4, 57.9, 50.3, 55.6]
ANIOS = [2015, 2018, 2021, 2023]


def leer(ruta):
    """Lee el PNG (falla si esta mal formado) y devuelve sus pixeles."""
    ancho, alto, datos = png.leer_png(str(ruta))
    assert (ancho, alto) == (graficas.ANCHO, graficas.ALTO)
    return datos


def tiene_color(datos, color) -> bool:
    objetivo = bytes(color)
    return any(datos[i:i + 3] == objetivo for i in range(0, len(datos), 3))


# =====================================================================
# CALCULOS
# =====================================================================

@pytest.mark.parametrize("minimo, maximo, esperado", [
    (38.7, 92.5, [20, 40, 60, 80, 100]),
    (0, 4, [0, 1, 2, 3, 4]),
    (0, 0.83, [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    (-3.2, 7.9, [-5, 0, 5, 10]),
])
def test_marcas_bonitas(minimo, maximo, esperado):
    marcas, _ = graficas.marcas_bonitas(minimo, maximo)
    assert marcas == pytest.approx(esperado)


def test_marcas_con_rango_cero():
    marcas, _ = graficas.marcas_bonitas(5, 5)
    assert marcas[0] < 5 < marcas[-1]


@pytest.mark.parametrize("valor, paso, texto", [
    (3, 1, "3"),
    (0.5, 0.1, "0.5"),
    (0.25, 0.25, "0.25"),
    (20000, 5000, "20K"),
    (1500000, 500000, "1.5M"),
])
def test_formatear_numero(valor, paso, texto):
    assert graficas.formatear_numero(valor, paso) == texto


@pytest.mark.parametrize("p, esperado", [
    (0, 1), (0.25, 2), (0.5, 3), (0.75, 4), (1, 5),
])
def test_cuartil_con_datos_exactos(p, esperado):
    assert graficas.cuartil([1, 2, 3, 4, 5], p) == esperado


def test_cuartil_interpola():
    # posicion 0.25 * 3 = 0.75: tres cuartos entre 10 y 20
    assert graficas.cuartil([10, 20, 30, 40], 0.25) == pytest.approx(17.5)


# =====================================================================
# LAS CINCO GRAFICAS
# =====================================================================

def test_barras(tmp_path):
    ruta = tmp_path / "barras.png"
    graficas.grafica_barras(["Colombia", "Peru"], [346.6, 216.2], str(ruta),
                            "CO2", "pais", "Mt")
    assert tiene_color(leer(ruta), PALETA[0])


def test_barras_con_valores_negativos(tmp_path):
    ruta = tmp_path / "negativas.png"
    graficas.grafica_barras(["a", "b"], [-3, 5], str(ruta))
    leer(ruta)


def test_lineas_con_dos_series_y_leyenda(tmp_path):
    ruta = tmp_path / "lineas.png"
    graficas.grafica_lineas(ANIOS, [CO2_CO, CO2_PE], str(ruta),
                            "Evolucion", "anio", "Mt",
                            ["Colombia", "Peru"])
    datos = leer(ruta)
    assert tiene_color(datos, PALETA[0])
    assert tiene_color(datos, PALETA[1])      # la segunda serie


def test_lineas_ignora_nulos(tmp_path):
    ruta = tmp_path / "nulos.png"
    graficas.grafica_lineas([1, 2, 3], [[1.0, None, 3.0]], str(ruta))
    leer(ruta)


def test_histograma_cuenta_bien(tmp_path):
    ruta = tmp_path / "hist.png"
    conteos = graficas.grafica_histograma([1, 2, 2, 3, 3, 3, 10], str(ruta),
                                          bins=3)
    # intervalos de ancho 3: [1,4) [4,7) [7,10] -> el maximo va al ultimo
    assert conteos == [6, 0, 1]
    assert sum(conteos) == 7
    leer(ruta)


def test_histograma_con_valores_iguales(tmp_path):
    ruta = tmp_path / "iguales.png"
    conteos = graficas.grafica_histograma([4, 4, 4], str(ruta), bins=2)
    assert sum(conteos) == 3


def test_dispersion_por_grupos(tmp_path):
    ruta = tmp_path / "disp.png"
    graficas.grafica_dispersion(CO2_CO + CO2_PE, [50, 51, 50, 52, 33, 33,
                                                  34, 34], str(ruta),
                                grupos=["CO"] * 4 + ["PE"] * 4)
    datos = leer(ruta)
    assert tiene_color(datos, PALETA[0])
    assert tiene_color(datos, PALETA[1])


def test_caja_calcula_cuartiles_y_atipicos(tmp_path):
    ruta = tmp_path / "caja.png"
    resumen = graficas.grafica_caja([[1, 2, 3, 4, 5, 100]], ["datos"],
                                    str(ruta))
    caja = resumen[0]
    assert caja["mediana"] == pytest.approx(3.5)
    assert caja["atipicos"] == [100]
    assert caja["bigote_alto"] == 5
    leer(ruta)


# =====================================================================
# ERRORES
# =====================================================================

def test_barras_con_texto_en_los_valores(tmp_path):
    with pytest.raises(ErrorDatos, match="deben ser numeros"):
        graficas.grafica_barras(["a"], ["x"], str(tmp_path / "e.png"))


def test_lineas_con_largos_distintos(tmp_path):
    with pytest.raises(ErrorDatos, match="tantos valores"):
        graficas.grafica_lineas([1, 2, 3], [[1, 2]], str(tmp_path / "e.png"))


def test_grafica_sin_datos(tmp_path):
    with pytest.raises(ErrorDatos, match="no hay datos"):
        graficas.grafica_histograma([None, None], str(tmp_path / "e.png"))
