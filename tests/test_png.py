"""
test_png.py
===========
Pruebas del escritor de PNG propio (src/png.py).

Los valores de referencia de CRC-32 y Adler-32 son los publicados en los
estandares (por ejemplo, el CRC-32 de "123456789" es 0xCBF43926), asi que
no hace falta ninguna libreria para comprobarlos.

Ejecucion:
    pytest tests/test_png.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import png
from src.tabla import ErrorDatos


# =====================================================================
# SUMAS DE VERIFICACION
# =====================================================================

@pytest.mark.parametrize("datos, esperado", [
    (b"", 0x00000000),
    (b"a", 0xE8B7BE43),
    (b"123456789", 0xCBF43926),      # valor de prueba estandar del CRC-32
    (b"IEND", 0xAE426082),           # el CRC que lleva todo PNG al final
])
def test_crc32_valores_conocidos(datos, esperado):
    assert png.crc32(datos) == esperado


@pytest.mark.parametrize("datos, esperado", [
    (b"", 0x00000001),
    (b"a", 0x00620062),
    (b"Wikipedia", 0x11E60398),      # ejemplo clasico de Adler-32
])
def test_adler32_valores_conocidos(datos, esperado):
    assert png.adler32(datos) == esperado


# =====================================================================
# ESCRITURA Y LECTURA
# =====================================================================

def test_firma_y_bloques(tmp_path):
    ruta = tmp_path / "uno.png"
    png.escribir_png(str(ruta), 1, 1, bytearray([10, 20, 30]))
    contenido = ruta.read_bytes()
    assert contenido[:8] == png.FIRMA_PNG
    assert contenido[12:16] == b"IHDR"
    assert contenido[-8:-4] == b"IEND"


def test_ida_y_vuelta_conserva_los_pixeles(tmp_path):
    datos = bytearray([255, 0, 0,   0, 255, 0,   0, 0, 255,
                       255, 255, 255, 0, 0, 0,   128, 128, 128])
    ruta = str(tmp_path / "colores.png")
    png.escribir_png(ruta, 3, 2, datos)
    ancho, alto, leidos = png.leer_png(ruta)
    assert (ancho, alto) == (3, 2)
    assert leidos == datos


def test_imagen_grande_usa_varios_bloques(tmp_path):
    # 200 x 200 x 3 = 120 000 bytes: no caben en un solo bloque de 65 535
    datos = bytearray([7, 8, 9]) * (200 * 200)
    ruta = str(tmp_path / "grande.png")
    png.escribir_png(ruta, 200, 200, datos)
    assert png.leer_png(ruta)[2] == datos


def test_archivo_danado_se_detecta(tmp_path):
    ruta = tmp_path / "danado.png"
    png.escribir_png(str(ruta), 2, 2, bytearray(12))
    contenido = bytearray(ruta.read_bytes())
    contenido[20] ^= 0xFF                  # cambiar un byte del IHDR
    ruta.write_bytes(bytes(contenido))
    with pytest.raises(ErrorDatos, match="CRC"):
        png.leer_png(str(ruta))


# =====================================================================
# ERRORES
# =====================================================================

def test_tamano_de_datos_incorrecto():
    with pytest.raises(ErrorDatos, match="se esperaban 12 bytes"):
        png.codificar_png(2, 2, bytearray(5))


def test_imagen_vacia():
    with pytest.raises(ErrorDatos, match="al menos 1 x 1"):
        png.codificar_png(0, 5, bytearray())


def test_carpeta_inexistente(tmp_path):
    ruta = str(tmp_path / "no_existe" / "x.png")
    with pytest.raises(ErrorDatos, match="la carpeta no existe"):
        png.escribir_png(ruta, 1, 1, bytearray(3))
