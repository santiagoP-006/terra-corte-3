"""
test_eficiencia.py
==================
Pruebas del contador de operaciones (src/eficiencia.py).

Ejecucion:
    pytest tests/test_eficiencia.py -v
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visitante_evaluador import ErrorSemantico
from tests.test_terra import evaluar

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMA_HASTA = '''
funcion suma_hasta(n):
    var total = 0
    var i = 1
    mientras i <= n:
        total = total + i
        i = i + 1
    fin
    retornar total
fin
'''

PARES = '''
funcion pares(n):
    var c = 0
    para i en rango(n):
        para j en rango(n):
            si i < j:
                c = c + 1
            fin
        fin
    fin
    retornar c
fin
'''


def cuentas(tabla) -> dict:
    """Convierte la tabla categoria | cantidad en diccionario."""
    return {fila[0]: fila[1] for fila in tabla.filas}


def test_cuenta_cada_categoria():
    t = evaluar(SUMA_HASTA +
                "fijo r = contar_operaciones(suma_hasta, [10])\n")
    c = cuentas(t)
    assert c["iteraciones"] == 10           # 10 vueltas del mientras
    assert c["comparaciones"] == 11         # i <= n se evalua 11 veces
    assert c["aritmeticas"] == 20           # dos sumas por vuelta
    assert c["asignaciones"] == 22          # 2 iniciales + 2 por vuelta
    assert c["total de operaciones"] == 63


def test_crecimiento_lineal():
    t = evaluar(SUMA_HASTA +
                "fijo r = crecimiento(suma_hasta, [10, 20, 40])\n")
    operaciones = [fila[1] for fila in t.filas]
    # duplicar n casi duplica el trabajo: O(n)
    assert operaciones[1] / operaciones[0] == pytest.approx(2, abs=0.1)
    assert operaciones[2] / operaciones[1] == pytest.approx(2, abs=0.1)


def test_crecimiento_cuadratico():
    t = evaluar(PARES + "fijo r = crecimiento(pares, [10, 20])\n")
    operaciones = [fila[1] for fila in t.filas]
    # duplicar n multiplica el trabajo casi por 4: O(n^2)
    assert operaciones[1] / operaciones[0] == pytest.approx(4, abs=0.2)


def test_contar_una_funcion_anonima():
    t = evaluar("fijo r = contar_operaciones(funcion(x) => x * 2 + 1, [5])\n")
    assert cuentas(t)["aritmeticas"] == 2


def test_iniciar_y_reportar():
    t = evaluar("iniciar_conteo()\n"
                "fijo a = 1 + 2\n"
                "fijo b = a > 2 && a < 5\n"
                "fijo r = reporte_conteo()\n")
    c = cuentas(t)
    assert c["aritmeticas"] == 1
    assert c["comparaciones"] == 2
    assert c["logicas"] == 1
    # a, b y tambien r: el 'fijo r = ...' se cuenta antes de que
    # reporte_conteo() haga la cuenta
    assert c["asignaciones"] == 3


def test_memoria_total_crece_con_los_datos():
    chica = evaluar("fijo a = [1, 2]\nfijo m = memoria_total()\n")
    grande = evaluar("fijo a = rango(1000)\nfijo m = memoria_total()\n")
    assert grande > chica > 0


def test_el_primer_argumento_debe_ser_funcion():
    with pytest.raises(ErrorSemantico, match="debe ser una funcion"):
        evaluar("fijo r = contar_operaciones(5)\n")


def test_bandera_eficiencia_en_consola(tmp_path):
    programa = tmp_path / "p.terra"
    programa.write_text("fijo x = 2 + 3\nmostrar(x)\n", encoding="utf-8")
    proceso = subprocess.run(
        [sys.executable, os.path.join(RAIZ, "src", "terra.py"),
         str(programa), "--eficiencia"],
        cwd=RAIZ, capture_output=True, text=True, timeout=60)
    assert proceso.returncode == 0
    assert "Eficiencia del programa" in proceso.stdout
    assert "aritmeticas" in proceso.stdout
