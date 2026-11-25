"""
test_arreglos.py
================
Pruebas de los errores corregidos al empezar el tercer corte:

    - mostrar() de arrays, DataFrames, modelos y evaluaciones
    - nombres de tipo (array, dataframe, modelo) en los mensajes de error
    - funciones de matrices que aceptan listas y Array por igual
    - potencia: -2 ^ 2 es -4 y potencias que no dan numero real
    - red de seguridad: errores internos de Python con linea y en espanol
    - validacion de argumentos de las funciones de graficado
    - sample() con el generador propio y dataframe(nombres, listas)
    - codigo de salida de la consola y regresion de los ejemplos

Ejecucion:
    pytest tests/test_arreglos.py -v
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.terra_numpy import Array
from src.terra_pandas import DataFrame
from src.visitante_evaluador import ErrorSemantico
from tests.test_terra import evaluar, ejecutar

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TERRA = os.path.join(RAIZ, "src", "terra.py")


def correr(ruta):
    """Ejecuta un archivo .terra por consola y devuelve el proceso."""
    return subprocess.run([sys.executable, TERRA, ruta], cwd=RAIZ,
                          capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=300)


# =====================================================================
# MOSTRAR Y TIPOS
# =====================================================================

def test_mostrar_array_muestra_los_valores(capsys):
    salida = ejecutar("mostrar(array([[1.0, 2.0], [3.0, 4.0]]))\n", capsys)
    assert "Array(2, 2)" in salida
    assert "4.0000" in salida


def test_mostrar_modelo_de_arbol_es_legible(capsys):
    salida = ejecutar(
        'fijo m = arbol_decision([[1.0], [2.0], [3.0], [4.0]], '
        '["a", "a", "b", "b"])\nmostrar(m)\n', capsys)
    assert salida.startswith("Arbol de decision")
    assert "{" not in salida


def test_mostrar_evaluacion_es_legible(capsys):
    salida = ejecutar(
        'fijo m = arbol_decision([[1.0], [2.0], [3.0], [4.0]], '
        '["a", "a", "b", "b"])\n'
        'mostrar(evaluar_arbol(m, [[1.0], [4.0]], ["a", "b"]))\n', capsys)
    assert salida == "Evaluacion: exactitud=100.0%  (2/2)"


def test_error_nombra_el_tipo_array():
    # array + array ya es valido (conversiones.py); con un texto no
    with pytest.raises(ErrorSemantico, match=r"array \+ texto"):
        evaluar('fijo a = array([1.0])\nfijo b = a + "x"\n')


def test_tipo_dataframe_en_mensajes():
    with pytest.raises(ErrorSemantico, match=r"dataframe \+ texto"):
        evaluar('fijo d = dataframe(["x"], [[1, 2]])\nfijo r = d + "x"\n')

# =====================================================================
# MATRICES: LISTAS Y ARRAY
# =====================================================================

def test_transpuesta_de_array():
    t = evaluar("fijo t = transpuesta(array([[1.0, 2.0], [3.0, 4.0]]))\n")
    assert isinstance(t, Array)
    assert t.como_listas() == [[1.0, 3.0], [2.0, 4.0]]


def test_transpuesta_de_lista_sigue_funcionando():
    assert evaluar("fijo t = transpuesta([[1, 2], [3, 4]])\n") == \
        [[1, 3], [2, 4]]


@pytest.mark.parametrize("expresion, esperado", [
    ("determinante(array([[4.0, 7.0], [2.0, 6.0]]))", 10.0),
    ("traza(array([[4.0, 7.0], [2.0, 6.0]]))", 10.0),
    ("norma(array([3.0, 4.0]))", 5.0),
    ("determinante([[4, 7], [2, 6]])", 10.0),
])
def test_funciones_de_matrices_con_ambos_tipos(expresion, esperado):
    assert evaluar(f"fijo r = {expresion}\n") == pytest.approx(esperado)


def test_inversa_de_array_devuelve_array():
    inv = evaluar("fijo r = inversa(array([[4.0, 7.0], [2.0, 6.0]]))\n")
    assert isinstance(inv, Array)
    assert inv.como_listas()[0] == pytest.approx([0.6, -0.7])


def test_dimension_de_array():
    assert evaluar("fijo r = dimension(array([[1.0, 2.0, 3.0]]))\n") == [1, 3]


def test_concatenar_arrays():
    r = evaluar("fijo a = array([[1.0, 2.0]])\n"
                "fijo r = concatenar(a, a)\n")
    assert r.forma == (2, 2)


# =====================================================================
# POTENCIA
# =====================================================================

@pytest.mark.parametrize("expresion, esperado", [
    ("-2 ^ 2", -4),
    ("(-2) ^ 2", 4),
    ("-2 ^ 3", -8),
    ("2 ^ -1", 0.5),
    ("2 ^ 3 ^ 2", 512),
    ("- -2 ^ 2", 4),
])
def test_potencia_antes_que_el_signo(expresion, esperado):
    assert evaluar(f"fijo r = {expresion}\n") == esperado


def test_base_negativa_con_exponente_decimal():
    with pytest.raises(ErrorSemantico, match="no es un numero real"):
        evaluar("fijo r = (-8) ^ 0.5\n")


def test_cero_a_la_negativa():
    with pytest.raises(ErrorSemantico, match="division por cero"):
        evaluar("fijo r = 0 ^ -1\n")


# =====================================================================
# RED DE SEGURIDAD Y VALIDACION DE GRAFICAS
# =====================================================================

def test_error_interno_sale_con_linea_y_en_espanol():
    # tail() no valida n: el texto llega a Python como error de tipo y la
    # red de seguridad de visitSentencia lo traduce
    with pytest.raises(ErrorSemantico,
                       match=r"\[linea 2\] los argumentos no tienen"):
        evaluar('fijo d = dataframe(["x"], [[1, 2, 3]])\n'
                'fijo r = tail(d, "x")\n')


@pytest.mark.parametrize("llamada, mensaje", [
    ('exportar_histograma([1.0, 2.0], 3, "x.svg")',
     "argumento 2 debe ser un texto"),
    ('graficar_barras(["a", "b"], [1, "x"])',
     "argumento 2 debe ser una lista de numeros"),
    ('graficar_histograma([1.0, 2.0], 0)',
     "entero mayor o igual a 1"),
    ('graficar_lineas([1, 2], [3, 4], "t", "x", "y", "sobra")',
     "maximo 5 argumentos"),
])
def test_graficas_validan_argumentos(llamada, mensaje):
    with pytest.raises(ErrorSemantico, match=mensaje):
        evaluar(llamada + "\n")


def test_graficas_aceptan_array(capsys):
    salida = ejecutar(
        'graficar_histograma(array([1.0, 2.0, 2.5, 3.0]), 2, "Arr")\n',
        capsys)
    assert "Arr" in salida


# =====================================================================
# DATAFRAME Y ALEATORIOS PROPIOS
# =====================================================================

def test_dataframe_con_listas_paralelas():
    df = evaluar('fijo d = dataframe(["pais", "co2"], '
                 '[["Colombia", "Peru"], [86.2, 52.4]])\n')
    assert isinstance(df, DataFrame)
    assert df.columnas == ["pais", "co2"]
    assert df.num_filas == 2


def test_dataframe_rechaza_columnas_de_distinto_largo():
    with pytest.raises(ErrorSemantico, match="misma"):
        evaluar('fijo d = dataframe(["a", "b"], [[1, 2], [3]])\n')


def test_sample_reproducible_con_semilla():
    codigo = ('fijo d = dataframe(["x"], [[1, 2, 3, 4, 5, 6, 7, 8]])\n'
              'fijo r = sample(d, 3, 42)\n')
    primera = evaluar(codigo)
    segunda = evaluar(codigo)
    assert primera.filas == segunda.filas
    assert len(primera.filas) == 3
    valores = [f[0] for f in primera.filas]
    assert len(set(valores)) == 3          # sin reemplazo


def test_pandas_no_usa_random():
    with open(os.path.join(RAIZ, "src", "terra_pandas.py"),
              encoding="utf-8") as f:
        assert "import random" not in f.read()


# =====================================================================
# CONSOLA Y REGRESION DE EJEMPLOS
# =====================================================================

def test_consola_sale_con_1_si_hay_error(tmp_path):
    programa = tmp_path / "malo.terra"
    programa.write_text("mostrar(10 / 0)\n", encoding="utf-8")
    proceso = correr(str(programa))
    assert proceso.returncode == 1
    assert "division por cero" in proceso.stderr


def test_consola_sale_con_1_si_hay_error_sintactico(tmp_path):
    programa = tmp_path / "malo.terra"
    programa.write_text("fijo = 3\n", encoding="utf-8")
    assert correr(str(programa)).returncode == 1


def test_consola_sale_con_0_si_todo_va_bien(tmp_path):
    programa = tmp_path / "bueno.terra"
    programa.write_text("mostrar(1 + 1)\n", encoding="utf-8")
    proceso = correr(str(programa))
    assert proceso.returncode == 0
    assert proceso.stdout.strip() == "2"


# Ejemplos que deben fallar a proposito o que necesitan algo especial
ESPERAN_ERROR = {"06_errores.terra"}        # errores sintacticos de muestra
NECESITAN_TECLADO = {"17_entrada.terra"}    # piden datos con leer()
PENDIENTES = {}                             # ejemplos que aun no funcionan
EJEMPLOS = sorted(f for f in os.listdir(os.path.join(RAIZ, "ejemplos"))
                  if f.endswith(".terra"))


@pytest.mark.parametrize("nombre", EJEMPLOS)
def test_regresion_ejemplos(nombre):
    """Todos los ejemplos deben seguir corriendo despues de cada cambio."""
    if nombre in NECESITAN_TECLADO:
        pytest.skip("necesita entrada por teclado")
    if nombre in PENDIENTES:
        pytest.xfail(PENDIENTES[nombre])

    proceso = correr(os.path.join("ejemplos", nombre))
    if nombre in ESPERAN_ERROR:
        assert proceso.returncode == 1
    else:
        assert proceso.returncode == 0, proceso.stderr
