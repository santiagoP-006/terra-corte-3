"""
test_aprendizaje.py
===================
Pruebas del Machine Learning propio (src/aprendizaje.py): preparacion de
datos, metricas, KNN, k-medias y perceptron multicapa.

Los datos de prueba son pequenos y con grupos bien separados, para que
el resultado correcto se conozca de antemano.

Ejecucion:
    pytest tests/test_aprendizaje.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import aprendizaje as ml
from src.tabla import Tabla, ErrorDatos
from src.terra_numpy import Array
from src.visitante_evaluador import ErrorSemantico
from tests.test_terra import evaluar, ejecutar

# Dos nubes de puntos bien separadas: "a" cerca de (0, 0), "b" cerca de (10, 10)
X = [[0, 0], [1, 0], [0, 1], [1, 1], [10, 10], [11, 10], [10, 11], [11, 11]]
Y = ["a", "a", "a", "a", "b", "b", "b", "b"]


# =====================================================================
# PREPARACION DE DATOS
# =====================================================================

def test_dividir_datos_tamanos_y_reproducible():
    uno = ml.dividir_datos(X, Y, 0.25, 7)
    dos = ml.dividir_datos(X, Y, 0.25, 7)
    assert uno == dos
    x_ent, x_pru, y_ent, y_pru = uno
    assert len(x_ent) == 6 and len(x_pru) == 2
    assert len(y_ent) == 6 and len(y_pru) == 2
    # ninguna fila se pierde ni se repite
    assert sorted(map(tuple, x_ent + x_pru)) == sorted(map(tuple, X))


def test_dividir_datos_con_tabla_devuelve_tablas():
    t = Tabla(["pais", "v"], [["x", 1], ["y", 2], ["z", 3], ["w", 4]])
    x_ent, x_pru, _, _ = ml.dividir_datos(t, [1, 2, 3, 4], 0.5)
    assert isinstance(x_ent, Tabla) and x_ent.columnas == ["pais", "v"]
    assert x_ent.num_filas + x_pru.num_filas == 4


def test_normalizar_min_max():
    assert ml.normalizar([[0, 10], [5, 20], [10, 30]]) == \
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]


def test_normalizar_con_referencia():
    # el 20 se escala con el minimo y maximo de la referencia (0 a 10)
    assert ml.normalizar([[20]], [[0], [10]]) == [[2.0]]


def test_estandarizar_media_cero():
    r = ml.estandarizar([[1], [2], [3]])
    assert sum(f[0] for f in r) == pytest.approx(0)
    assert r[2][0] == pytest.approx(1.2247, abs=1e-4)


def test_normalizar_respeta_el_tipo():
    assert isinstance(ml.normalizar(Array([1.0, 2.0], (2, 1))), Array)
    t = Tabla(["n", "v"], [["a", 0], ["b", 10]])
    r = ml.normalizar(t)
    assert r.filas == [["a", 0.0], ["b", 1.0]]


# =====================================================================
# METRICAS
# =====================================================================

def test_metricas_de_regresion():
    reales, pred = [3, 5, 7], [2, 5, 9]
    assert ml.mse(reales, pred) == pytest.approx(5 / 3)
    assert ml.rmse(reales, pred) == pytest.approx((5 / 3) ** 0.5)
    assert ml.mae(reales, pred) == pytest.approx(1.0)
    assert ml.r2(reales, reales) == 1.0


def test_metricas_de_clasificacion():
    reales = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", "b"]
    assert ml.exactitud(reales, pred) == 0.75
    assert ml.precision(reales, pred, "b") == pytest.approx(2 / 3)
    assert ml.recall(reales, pred, "a") == 0.5
    assert ml.f1(reales, pred, "a") == pytest.approx(2 / 3)


def test_exactitud_acepta_resultado_de_evaluar_arbol():
    assert ml.exactitud({"exactitud": 0.8}) == 0.8


def test_matriz_de_confusion():
    t = ml.matriz_confusion(["a", "a", "b"], ["a", "b", "b"])
    assert t.columnas == ["real", "a", "b"]
    assert t.filas == [["a", 1, 1], ["b", 0, 1]]


def test_reporte_tiene_fila_de_promedio():
    t = ml.reporte_clasificacion(Y, Y)
    assert t.filas[-1][0] == "promedio"
    assert t.filas[-1][3] == 1.0


# =====================================================================
# MODELOS
# =====================================================================

def test_knn_clasifica_cada_nube():
    modelo = ml.knn(X, Y, 3)
    assert ml.aplicar_modelo(modelo, [[0.5, 0.5], [10.5, 10.5]]) == ["a", "b"]
    assert ml.aplicar_modelo(modelo, [0.2, 0.1]) == "a"     # una sola fila


def test_knn_regresion_promedia_los_vecinos():
    modelo = ml.knn_regresion([[1], [2], [3], [10]], [10, 20, 30, 100], 2)
    assert ml.aplicar_modelo(modelo, [[1.4]]) == [15.0]


def test_knn_con_k_muy_grande():
    with pytest.raises(ErrorDatos, match="mayor que la cantidad"):
        ml.knn(X, Y, 20)


def test_kmeans_encuentra_las_dos_nubes():
    modelo = ml.kmeans(X, 2, semilla=1)
    grupos = modelo["etiquetas"]
    assert len(set(grupos[:4])) == 1 and len(set(grupos[4:])) == 1
    assert grupos[0] != grupos[4]
    assert ml.aplicar_modelo(modelo, [[0, 0]]) == [grupos[0]]


def test_inercia_baja_al_aumentar_k():
    inercias = ml.inercias_kmeans(X, 4)
    assert len(inercias) == 4
    assert inercias[0] > inercias[1] >= inercias[3]


def test_perceptron_clasificador_aprende():
    modelo = ml.perceptron_clasificador(X, Y, [4], 200, 0.1)
    assert modelo["perdidas"][-1] < modelo["perdidas"][0]
    assert ml.aplicar_modelo(modelo, X) == Y


def test_perceptron_regresor_aprende_una_recta():
    xs = [[i] for i in range(10)]
    ys = [2 * i + 1 for i in range(10)]
    modelo = ml.perceptron_regresor(xs, ys, [6], 500, 0.01)
    predichos = ml.aplicar_modelo(modelo, xs)
    assert ml.r2(ys, predichos) > 0.95


def test_perceptron_es_reproducible():
    a = ml.perceptron_clasificador(X, Y, [3], 20, 0.1, 5)
    b = ml.perceptron_clasificador(X, Y, [3], 20, 0.1, 5)
    assert a["perdidas"] == b["perdidas"]


def test_aplicar_modelo_con_regresion_del_segundo_corte():
    from src import regresion as reg
    modelo = reg.entrenar_simple([1, 2, 3], [2, 4, 6])
    assert ml.aplicar_modelo(modelo, [4, 5]) == pytest.approx([8, 10])


def test_numero_de_variables_incorrecto():
    with pytest.raises(ErrorDatos, match="2 variable"):
        ml.aplicar_modelo(ml.knn(X, Y), [[1, 2, 3]])


def test_x_con_nulos():
    with pytest.raises(ErrorDatos, match="nulo"):
        ml.knn([[1, None], [2, 3]], ["a", "b"], 1)


# =====================================================================
# DESDE EL LENGUAJE
# =====================================================================

PROGRAMA = '''
fijo d = cargar "datos/emisiones_ejemplo.csv" |> rellenar nulos en energia_renovable con 0.0
fijo X = d |> seleccionar [co2, energia_renovable, poblacion]
fijo y = columna(d, "pais")
fijo partes = dividir_datos(X, y, 0.25, 7)
'''


def test_flujo_completo_en_terra(capsys):
    salida = ejecutar(PROGRAMA + '''
fijo m = perceptron_clasificador(partes[0], partes[2], [8], 300, 0.05)
fijo p = aplicar_modelo(m, partes[1])
mostrar(exactitud(partes[3], p))
mostrar(m)
''', capsys)
    lineas = salida.splitlines()
    assert float(lineas[0]) >= 0.75
    assert "Perceptron multicapa (clasificacion)" in salida


def test_mostrar_kmeans_y_knn(capsys):
    salida = ejecutar(PROGRAMA + '''
mostrar(kmeans(normalizar(X), 3))
mostrar(knn(X, y, 3))
''', capsys)
    assert "K-medias con k=3" in salida
    assert "KNN (clasificacion) con k=3" in salida


def test_error_de_ml_sale_con_linea():
    with pytest.raises(ErrorSemantico, match=r"\[linea 2\].*mayor que"):
        evaluar('fijo X = [[1], [2]]\nfijo m = knn(X, ["a", "b"], 5)\n')
