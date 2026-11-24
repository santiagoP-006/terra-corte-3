"""
conversiones.py
===============
Tablas, matrices y arrays: pasar de uno a otro y operar entre ellos.

En Terra hay tres formas de guardar datos en filas y columnas:

    Tabla / DataFrame   columnas con nombre, cada una de su tipo
                        (texto, numero...). Es lo que devuelve 'cargar'.
    Array               numeros en una o dos dimensiones, para algebra
                        lineal (terra_numpy.py).
    lista de listas     la "matriz" de la biblioteca: [[1, 2], [3, 4]].

Conversiones (funciones del lenguaje)
-------------------------------------
    a_array(tabla [, [columnas]])   Tabla o lista de listas -> Array.
                                    Sin columnas, toma todas las numericas.
    a_tabla(array [, [nombres]])    Array o lista de listas -> DataFrame.
                                    Sin nombres, las columnas son c1, c2...
    a_listas(valor)                 Tabla o Array -> lista de listas.
    producto_mat(a, b)              producto de matrices entre cualquier
                                    combinacion de los tres tipos.

Operadores + - * / con tablas y arrays
--------------------------------------
    array  (op) array    elemento a elemento si tienen la misma forma.
                         Una fila (array 1D) con una matriz se aplica a
                         cada fila: X - medias resta la media de cada
                         columna.
    array  (op) numero   el numero se aplica a cada elemento.
    tabla  (op) numero   se aplica a cada columna numerica; las columnas
                         de texto quedan igual y los nulos siguen nulos.
    tabla  (op) tabla    columna por columna; deben tener las mismas
                         columnas y filas. Las de texto deben coincidir.
    tabla  (op) array    el array opera con las columnas numericas de la
                         tabla, en orden.
    -array, -tabla       cambia el signo de los numeros.

Las listas no entran aqui a proposito: [1, 2] + [3] sigue uniendo listas,
como antes. Para operar matrices hechas con listas se convierten con
array(...).

Este modulo, igual que operaciones.py, no conoce el arbol de ANTLR: lanza
ErrorDatos y el evaluador le agrega la linea.
"""

from src import terra_numpy as np
from src.terra_numpy import Array
from src.terra_pandas import DataFrame
from src.tabla import Tabla, ErrorDatos
from src.biblioteca import _registrar

OPERADORES = ("+", "-", "*", "/")


def es_matricial(valor) -> bool:
    """True para los valores que se operan con este modulo."""
    return isinstance(valor, (Array, Tabla))


def _es_numero(valor) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def _nombre(valor) -> str:
    if isinstance(valor, DataFrame):
        return "dataframe"
    if isinstance(valor, Tabla):
        return "tabla"
    if isinstance(valor, Array):
        return "array"
    if isinstance(valor, list):
        return "lista"
    if isinstance(valor, str):
        return "texto"
    if _es_numero(valor):
        return "numero"
    return "otro valor"


# =====================================================================
# OPERACION BASICA ENTRE DOS NUMEROS
# =====================================================================

def _aplicar(op: str, x, y):
    """Opera dos numeros; un nulo en cualquiera de los dos da nulo."""
    if x is None or y is None:
        return None
    if op == "+":
        return x + y
    if op == "-":
        return x - y
    if op == "*":
        return x * y
    if y == 0:
        raise ErrorDatos("division por cero dentro de la tabla o array")
    return x / y


# =====================================================================
# COLUMNAS NUMERICAS
# =====================================================================

def columnas_numericas(tabla) -> list:
    """Columnas cuyos valores (sin contar nulos) son todos numeros."""
    numericas = []
    for nombre in tabla.columnas:
        valores = [v for v in tabla.columna(nombre) if v is not None]
        if valores and all(_es_numero(v) for v in valores):
            numericas.append(nombre)
    return numericas


def _elegir_columnas(tabla, columnas, funcion) -> list:
    numericas = columnas_numericas(tabla)
    if columnas is None:
        if not numericas:
            raise ErrorDatos(f"{funcion}(): la tabla no tiene columnas "
                             f"numericas")
        return numericas
    if not isinstance(columnas, list) or not all(
            isinstance(c, str) for c in columnas):
        raise ErrorDatos(f"{funcion}(): las columnas deben ir en una lista "
                         f"de textos, por ejemplo [\"co2\", \"poblacion\"]")
    for c in columnas:
        if not tabla.tiene_columna(c):
            raise ErrorDatos(f"{funcion}(): la columna '{c}' no existe")
        if c not in numericas:
            raise ErrorDatos(f"{funcion}(): la columna '{c}' no es numerica")
    return columnas


# =====================================================================
# CONVERSIONES
# =====================================================================

def a_array(valor, columnas=None) -> Array:
    """Tabla, lista de listas o Array -> Array de decimales."""
    if isinstance(valor, Array):
        return valor
    if isinstance(valor, list):
        return np.array(valor)
    if not isinstance(valor, Tabla):
        raise ErrorDatos(f"a_array(): se esperaba una tabla o una lista, se "
                         f"recibio {_nombre(valor)}")
    if valor.num_filas == 0:
        raise ErrorDatos("a_array(): la tabla no tiene filas")

    nombres = _elegir_columnas(valor, columnas, "a_array")
    indices = [valor.indice(c) for c in nombres]
    datos = []
    for numero_fila, fila in enumerate(valor.filas, 1):
        for i, nombre in zip(indices, nombres):
            if fila[i] is None:
                raise ErrorDatos(
                    f"a_array(): la fila {numero_fila} tiene un nulo en "
                    f"'{nombre}'; usa '|> rellenar nulos' o "
                    f"'|> eliminar nulos' antes de convertir")
            datos.append(float(fila[i]))
    return Array(datos, (valor.num_filas, len(nombres)))


def a_tabla(valor, nombres=None) -> DataFrame:
    """Array o lista de listas -> DataFrame (que tambien es una Tabla)."""
    if isinstance(valor, Tabla):
        return valor
    if isinstance(valor, list):
        valor = np.array(valor)
    if not isinstance(valor, Array):
        raise ErrorDatos(f"a_tabla(): se esperaba un array o una lista, se "
                         f"recibio {_nombre(valor)}")

    filas = valor.como_listas() if valor.es_2d else \
        [[v] for v in valor.datos]
    total = len(filas[0]) if filas else 0

    if nombres is None:
        nombres = [f"c{i}" for i in range(1, total + 1)]
    if not isinstance(nombres, list) or not all(
            isinstance(n, str) for n in nombres):
        raise ErrorDatos("a_tabla(): los nombres deben ir en una lista de "
                         "textos")
    if len(nombres) != total:
        raise ErrorDatos(f"a_tabla(): el array tiene {total} columnas pero "
                         f"se dieron {len(nombres)} nombres")
    if len(set(nombres)) != len(nombres):
        raise ErrorDatos("a_tabla(): hay nombres de columna repetidos")
    return DataFrame(nombres, filas)


def a_listas(valor) -> list:
    """Tabla (columnas numericas) o Array -> lista de listas."""
    if isinstance(valor, list):
        return valor
    if isinstance(valor, Tabla):
        valor = a_array(valor)
    if isinstance(valor, Array):
        return valor.como_listas() if valor.es_2d else list(valor.datos)
    raise ErrorDatos(f"a_listas(): se esperaba una tabla o un array, se "
                     f"recibio {_nombre(valor)}")


def producto_mat(a, b) -> Array:
    """Producto de matrices A x B; acepta tablas, arrays y listas."""
    return np.producto_matricial(a_array(a), a_array(b))


# =====================================================================
# OPERADORES
# =====================================================================

def operar(op: str, a, b):
    """a (op) b cuando al menos uno de los dos es Tabla o Array."""
    if op not in OPERADORES:
        raise ErrorDatos(f"'{op}' no se puede usar con tablas o arrays; "
                         f"solo + - * /")
    for valor in (a, b):
        if not (es_matricial(valor) or _es_numero(valor)):
            raise ErrorDatos(
                f"no se puede operar {_nombre(a)} {op} {_nombre(b)}; si es "
                f"una matriz hecha con listas, conviertela con array(...)")
    if isinstance(a, Tabla) or isinstance(b, Tabla):
        return _operar_tablas(op, a, b)
    return _operar_arrays(op, a, b)


def _operar_arrays(op, a, b) -> Array:
    if _es_numero(b):
        return Array([_aplicar(op, x, b) for x in a.datos], a.forma)
    if _es_numero(a):
        return Array([_aplicar(op, a, y) for y in b.datos], b.forma)

    if a.forma == b.forma:
        return Array([_aplicar(op, x, y) for x, y in zip(a.datos, b.datos)],
                     a.forma)

    # Una fila contra una matriz: la fila se aplica a cada fila
    if a.es_2d and b.es_1d and a.forma[1] == b.forma[0]:
        columnas = a.forma[1]
        return Array([_aplicar(op, x, b.datos[i % columnas])
                      for i, x in enumerate(a.datos)], a.forma)
    if a.es_1d and b.es_2d and b.forma[1] == a.forma[0]:
        columnas = b.forma[1]
        return Array([_aplicar(op, a.datos[i % columnas], y)
                      for i, y in enumerate(b.datos)], b.forma)

    raise ErrorDatos(f"no se puede operar arrays de forma {a.forma} y "
                     f"{b.forma}: deben tener la misma forma, o uno debe "
                     f"ser una fila con tantos valores como columnas tiene "
                     f"el otro")


def _operar_tablas(op, a, b):
    if isinstance(a, Tabla) and isinstance(b, Tabla):
        return _tabla_con_tabla(op, a, b)

    tabla = a if isinstance(a, Tabla) else b
    otro = b if tabla is a else a
    tabla_a_la_izquierda = tabla is a

    def calcular(x, y):
        return _aplicar(op, x, y) if tabla_a_la_izquierda else \
            _aplicar(op, y, x)

    numericas = columnas_numericas(tabla)
    indices = [tabla.indice(c) for c in numericas]
    filas = [list(f) for f in tabla.filas]

    if _es_numero(otro):
        for fila in filas:
            for i in indices:
                fila[i] = calcular(fila[i], otro)
        return type(tabla)(tabla.columnas, filas)

    # tabla con array: el array opera con las columnas numericas
    arreglo = otro
    if arreglo.es_1d:
        if arreglo.forma[0] != len(indices):
            raise ErrorDatos(
                f"el array tiene {arreglo.forma[0]} valores pero la tabla "
                f"tiene {len(indices)} columnas numericas ({', '.join(numericas)})")
        for fila in filas:
            for k, i in enumerate(indices):
                fila[i] = calcular(fila[i], arreglo.datos[k])
    else:
        if arreglo.forma != (tabla.num_filas, len(indices)):
            raise ErrorDatos(
                f"el array es de forma {arreglo.forma} pero la parte "
                f"numerica de la tabla es de {tabla.num_filas} x "
                f"{len(indices)} ({', '.join(numericas)})")
        for r, fila in enumerate(filas):
            for k, i in enumerate(indices):
                fila[i] = calcular(fila[i], arreglo.obtener(r, k))
    return type(tabla)(tabla.columnas, filas)


def _tabla_con_tabla(op, a, b):
    if a.columnas != b.columnas:
        raise ErrorDatos(
            f"para operar dos tablas deben tener las mismas columnas en el "
            f"mismo orden: {a.columnas} y {b.columnas}")
    if a.num_filas != b.num_filas:
        raise ErrorDatos(
            f"para operar dos tablas deben tener las mismas filas: "
            f"{a.num_filas} y {b.num_filas}")

    numericas_a = set(columnas_numericas(a))
    numericas_b = set(columnas_numericas(b))
    filas = []
    for fila_a, fila_b in zip(a.filas, b.filas):
        nueva = []
        for nombre, x, y in zip(a.columnas, fila_a, fila_b):
            if nombre in numericas_a and nombre in numericas_b:
                nueva.append(_aplicar(op, x, y))
            elif x == y:
                nueva.append(x)
            else:
                raise ErrorDatos(
                    f"la columna '{nombre}' no es numerica y sus valores no "
                    f"coinciden ('{x}' y '{y}'): no se puede operar")
        filas.append(nueva)
    return type(a)(a.columnas, filas)


# =====================================================================
# REGISTRO EN EL LENGUAJE
# =====================================================================

def funciones() -> dict:
    """Funciones que este modulo agrega al lenguaje."""
    return _registrar([
        ("a_array",      a_array,      1, 2),
        ("a_tabla",      a_tabla,      1, 2),
        ("a_listas",     a_listas,     1),
        ("producto_mat", producto_mat, 2),
    ])
