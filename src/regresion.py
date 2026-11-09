"""
regresion.py
============
Implementacion de regresion lineal simple y multiple desde cero.

Regresion lineal simple:   y = a + b*x
Regresion lineal multiple: y = b0 + b1*x1 + b2*x2 + ...

No usa numpy, scipy ni sklearn. Solo aritmetica de Python puro.
"""

from src.tabla import ErrorDatos


# =====================================================================
# REGRESION LINEAL SIMPLE
# =====================================================================

def _medias(xs: list, ys: list):
    n = len(xs)
    return sum(xs) / n, sum(ys) / n


def _coeficientes_simple(xs: list, ys: list):
    """
    Calcula la pendiente (b) y el intercepto (a) por minimos cuadrados.

    b = sum((xi - x_media) * (yi - y_media)) / sum((xi - x_media)^2)
    a = y_media - b * x_media
    """
    n = len(xs)
    if n < 2:
        raise ErrorDatos(
            "regresion_lineal() necesita al menos 2 observaciones")

    x_media, y_media = _medias(xs, ys)

    numerador   = sum((xs[i] - x_media) * (ys[i] - y_media) for i in range(n))
    denominador = sum((xs[i] - x_media) ** 2 for i in range(n))

    if denominador == 0:
        raise ErrorDatos(
            "regresion_lineal(): todos los valores de X son iguales, "
            "no se puede ajustar una recta")

    b = numerador / denominador
    a = y_media - b * x_media
    return a, b


def _r_cuadrado(xs: list, ys: list, a: float, b: float) -> float:
    """
    Coeficiente de determinacion R^2.

    R^2 = 1 - SS_res / SS_tot
    SS_res = sum((yi - y_pred)^2)
    SS_tot = sum((yi - y_media)^2)
    """
    y_media = sum(ys) / len(ys)
    ss_res  = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(len(xs)))
    ss_tot  = sum((yi - y_media) ** 2 for yi in ys)
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


def entrenar_simple(xs: list, ys: list) -> dict:
    """
    Entrena un modelo de regresion lineal simple.

    Devuelve un diccionario con:
        tipo         : "simple"
        intercepto   : valor de a
        pendiente    : valor de b
        r_cuadrado   : bondad de ajuste
        n            : numero de observaciones
    """
    _validar_listas(xs, ys, "regresion_lineal")
    a, b = _coeficientes_simple(xs, ys)
    r2   = _r_cuadrado(xs, ys, a, b)
    return {
        "tipo":       "simple",
        "intercepto": a,
        "pendiente":  b,
        "r_cuadrado": r2,
        "n":          len(xs),
    }


def predecir_simple(modelo: dict, x) -> float:
    """y = intercepto + pendiente * x"""
    _validar_modelo(modelo, "simple", "predecir")
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise ErrorDatos(
            "predecir() espera un numero como valor de X")
    return modelo["intercepto"] + modelo["pendiente"] * x


# =====================================================================
# REGRESION LINEAL MULTIPLE
# =====================================================================

def _transponer(matriz: list) -> list:
    """Transpone una matriz representada como lista de listas."""
    filas = len(matriz)
    cols  = len(matriz[0])
    return [[matriz[f][c] for f in range(filas)] for c in range(cols)]


def _multiplicar(A: list, B: list) -> list:
    """Multiplicacion de matrices (listas de listas)."""
    filas_A = len(A)
    cols_A  = len(A[0])
    cols_B  = len(B[0])
    C = [[0.0] * cols_B for _ in range(filas_A)]
    for i in range(filas_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C


def _invertir(M: list) -> list:
    """
    Invierte una matriz cuadrada por eliminacion de Gauss-Jordan.
    Lanza ErrorDatos si la matriz es singular.
    """
    n = len(M)
    # Copia aumentada [M | I]
    aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]

    for col in range(n):
        # Busca el pivote maximo en esta columna
        max_fila = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_fila] = aug[max_fila], aug[col]

        pivote = aug[col][col]
        if abs(pivote) < 1e-12:
            raise ErrorDatos(
                "regresion_multiple(): las variables predictoras son "
                "linealmente dependientes (matriz singular)")

        # Escala la fila del pivote
        aug[col] = [v / pivote for v in aug[col]]

        # Elimina la columna en las demas filas
        for fila in range(n):
            if fila != col:
                factor = aug[fila][col]
                aug[fila] = [aug[fila][k] - factor * aug[col][k]
                             for k in range(2 * n)]

    return [fila[n:] for fila in aug]


def entrenar_multiple(X: list, ys: list) -> dict:
    """
    Entrena un modelo de regresion lineal multiple.

    X  : lista de listas — cada sublista es una observacion [x1, x2, ...]
    ys : lista de valores objetivo

    Resuelve: beta = (X^T X)^{-1} X^T y

    Devuelve un diccionario con:
        tipo         : "multiple"
        coeficientes : [b0_intercepto, b1, b2, ...]
        r_cuadrado   : bondad de ajuste
        n            : numero de observaciones
        p            : numero de predictores (sin el intercepto)
    """
    n = len(ys)
    if n < 2:
        raise ErrorDatos(
            "regresion_multiple() necesita al menos 2 observaciones")
    if len(X) != n:
        raise ErrorDatos(
            "regresion_multiple(): X e y deben tener el mismo numero "
            "de filas")

    # Agrega columna de unos para el intercepto
    X_aug = [[1.0] + [float(v) for v in fila] for fila in X]
    y_col = [[float(v)] for v in ys]

    Xt      = _transponer(X_aug)
    XtX     = _multiplicar(Xt, X_aug)
    XtX_inv = _invertir(XtX)
    Xty     = _multiplicar(Xt, y_col)
    beta    = _multiplicar(XtX_inv, Xty)
    coefs   = [fila[0] for fila in beta]

    # R^2
    y_media = sum(ys) / n
    ss_res  = sum((ys[i] - sum(coefs[j] * X_aug[i][j]
                               for j in range(len(coefs)))) ** 2
                  for i in range(n))
    ss_tot  = sum((yi - y_media) ** 2 for yi in ys)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot

    return {
        "tipo":         "multiple",
        "coeficientes": coefs,
        "r_cuadrado":   r2,
        "n":            n,
        "p":            len(X[0]),
    }


def predecir_multiple(modelo: dict, xs: list) -> float:
    """y = b0 + b1*x1 + b2*x2 + ..."""
    _validar_modelo(modelo, "multiple", "predecir_multiple")
    coefs = modelo["coeficientes"]
    if len(xs) != len(coefs) - 1:
        raise ErrorDatos(
            f"predecir_multiple() espera {len(coefs) - 1} predictores, "
            f"se recibieron {len(xs)}")
    return coefs[0] + sum(coefs[i + 1] * xs[i] for i in range(len(xs)))


# =====================================================================
# VALIDACIONES COMPARTIDAS
# =====================================================================

def _validar_listas(xs: list, ys: list, funcion: str):
    if not isinstance(xs, list) or not isinstance(ys, list):
        raise ErrorDatos(f"{funcion}() espera dos listas")
    if len(xs) != len(ys):
        raise ErrorDatos(
            f"{funcion}(): X e y deben tener el mismo numero de elementos "
            f"(X={len(xs)}, y={len(ys)})")
    for v in xs + ys:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ErrorDatos(
                f"{funcion}(): todas las listas deben contener solo numeros")


def _validar_modelo(modelo: dict, tipo_esperado: str, funcion: str):
    if not isinstance(modelo, dict) or modelo.get("tipo") != tipo_esperado:
        raise ErrorDatos(
            f"{funcion}() recibio un modelo incorrecto; "
            f"usa regresion_lineal() o regresion_multiple() primero")