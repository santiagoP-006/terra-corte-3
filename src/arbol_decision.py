"""
arbol_decision.py
=================
Implementacion de un arbol de decision para clasificacion (CART)
escrito desde cero, sin sklearn ni numpy.

Algoritmo: CART (Classification and Regression Trees)
Criterio de division: indice de Gini
Estrategia de parada:
    - profundidad maxima alcanzada
    - nodo con menos ejemplos que min_muestras
    - nodo puro (una sola clase)

Representacion interna del arbol:
    Cada nodo es un diccionario con una de dos formas:

    Nodo hoja:
        { "hoja": True, "clase": valor, "conteo": {clase: n, ...} }

    Nodo interno:
        { "hoja": False,
          "columna": int,          # indice de la columna de division
          "umbral": float/str,     # valor de corte
          "izquierda": nodo,       # <= umbral (numerico) o == umbral (texto)
          "derecha": nodo,         # >  umbral (numerico) o != umbral (texto)
          "conteo": {clase: n, ...}
        }
"""

from src.tabla import ErrorDatos


# =====================================================================
# GINI Y DIVISION
# =====================================================================

def _conteo_clases(ys: list) -> dict:
    """Frecuencia de cada clase en la lista."""
    conteo = {}
    for y in ys:
        conteo[y] = conteo.get(y, 0) + 1
    return conteo


def _gini(ys: list) -> float:
    """
    Indice de Gini de un conjunto de etiquetas.
    gini = 1 - sum(p_i^2)
    0.0 significa nodo puro; maximo en distribucion uniforme.
    """
    n = len(ys)
    if n == 0:
        return 0.0
    conteo = _conteo_clases(ys)
    return 1.0 - sum((c / n) ** 2 for c in conteo.values())


def _gini_ponderado(izq: list, der: list) -> float:
    """Gini ponderado de una division en dos grupos."""
    n = len(izq) + len(der)
    if n == 0:
        return 0.0
    return (len(izq) / n) * _gini(izq) + (len(der) / n) * _gini(der)


def _dividir(X: list, ys: list, col: int, umbral):
    """
    Divide el conjunto en dos subconjuntos segun columna y umbral.

    Para columnas numericas: izquierda <= umbral, derecha > umbral.
    Para columnas de texto:  izquierda == umbral, derecha != umbral.
    """
    izq_X, izq_y, der_X, der_y = [], [], [], []
    for i in range(len(X)):
        valor = X[i][col]
        if isinstance(umbral, str):
            if valor == umbral:
                izq_X.append(X[i]); izq_y.append(ys[i])
            else:
                der_X.append(X[i]); der_y.append(ys[i])
        else:
            if valor is not None and valor <= umbral:
                izq_X.append(X[i]); izq_y.append(ys[i])
            else:
                der_X.append(X[i]); der_y.append(ys[i])
    return izq_X, izq_y, der_X, der_y


def _mejor_division(X: list, ys: list) -> tuple:
    """
    Busca la columna y el umbral que minimiza el Gini ponderado.

    Devuelve (col, umbral, gini_minimo) o (None, None, inf) si no
    hay ninguna division util.
    """
    mejor_col    = None
    mejor_umbral = None
    mejor_gini   = float("inf")
    n_cols       = len(X[0])

    for col in range(n_cols):
        # Candidatos: valores unicos no nulos de esta columna
        valores = sorted({X[i][col] for i in range(len(X))
                          if X[i][col] is not None})

        for umbral in valores:
            _, izq_y, _, der_y = _dividir(X, ys, col, umbral)
            if not izq_y or not der_y:
                continue
            g = _gini_ponderado(izq_y, der_y)
            if g < mejor_gini:
                mejor_gini   = g
                mejor_col    = col
                mejor_umbral = umbral

    return mejor_col, mejor_umbral, mejor_gini


# =====================================================================
# CONSTRUCCION DEL ARBOL
# =====================================================================

def _clase_mayoritaria(ys: list):
    """Clase con mas ocurrencias; en empate gana la primera en orden."""
    conteo = _conteo_clases(ys)
    return max(conteo, key=lambda c: (conteo[c], str(c)))


def _construir(X: list, ys: list, profundidad: int,
               max_prof: int, min_muestras: int) -> dict:
    """Construye recursivamente el arbol."""
    conteo = _conteo_clases(ys)

    # Condiciones de parada
    if (profundidad >= max_prof
            or len(ys) < min_muestras
            or len(conteo) == 1):
        return {"hoja": True,
                "clase": _clase_mayoritaria(ys),
                "conteo": conteo}

    col, umbral, _ = _mejor_division(X, ys)

    # Si no se encontro ninguna division util
    if col is None:
        return {"hoja": True,
                "clase": _clase_mayoritaria(ys),
                "conteo": conteo}

    izq_X, izq_y, der_X, der_y = _dividir(X, ys, col, umbral)

    # Si la division no separa nada
    if not izq_y or not der_y:
        return {"hoja": True,
                "clase": _clase_mayoritaria(ys),
                "conteo": conteo}

    return {
        "hoja":      False,
        "columna":   col,
        "umbral":    umbral,
        "izquierda": _construir(izq_X, izq_y, profundidad + 1,
                                max_prof, min_muestras),
        "derecha":   _construir(der_X, der_y, profundidad + 1,
                                max_prof, min_muestras),
        "conteo":    conteo,
    }


# =====================================================================
# API PUBLICA
# =====================================================================

def entrenar_arbol(X: list, ys: list,
                   max_profundidad: int = 5,
                   min_muestras: int = 2) -> dict:
    """
    Entrena un arbol de decision para clasificacion.

    Parametros:
        X               : lista de listas — cada sublista es una
                          observacion [x1, x2, ...]
        ys              : lista de etiquetas (clases)
        max_profundidad : profundidad maxima del arbol (default 5)
        min_muestras    : minimo de muestras para dividir un nodo (default 2)

    Devuelve un modelo (diccionario) con:
        tipo            : "arbol_decision"
        raiz            : nodo raiz del arbol
        clases          : lista de clases unicas
        n               : numero de observaciones de entrenamiento
        p               : numero de columnas (predictores)
        max_profundidad : hiperparametro usado
        min_muestras    : hiperparametro usado
    """
    _validar_entrenamiento(X, ys)

    raiz = _construir(X, ys, 0, max_profundidad, min_muestras)

    return {
        "tipo":            "arbol_decision",
        "raiz":            raiz,
        "clases":          sorted({str(y) for y in ys}),
        "n":               len(ys),
        "p":               len(X[0]),
        "max_profundidad": max_profundidad,
        "min_muestras":    min_muestras,
    }


def predecir_clase(modelo: dict, x: list):
    """
    Clasifica una observacion recorriendo el arbol.

    Parametros:
        modelo : resultado de entrenar_arbol()
        x      : lista con los valores de los predictores

    Devuelve la clase predicha.
    """
    _validar_modelo(modelo)
    if len(x) != modelo["p"]:
        raise ErrorDatos(
            f"clasificar() espera {modelo['p']} predictores, "
            f"se recibieron {len(x)}")

    nodo = modelo["raiz"]
    while not nodo["hoja"]:
        col     = nodo["columna"]
        umbral  = nodo["umbral"]
        valor   = x[col]

        if isinstance(umbral, str):
            nodo = nodo["izquierda"] if valor == umbral else nodo["derecha"]
        else:
            if valor is None or valor > umbral:
                nodo = nodo["derecha"]
            else:
                nodo = nodo["izquierda"]

    return nodo["clase"]


def predecir_lote(modelo: dict, X: list) -> list:
    """
    Clasifica multiples observaciones de una vez.
    Devuelve una lista de clases predichas, una por fila de X.
    """
    _validar_modelo(modelo)
    return [predecir_clase(modelo, fila) for fila in X]


def evaluar_modelo(modelo: dict, X: list, ys: list) -> dict:
    """
    Calcula metricas de evaluacion sobre un conjunto de prueba.

    Devuelve un diccionario con:
        exactitud       : proporcion de predicciones correctas
        correctas       : numero de predicciones correctas
        total           : total de observaciones evaluadas
        matriz          : matriz de confusion {clase_real: {clase_pred: n}}
    """
    _validar_modelo(modelo)
    if len(X) != len(ys):
        raise ErrorDatos(
            "evaluar_arbol(): X e y deben tener el mismo numero de filas")

    predicciones = predecir_lote(modelo, X)
    correctas    = sum(1 for p, r in zip(predicciones, ys) if p == r)

    # Matriz de confusion
    matriz = {}
    for real, pred in zip(ys, predicciones):
        real_str = str(real)
        pred_str = str(pred)
        if real_str not in matriz:
            matriz[real_str] = {}
        matriz[real_str][pred_str] = matriz[real_str].get(pred_str, 0) + 1

    return {
        "exactitud": correctas / len(ys) if ys else 0.0,
        "correctas": correctas,
        "total":     len(ys),
        "matriz":    matriz,
    }


def _profundidad_real(nodo: dict) -> int:
    """Calcula la profundidad real del arbol construido."""
    if nodo["hoja"]:
        return 0
    return 1 + max(_profundidad_real(nodo["izquierda"]),
                   _profundidad_real(nodo["derecha"]))


def _contar_nodos(nodo: dict) -> tuple:
    """Devuelve (nodos_internos, hojas)."""
    if nodo["hoja"]:
        return 0, 1
    izq_i, izq_h = _contar_nodos(nodo["izquierda"])
    der_i, der_h = _contar_nodos(nodo["derecha"])
    return 1 + izq_i + der_i, izq_h + der_h


def _imprimir_arbol(nodo: dict, nombres_col: list = None,
                    prefijo: str = "", es_izquierda: bool = True,
                    profundidad: int = 0) -> str:
    """
    Genera una representacion en texto del arbol para mostrar().
    """
    lineas = []
    conector = "├── " if es_izquierda else "└── "

    if nodo["hoja"]:
        total = sum(nodo["conteo"].values())
        dist  = ", ".join(f"{k}:{v}" for k, v in
                          sorted(nodo["conteo"].items(), key=lambda x: str(x[0])))
        lineas.append(f"{prefijo}{conector}HOJA → {nodo['clase']} "
                      f"[{dist}] (n={total})")
    else:
        col_nombre = (nombres_col[nodo["columna"]]
                      if nombres_col and nodo["columna"] < len(nombres_col)
                      else f"col{nodo['columna']}")
        total = sum(nodo["conteo"].values())

        if isinstance(nodo["umbral"], str):
            condicion = f"{col_nombre} == '{nodo['umbral']}'"
        else:
            condicion = f"{col_nombre} <= {nodo['umbral']}"

        lineas.append(f"{prefijo}{conector}[{condicion}] (n={total})")

        nuevo_prefijo = prefijo + ("│   " if es_izquierda else "    ")
        lineas.append(_imprimir_arbol(nodo["izquierda"],  nombres_col,
                                      nuevo_prefijo, True,  profundidad + 1))
        lineas.append(_imprimir_arbol(nodo["derecha"],    nombres_col,
                                      nuevo_prefijo, False, profundidad + 1))

    return "\n".join(lineas)


# =====================================================================
# VALIDACIONES
# =====================================================================

def _validar_entrenamiento(X: list, ys: list):
    if not isinstance(X, list) or not isinstance(ys, list):
        raise ErrorDatos("arbol_decision() espera dos listas: X e ys")
    if len(X) == 0 or len(ys) == 0:
        raise ErrorDatos("arbol_decision(): X e ys no pueden estar vacios")
    if len(X) != len(ys):
        raise ErrorDatos(
            f"arbol_decision(): X tiene {len(X)} filas pero "
            f"ys tiene {len(ys)} elementos")
    if not isinstance(X[0], list):
        raise ErrorDatos(
            "arbol_decision(): X debe ser una lista de listas "
            "(cada fila es una observacion)")
    p = len(X[0])
    for i, fila in enumerate(X):
        if len(fila) != p:
            raise ErrorDatos(
                f"arbol_decision(): la fila {i} tiene {len(fila)} "
                f"columnas pero se esperaban {p}")


def _validar_modelo(modelo: dict):
    if not isinstance(modelo, dict) or modelo.get("tipo") != "arbol_decision":
        raise ErrorDatos(
            "se esperaba un modelo entrenado con arbol_decision(); "
            "usa primero arbol_decision(X, ys)")