"""
aprendizaje.py
==============
Machine Learning de Terra, escrito desde cero (sin scikit-learn, numpy
ni ninguna libreria).

Preparacion de datos
--------------------
    dividir_datos(X, y [, prueba] [, semilla])
        -> [X_entrenamiento, X_prueba, y_entrenamiento, y_prueba]
    normalizar(X [, referencia])     cada columna a [0, 1]  (min-max)
    estandarizar(X [, referencia])   cada columna con media 0 y desv. 1

Modelos
-------
    knn(X, y [, k])                  vecinos mas cercanos, clasificacion
    knn_regresion(X, y [, k])        vecinos mas cercanos, regresion
    kmeans(X, k [, semilla] [, max_iteraciones])     agrupamiento
    perceptron_clasificador(X, y [, ocultas] [, epocas] [, tasa] [, semilla])
    perceptron_regresor(X, y [, ocultas] [, epocas] [, tasa] [, semilla])

    aplicar_modelo(modelo, X)        predicciones de CUALQUIER modelo de
                                     Terra (tambien regresion y arbol)

Metricas
--------
    Regresion:      mse, rmse, mae, r2              (reales, predichos)
    Clasificacion:  exactitud, precision, recall, f1, matriz_confusion,
                    reporte_clasificacion           (reales, predichos)

En todas las funciones X puede ser una tabla (se usan sus columnas
numericas), un Array o una lista de listas; y puede ser una lista, un
Array de una dimension o una columna.

Los modelos son diccionarios con un campo "tipo", igual que los de
regresion.py y arbol_decision.py, para que el resto del lenguaje
(mostrar, memoria_info) los trate igual.
"""

from src import matematicas as m
from src import regresion as reg
from src import arbol_decision as ad
from src.biblioteca import _registrar
from src.matematicas import GeneradorAleatorio
from src.tabla import Tabla, ErrorDatos
from src.terra_numpy import Array


# =====================================================================
# ENTRADA: CONVERTIR X e y A LISTAS
# =====================================================================

def _es_numero(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _columnas_numericas(tabla) -> list:
    numericas = []
    for nombre in tabla.columnas:
        valores = [v for v in tabla.columna(nombre) if v is not None]
        if valores and all(_es_numero(v) for v in valores):
            numericas.append(nombre)
    return numericas


def _matriz(X, funcion: str) -> list:
    """X como lista de filas de decimales, sin nulos."""
    if isinstance(X, Tabla):
        columnas = _columnas_numericas(X)
        if not columnas:
            raise ErrorDatos(f"{funcion}(): la tabla no tiene columnas "
                             f"numericas")
        indices = [X.indice(c) for c in columnas]
        filas = [[fila[i] for i in indices] for fila in X.filas]
    elif isinstance(X, Array):
        filas = X.como_listas() if X.es_2d else [[v] for v in X.datos]
    elif isinstance(X, list):
        filas = [f if isinstance(f, list) else [f] for f in X]
    else:
        raise ErrorDatos(f"{funcion}(): X debe ser una tabla, un array o "
                         f"una lista de listas")
    if not filas:
        raise ErrorDatos(f"{funcion}(): no hay datos (X esta vacio)")
    ancho = len(filas[0])
    for numero, fila in enumerate(filas, 1):
        if len(fila) != ancho:
            raise ErrorDatos(f"{funcion}(): la fila {numero} de X tiene "
                             f"{len(fila)} valores y la primera {ancho}")
        for v in fila:
            if v is None:
                raise ErrorDatos(
                    f"{funcion}(): la fila {numero} de X tiene un nulo; usa "
                    f"'|> rellenar nulos' o '|> eliminar nulos' antes")
            if not _es_numero(v):
                raise ErrorDatos(f"{funcion}(): X solo puede tener numeros, "
                                 f"se encontro '{v}'")
    return [[float(v) for v in fila] for fila in filas]


def _vector(y, funcion: str, numerico: bool = False) -> list:
    """y como lista. Acepta lista, Array 1D o tabla de una columna."""
    if isinstance(y, Array):
        y = list(y.datos)
    elif isinstance(y, Tabla):
        if y.num_columnas != 1:
            raise ErrorDatos(f"{funcion}(): y debe ser una sola columna")
        y = y.columna(y.columnas[0])
    if not isinstance(y, list) or not y:
        raise ErrorDatos(f"{funcion}(): y debe ser una lista con valores")
    if any(v is None for v in y):
        raise ErrorDatos(f"{funcion}(): y tiene nulos")
    if numerico:
        if not all(_es_numero(v) for v in y):
            raise ErrorDatos(f"{funcion}(): y debe tener solo numeros")
        return [float(v) for v in y]
    return list(y)


def _mismo_largo(X, y, funcion: str):
    if len(X) != len(y):
        raise ErrorDatos(f"{funcion}(): X tiene {len(X)} filas pero y tiene "
                         f"{len(y)} valores")


def _entero(valor, funcion: str, que: str, minimo: int = 1) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int) \
            or valor < minimo:
        raise ErrorDatos(f"{funcion}(): {que} debe ser un entero mayor o "
                         f"igual a {minimo}")
    return valor


def _semilla(valor, funcion: str) -> GeneradorAleatorio:
    return GeneradorAleatorio(_entero(valor, funcion, "la semilla", 0))


def _barajar(n: int, generador) -> list:
    """Posiciones 0..n-1 en orden aleatorio (Fisher-Yates)."""
    posiciones = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(generador.siguiente() * (i + 1))
        posiciones[i], posiciones[j] = posiciones[j], posiciones[i]
    return posiciones


# =====================================================================
# PREPARACION DE DATOS
# =====================================================================

def dividir_datos(X, y, prueba=0.25, semilla=42) -> list:
    """
    Separa los datos en entrenamiento y prueba al azar (pero siempre igual
    con la misma semilla). prueba es la proporcion que va a prueba.

    Si X es una tabla, X_entrenamiento y X_prueba tambien son tablas, con
    todas sus columnas.
    """
    filas_y = _vector(y, "dividir_datos")
    if isinstance(X, Tabla):
        filas_x = X.filas
    else:
        filas_x = _matriz(X, "dividir_datos")
    _mismo_largo(filas_x, filas_y, "dividir_datos")
    if not _es_numero(prueba) or not 0 < prueba < 1:
        raise ErrorDatos("dividir_datos(): la proporcion de prueba debe "
                         "estar entre 0 y 1, por ejemplo 0.25")
    n = len(filas_y)
    if n < 2:
        raise ErrorDatos("dividir_datos(): se necesitan al menos 2 filas")

    orden = _barajar(n, _semilla(semilla, "dividir_datos"))
    cantidad = min(max(int(round(n * prueba)), 1), n - 1)
    de_prueba = sorted(orden[:cantidad])
    de_entrenamiento = sorted(orden[cantidad:])

    def tomar(filas, posiciones):
        return [list(filas[i]) if isinstance(filas[i], list) else filas[i]
                for i in posiciones]

    x_ent = tomar(filas_x, de_entrenamiento)
    x_pru = tomar(filas_x, de_prueba)
    if isinstance(X, Tabla):
        x_ent = type(X)(X.columnas, x_ent)
        x_pru = type(X)(X.columnas, x_pru)
    return [x_ent, x_pru,
            [filas_y[i] for i in de_entrenamiento],
            [filas_y[i] for i in de_prueba]]


def _escalar(X, referencia, funcion, calcular):
    """
    Aplica una transformacion columna por columna. Los parametros (minimo
    y maximo, o media y desviacion) salen de 'referencia' si se da: asi
    los datos de prueba se escalan con los numeros del entrenamiento.
    """
    filas = _matriz(X, funcion)
    base = _matriz(referencia, funcion) if referencia is not None else filas
    if len(base[0]) != len(filas[0]):
        raise ErrorDatos(f"{funcion}(): X y la referencia deben tener las "
                         f"mismas columnas")
    columnas = len(filas[0])
    parametros = [calcular([fila[j] for fila in base])
                  for j in range(columnas)]
    nuevas = [[_aplicar_escala(fila[j], parametros[j])
               for j in range(columnas)] for fila in filas]
    return _reconstruir(X, nuevas)


def _aplicar_escala(valor, parametros):
    centro, ancho = parametros
    return 0.0 if ancho == 0 else (valor - centro) / ancho


def _reconstruir(original, filas):
    """Devuelve el resultado en el mismo tipo que llego X."""
    if isinstance(original, Array):
        return Array([v for f in filas for v in f],
                     (len(filas), len(filas[0])))
    if isinstance(original, Tabla):
        columnas = _columnas_numericas(original)
        indices = [original.indice(c) for c in columnas]
        nuevas = [list(f) for f in original.filas]
        for r, fila in enumerate(nuevas):
            for k, i in enumerate(indices):
                fila[i] = filas[r][k]
        return type(original)(original.columnas, nuevas)
    return filas


def normalizar(X, referencia=None):
    """Min-max: (x - minimo) / (maximo - minimo), cada columna a [0, 1]."""
    return _escalar(X, referencia, "normalizar",
                    lambda col: (min(col), max(col) - min(col)))


def _media(valores):
    return sum(valores) / len(valores)


def _desviacion(valores):
    """Desviacion estandar poblacional (dividida por n)."""
    media = _media(valores)
    return (sum((v - media) ** 2 for v in valores) / len(valores)) ** 0.5


def estandarizar(X, referencia=None):
    """Puntaje z: (x - media) / desviacion, cada columna."""
    return _escalar(X, referencia, "estandarizar",
                    lambda col: (_media(col), _desviacion(col)))


# =====================================================================
# METRICAS
# =====================================================================

def _pares_numericos(reales, predichos, funcion):
    r = _vector(reales, funcion, numerico=True)
    p = _vector(predichos, funcion, numerico=True)
    _mismo_largo(r, p, funcion)
    return r, p


def mse(reales, predichos) -> float:
    """Error cuadratico medio: promedio de (real - predicho)^2."""
    r, p = _pares_numericos(reales, predichos, "mse")
    return sum((a - b) ** 2 for a, b in zip(r, p)) / len(r)


def rmse(reales, predichos) -> float:
    """Raiz del MSE: el error en las mismas unidades que los datos."""
    return mse(reales, predichos) ** 0.5


def mae(reales, predichos) -> float:
    """Error absoluto medio: promedio de |real - predicho|."""
    r, p = _pares_numericos(reales, predichos, "mae")
    return sum(abs(a - b) for a, b in zip(r, p)) / len(r)


def r2(reales, predichos) -> float:
    """
    Coeficiente de determinacion: 1 - SS_residuos / SS_total.
    1 es perfecto; 0 es igual que predecir siempre la media.
    """
    r, p = _pares_numericos(reales, predichos, "r2")
    media = _media(r)
    ss_total = sum((a - media) ** 2 for a in r)
    ss_residuos = sum((a - b) ** 2 for a, b in zip(r, p))
    if ss_total == 0:
        return 1.0 if ss_residuos == 0 else 0.0
    return 1 - ss_residuos / ss_total


def _pares_clases(reales, predichos, funcion):
    r = _vector(reales, funcion)
    p = _vector(predichos, funcion)
    _mismo_largo(r, p, funcion)
    return r, p


def _clases(r, p) -> list:
    vistas = []
    for c in r + p:
        if c not in vistas:
            vistas.append(c)
    return vistas


def exactitud(reales, predichos=None) -> float:
    """
    Proporcion de aciertos.

    Con un solo argumento acepta tambien el resultado de evaluar_arbol(),
    como en el segundo corte.
    """
    if predichos is None:
        if isinstance(reales, dict) and "exactitud" in reales:
            return reales["exactitud"]
        raise ErrorDatos("exactitud() espera (reales, predichos) o el "
                         "resultado de evaluar_arbol()")
    r, p = _pares_clases(reales, predichos, "exactitud")
    return sum(1 for a, b in zip(r, p) if a == b) / len(r)


def _conteos(r, p, clase):
    vp = sum(1 for a, b in zip(r, p) if a == clase and b == clase)
    fp = sum(1 for a, b in zip(r, p) if a != clase and b == clase)
    fn = sum(1 for a, b in zip(r, p) if a == clase and b != clase)
    return vp, fp, fn


def _precision_clase(r, p, clase):
    vp, fp, _ = _conteos(r, p, clase)
    return vp / (vp + fp) if vp + fp else 0.0


def _recall_clase(r, p, clase):
    vp, _, fn = _conteos(r, p, clase)
    return vp / (vp + fn) if vp + fn else 0.0


def _f1_clase(r, p, clase):
    pr = _precision_clase(r, p, clase)
    re = _recall_clase(r, p, clase)
    return 2 * pr * re / (pr + re) if pr + re else 0.0


def _promedio_o_clase(funcion_clase, reales, predichos, clase, nombre):
    """
    Con clase: la metrica de esa clase. Sin clase: el promedio "macro",
    es decir, el promedio simple de la metrica de cada clase real.
    """
    r, p = _pares_clases(reales, predichos, nombre)
    if clase is not None:
        return funcion_clase(r, p, clase)
    clases = _clases(r, [])
    return sum(funcion_clase(r, p, c) for c in clases) / len(clases)


def precision(reales, predichos, clase=None) -> float:
    """De los que el modelo dijo que eran 'clase', cuantos lo eran."""
    return _promedio_o_clase(_precision_clase, reales, predichos, clase,
                             "precision")


def recall(reales, predichos, clase=None) -> float:
    """De los que de verdad eran 'clase', cuantos encontro el modelo."""
    return _promedio_o_clase(_recall_clase, reales, predichos, clase,
                             "recall")


def f1(reales, predichos, clase=None) -> float:
    """Media armonica de precision y recall."""
    return _promedio_o_clase(_f1_clase, reales, predichos, clase, "f1")


def matriz_confusion(reales, predichos) -> Tabla:
    """
    Tabla con una fila por clase real y una columna por clase predicha:
    cada celda cuenta cuantas veces ocurrio esa combinacion. La diagonal
    son los aciertos.
    """
    r, p = _pares_clases(reales, predichos, "matriz_confusion")
    clases = _clases(r, p)
    filas = []
    for real in clases:
        fila = [str(real)]
        for pred in clases:
            fila.append(sum(1 for a, b in zip(r, p)
                            if a == real and b == pred))
        filas.append(fila)
    return Tabla(["real"] + [str(c) for c in clases], filas)


def reporte_clasificacion(reales, predichos) -> Tabla:
    """Precision, recall, f1 y soporte de cada clase, y el promedio."""
    r, p = _pares_clases(reales, predichos, "reporte_clasificacion")
    filas = []
    for c in _clases(r, []):
        filas.append([str(c),
                      round(_precision_clase(r, p, c), 4),
                      round(_recall_clase(r, p, c), 4),
                      round(_f1_clase(r, p, c), 4),
                      sum(1 for a in r if a == c)])
    total = len(filas)
    filas.append(["promedio",
                  round(sum(f[1] for f in filas) / total, 4),
                  round(sum(f[2] for f in filas) / total, 4),
                  round(sum(f[3] for f in filas) / total, 4),
                  len(r)])
    return Tabla(["clase", "precision", "recall", "f1", "soporte"], filas)


# =====================================================================
# K VECINOS MAS CERCANOS (KNN)
# =====================================================================

def _distancia2(a, b) -> float:
    """Distancia euclidiana AL CUADRADO. Para saber cual punto esta mas
    cerca no hace falta sacar la raiz: el orden es el mismo."""
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _knn(X, y, k, tarea, funcion):
    filas = _matriz(X, funcion)
    etiquetas = _vector(y, funcion, numerico=(tarea == "regresion"))
    _mismo_largo(filas, etiquetas, funcion)
    k = _entero(k, funcion, "k")
    if k > len(filas):
        raise ErrorDatos(f"{funcion}(): k = {k} es mayor que la cantidad de "
                         f"datos de entrenamiento ({len(filas)})")
    # KNN no "aprende" nada: el modelo es guardar los datos
    return {"tipo": "knn", "tarea": tarea, "k": k, "X": filas,
            "y": etiquetas, "n": len(filas), "p": len(filas[0])}


def knn(X, y, k=3) -> dict:
    """KNN para clasificacion: vota la clase de los k vecinos."""
    return _knn(X, y, k, "clasificacion", "knn")


def knn_regresion(X, y, k=3) -> dict:
    """KNN para regresion: promedia el valor de los k vecinos."""
    return _knn(X, y, k, "regresion", "knn_regresion")


def _knn_predecir(modelo, x):
    vecinos = sorted(range(modelo["n"]),
                     key=lambda i: (_distancia2(modelo["X"][i], x), i))
    cercanos = [modelo["y"][i] for i in vecinos[:modelo["k"]]]
    if modelo["tarea"] == "regresion":
        return sum(cercanos) / len(cercanos)
    # Votacion. Si hay empate gana la clase del vecino mas cercano, porque
    # los vecinos ya estan ordenados por distancia.
    votos = {}
    for c in cercanos:
        votos[c] = votos.get(c, 0) + 1
    mayor = max(votos.values())
    for c in cercanos:
        if votos[c] == mayor:
            return c


# =====================================================================
# K-MEDIAS (K-MEANS)
# =====================================================================

def _centroide_mas_cercano(x, centroides) -> int:
    mejor, menor = 0, None
    for j, c in enumerate(centroides):
        d = _distancia2(x, c)
        if menor is None or d < menor:
            mejor, menor = j, d
    return mejor


def kmeans(X, k, semilla=42, max_iteraciones=100) -> dict:
    """
    Agrupa los datos en k grupos sin usar etiquetas.

    1. Escoge k centros iniciales con k-means++: el primero al azar y
       cada siguiente con mas probabilidad mientras mas lejos este de los
       centros ya escogidos (asi no empiezan todos juntos).
    2. Asigna cada punto al centro mas cercano.
    3. Mueve cada centro al promedio de sus puntos.
    4. Repite 2 y 3 hasta que ningun punto cambie de grupo.

    La inercia es la suma de distancias al cuadrado de cada punto a su
    centro: mientras mas pequena, mas compactos los grupos.
    """
    filas = _matriz(X, "kmeans")
    k = _entero(k, "kmeans", "k")
    max_iteraciones = _entero(max_iteraciones, "kmeans", "max_iteraciones")
    if k > len(filas):
        raise ErrorDatos(f"kmeans(): k = {k} es mayor que la cantidad de "
                         f"datos ({len(filas)})")
    generador = _semilla(semilla, "kmeans")

    centroides = [list(filas[int(generador.siguiente() * len(filas))])]
    while len(centroides) < k:
        distancias = [min(_distancia2(x, c) for c in centroides)
                      for x in filas]
        total = sum(distancias)
        if total == 0:
            centroides.append(list(filas[len(centroides) % len(filas)]))
            continue
        objetivo = generador.siguiente() * total
        acumulado = 0.0
        for x, d in zip(filas, distancias):
            acumulado += d
            if acumulado >= objetivo:
                centroides.append(list(x))
                break

    etiquetas = [None] * len(filas)
    iteraciones = 0
    for iteraciones in range(1, max_iteraciones + 1):
        nuevas = [_centroide_mas_cercano(x, centroides) for x in filas]
        if nuevas == etiquetas:
            break
        etiquetas = nuevas
        for j in range(k):
            miembros = [x for x, e in zip(filas, etiquetas) if e == j]
            if miembros:          # un grupo vacio conserva su centro
                centroides[j] = [sum(col) / len(miembros)
                                 for col in zip(*miembros)]

    inercia = sum(_distancia2(x, centroides[e])
                  for x, e in zip(filas, etiquetas))
    return {"tipo": "kmeans", "k": k, "centroides": centroides,
            "etiquetas": etiquetas, "inercia": inercia,
            "iteraciones": iteraciones, "n": len(filas), "p": len(filas[0])}


def inercias_kmeans(X, k_maximo, semilla=42) -> list:
    """Inercia para k = 1, 2, ..., k_maximo: sirve para el metodo del codo
    (se busca el k donde la curva deja de bajar rapido)."""
    k_maximo = _entero(k_maximo, "inercias_kmeans", "k_maximo")
    return [kmeans(X, k, semilla)["inercia"] for k in range(1, k_maximo + 1)]


def centroides(modelo) -> list:
    _exigir_tipo(modelo, "kmeans", "centroides")
    return [list(c) for c in modelo["centroides"]]


def grupos_kmeans(modelo) -> list:
    """El grupo (0, 1, ..., k-1) de cada dato de entrenamiento."""
    _exigir_tipo(modelo, "kmeans", "grupos_kmeans")
    return list(modelo["etiquetas"])


# =====================================================================
# PERCEPTRON MULTICAPA (RED NEURONAL)
# =====================================================================

def _tanh(z: float) -> float:
    """tanh(z) = (e^(2z) - 1) / (e^(2z) + 1), con el exp propio."""
    if z > 20:
        return 1.0
    if z < -20:
        return -1.0
    e = m.exp(2 * z)
    return (e - 1) / (e + 1)


def _softmax(zs: list) -> list:
    """Convierte puntajes en probabilidades que suman 1. Se resta el
    maximo antes de exp para no desbordar."""
    mayor = max(zs)
    exps = [m.exp(z - mayor) for z in zs]
    total = sum(exps)
    return [e / total for e in exps]


def _capas(ocultas, funcion) -> list:
    if _es_numero(ocultas) and not isinstance(ocultas, float):
        ocultas = [ocultas]
    if not isinstance(ocultas, list) or not ocultas:
        raise ErrorDatos(f"{funcion}(): las capas ocultas van en una lista, "
                         f"por ejemplo [8] o [8, 4]")
    return [_entero(n, funcion, "cada capa oculta") for n in ocultas]


def _inicializar(tamanos, generador):
    """
    Pesos al azar en [-l, l] con l = raiz(6 / (entradas + salidas))
    (inicializacion de Xavier/Glorot, buena para tanh). Los sesgos
    empiezan en 0.
    """
    pesos, sesgos = [], []
    for entradas, salidas in zip(tamanos, tamanos[1:]):
        limite = (6 / (entradas + salidas)) ** 0.5
        pesos.append([[(generador.siguiente() * 2 - 1) * limite
                       for _ in range(entradas)] for _ in range(salidas)])
        sesgos.append([0.0] * salidas)
    return pesos, sesgos


def _adelante(pesos, sesgos, x, clasificacion):
    """
    Propagacion hacia adelante. Devuelve la activacion de cada capa (la
    primera es la entrada). Capas ocultas: tanh. Salida: softmax si es
    clasificacion, lineal (sin cambiar) si es regresion.
    """
    activaciones = [x]
    ultima = len(pesos) - 1
    for capa, (W, b) in enumerate(zip(pesos, sesgos)):
        entrada = activaciones[-1]
        zs = [sum(w * a for w, a in zip(fila, entrada)) + sesgo
              for fila, sesgo in zip(W, b)]
        if capa < ultima:
            activaciones.append([_tanh(z) for z in zs])
        elif clasificacion:
            activaciones.append(_softmax(zs))
        else:
            activaciones.append(zs)
    return activaciones


def _atras(pesos, sesgos, activaciones, objetivo, tasa):
    """
    Retropropagacion y un paso de descenso de gradiente.

    En la salida, tanto con softmax + entropia cruzada como con salida
    lineal + error cuadratico, el error de cada neurona es simplemente
    (salida - objetivo). Hacia atras, el error de una neurona oculta es
    la suma de los errores de la capa siguiente pesados por sus pesos,
    multiplicada por la derivada de tanh: 1 - a^2.
    Cada peso baja en tasa * error * entrada.
    """
    delta = [s - o for s, o in zip(activaciones[-1], objetivo)]
    for capa in range(len(pesos) - 1, -1, -1):
        entrada = activaciones[capa]
        W = pesos[capa]
        if capa > 0:
            delta_anterior = [
                sum(W[j][i] * delta[j] for j in range(len(delta)))
                * (1 - entrada[i] ** 2)
                for i in range(len(entrada))]
        for j, d in enumerate(delta):
            fila = W[j]
            for i, a in enumerate(entrada):
                fila[i] -= tasa * d * a
            sesgos[capa][j] -= tasa * d
        if capa > 0:
            delta = delta_anterior


def _perceptron(X, y, ocultas, epocas, tasa, semilla, clasificacion,
                funcion):
    filas = _matriz(X, funcion)
    etiquetas = _vector(y, funcion, numerico=not clasificacion)
    _mismo_largo(filas, etiquetas, funcion)
    capas = _capas(ocultas, funcion)
    epocas = _entero(epocas, funcion, "epocas")
    if not _es_numero(tasa) or tasa <= 0:
        raise ErrorDatos(f"{funcion}(): la tasa de aprendizaje debe ser un "
                         f"numero mayor que 0, por ejemplo 0.05")
    generador = _semilla(semilla, funcion)

    # La red aprende mejor con entradas en la misma escala: se guardan la
    # media y la desviacion de cada columna y se aplican siempre.
    escala_x = [(_media(col), _desviacion(col)) for col in zip(*filas)]
    entradas = [[_aplicar_escala(v, e) for v, e in zip(fila, escala_x)]
                for fila in filas]

    if clasificacion:
        clases = _clases(etiquetas, [])
        objetivos = [[1.0 if e == c else 0.0 for c in clases]
                     for e in etiquetas]
        salidas = len(clases)
        escala_y = None
    else:
        clases = None
        escala_y = (_media(etiquetas), _desviacion(etiquetas))
        objetivos = [[_aplicar_escala(v, escala_y)] for v in etiquetas]
        salidas = 1

    tamanos = [len(filas[0])] + capas + [salidas]
    pesos, sesgos = _inicializar(tamanos, generador)

    perdidas = []
    for _ in range(epocas):
        total = 0.0
        for i in _barajar(len(entradas), generador):
            activaciones = _adelante(pesos, sesgos, entradas[i],
                                     clasificacion)
            salida = activaciones[-1]
            if clasificacion:
                total -= m.ln(max(sum(s * o for s, o in
                                      zip(salida, objetivos[i])), 1e-12))
            else:
                total += 0.5 * (salida[0] - objetivos[i][0]) ** 2
            _atras(pesos, sesgos, activaciones, objetivos[i], tasa)
        perdidas.append(total / len(entradas))

    return {"tipo": "perceptron",
            "tarea": "clasificacion" if clasificacion else "regresion",
            "capas": tamanos, "pesos": pesos, "sesgos": sesgos,
            "clases": clases, "escala_x": escala_x, "escala_y": escala_y,
            "epocas": epocas, "tasa": tasa, "perdidas": perdidas,
            "n": len(filas), "p": len(filas[0])}


def perceptron_clasificador(X, y, ocultas=None, epocas=300, tasa=0.05,
                            semilla=42) -> dict:
    """Red neuronal para clasificar. ocultas: neuronas por capa oculta."""
    return _perceptron(X, y, [8] if ocultas is None else ocultas, epocas,
                       tasa, semilla, True, "perceptron_clasificador")


def perceptron_regresor(X, y, ocultas=None, epocas=300, tasa=0.01,
                        semilla=42) -> dict:
    """Red neuronal para predecir un numero."""
    return _perceptron(X, y, [8] if ocultas is None else ocultas, epocas,
                       tasa, semilla, False, "perceptron_regresor")


def _perceptron_predecir(modelo, x):
    entrada = [_aplicar_escala(v, e) for v, e in zip(x, modelo["escala_x"])]
    salida = _adelante(modelo["pesos"], modelo["sesgos"], entrada,
                       modelo["tarea"] == "clasificacion")[-1]
    if modelo["tarea"] == "clasificacion":
        mayor = max(range(len(salida)), key=lambda j: salida[j])
        return modelo["clases"][mayor]
    media, desviacion = modelo["escala_y"]
    return salida[0] * desviacion + media


def perdidas(modelo) -> list:
    """La perdida promedio de cada epoca: debe ir bajando."""
    _exigir_tipo(modelo, "perceptron", "perdidas")
    return list(modelo["perdidas"])


# =====================================================================
# PREDECIR CON CUALQUIER MODELO
# =====================================================================

def _exigir_tipo(modelo, tipo, funcion):
    if not isinstance(modelo, dict) or modelo.get("tipo") != tipo:
        raise ErrorDatos(f"{funcion}() necesita un modelo de tipo {tipo}")


def _variables(modelo) -> int:
    if modelo["tipo"] == "simple":
        return 1
    if modelo["tipo"] == "multiple":
        return len(modelo["coeficientes"]) - 1 \
            if "coeficientes" in modelo else modelo.get("p", 1)
    return modelo.get("p", 1)


def aplicar_modelo(modelo, X):
    """
    Predicciones de un modelo para los datos X.

    - Si X tiene varias filas, devuelve una lista con una prediccion por
      fila.
    - Si X es una sola observacion ([x1, x2, ...] con tantos valores como
      variables tiene el modelo), devuelve una sola prediccion.
    """
    if not isinstance(modelo, dict) or "tipo" not in modelo:
        raise ErrorDatos("aplicar_modelo() necesita un modelo entrenado")
    variables = _variables(modelo)

    una_sola = (isinstance(X, list) and X
                and all(_es_numero(v) for v in X)
                and variables > 1 and len(X) == variables)
    filas = _matriz([X] if una_sola else X, "aplicar_modelo")
    if len(filas[0]) != variables:
        raise ErrorDatos(f"aplicar_modelo(): el modelo usa {variables} "
                         f"variable(s) y los datos tienen {len(filas[0])}")

    tipo = modelo["tipo"]
    if tipo == "simple":
        resultado = [reg.predecir_simple(modelo, f[0]) for f in filas]
    elif tipo == "multiple":
        resultado = [reg.predecir_multiple(modelo, f) for f in filas]
    elif tipo == "arbol_decision":
        resultado = ad.predecir_lote(modelo, filas)
    elif tipo == "knn":
        resultado = [_knn_predecir(modelo, f) for f in filas]
    elif tipo == "kmeans":
        resultado = [_centroide_mas_cercano(f, modelo["centroides"])
                     for f in filas]
    elif tipo == "perceptron":
        resultado = [_perceptron_predecir(modelo, f) for f in filas]
    else:
        raise ErrorDatos(f"aplicar_modelo(): modelo desconocido '{tipo}'")
    return resultado[0] if una_sola else resultado


# =====================================================================
# MOSTRAR UN MODELO
# =====================================================================

def describir(modelo) -> str:
    """Texto corto para mostrar() de los modelos de este modulo."""
    tipo = modelo.get("tipo")
    if tipo == "knn":
        return (f"KNN ({modelo['tarea']}) con k={modelo['k']}  "
                f"(n={modelo['n']}, p={modelo['p']})")
    if tipo == "kmeans":
        tamanos = [modelo["etiquetas"].count(j) for j in range(modelo["k"])]
        return (f"K-medias con k={modelo['k']}  (n={modelo['n']}, "
                f"p={modelo['p']})\n"
                f"  Inercia: {round(modelo['inercia'], 4)}  |  "
                f"Iteraciones: {modelo['iteraciones']}  |  "
                f"Tamano de cada grupo: {tamanos}")
    if tipo == "perceptron":
        capas = " -> ".join(str(c) for c in modelo["capas"])
        texto = (f"Perceptron multicapa ({modelo['tarea']})  capas: {capas}\n"
                 f"  Epocas: {modelo['epocas']}  |  Tasa: {modelo['tasa']}"
                 f"  |  Perdida inicial: {round(modelo['perdidas'][0], 4)}"
                 f"  |  Perdida final: {round(modelo['perdidas'][-1], 4)}")
        if modelo["clases"]:
            texto += f"\n  Clases: {modelo['clases']}"
        return texto
    return str(modelo)


# =====================================================================
# REGISTRO EN EL LENGUAJE
# =====================================================================

def funciones() -> dict:
    return _registrar([
        ("dividir_datos",           dividir_datos,           2, 4),
        ("normalizar",              normalizar,              1, 2),
        ("estandarizar",            estandarizar,            1, 2),
        ("mse",                     mse,                     2),
        ("rmse",                    rmse,                    2),
        ("mae",                     mae,                     2),
        ("r2",                      r2,                      2),
        ("exactitud",               exactitud,               1, 2),
        ("precision",               precision,               2, 3),
        ("recall",                  recall,                  2, 3),
        ("f1",                      f1,                      2, 3),
        ("matriz_confusion",        matriz_confusion,        2),
        ("reporte_clasificacion",   reporte_clasificacion,   2),
        ("knn",                     knn,                     2, 3),
        ("knn_regresion",           knn_regresion,           2, 3),
        ("kmeans",                  kmeans,                  2, 4),
        ("inercias_kmeans",         inercias_kmeans,         2, 3),
        ("centroides",              centroides,              1),
        ("grupos_kmeans",           grupos_kmeans,           1),
        ("perceptron_clasificador", perceptron_clasificador, 2, 6),
        ("perceptron_regresor",     perceptron_regresor,     2, 6),
        ("perdidas",                perdidas,                1),
        ("aplicar_modelo",          aplicar_modelo,          2),
    ])
