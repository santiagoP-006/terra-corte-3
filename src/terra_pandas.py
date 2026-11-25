"""
terra_pandas.py
===============
Implementacion de un DataFrame para Terra, inspirado en pandas.
Escrito desde cero, sin dependencias externas.

Un DataFrame es una extension de Tabla con operaciones adicionales:
    - acceso y modificacion de columnas como vectores
    - merge (join) entre dos DataFrames
    - pivot (tabla dinamica)
    - apply (aplicar una funcion a cada elemento de una columna)
    - melt (formato ancho a largo)
    - head / tail
    - sample (muestra aleatoria)
    - info (resumen estructural)
    - duplicados (detectar filas duplicadas)
    - replace (reemplazar valores)
    - corr (correlacion entre columnas numericas)
"""

from src.matematicas import GENERADOR, GeneradorAleatorio
from src.tabla import Tabla, ErrorDatos, formatear_valor


# =====================================================================
# CLASE DATAFRAME
# =====================================================================

class DataFrame(Tabla):
    """
    Extension de Tabla con operaciones inspiradas en pandas.
    Internamente es identica a Tabla; el nombre distingue su uso
    semantico en el lenguaje.
    """

    def __init__(self, columnas, filas=None):
        super().__init__(columnas, filas)

    @classmethod
    def desde_tabla(cls, tabla: Tabla) -> "DataFrame":
        return cls(tabla.columnas, tabla.filas)

    @classmethod
    def desde_diccionario(cls, datos: dict) -> "DataFrame":
        """
        Crea un DataFrame desde un diccionario {nombre_col: [valores]}.
        Todas las listas deben tener la misma longitud.
        """
        if not datos:
            return cls([], [])
        columnas = list(datos.keys())
        longitudes = [len(v) for v in datos.values()]
        if len(set(longitudes)) > 1:
            raise ErrorDatos(
                "dataframe(): todas las columnas deben tener la misma "
                f"longitud; se encontraron longitudes: {longitudes}")
        n = longitudes[0]
        filas = [[datos[c][i] for c in columnas] for i in range(n)]
        return cls(columnas, filas)

    def como_texto(self, max_filas: int = 10) -> str:
        encabezado = super().como_texto(max_filas)
        return f"DataFrame\n{encabezado}"

    def __repr__(self):
        return (f"DataFrame({self.num_filas} filas x "
                f"{self.num_columnas} columnas)")


# =====================================================================
# CREACION
# =====================================================================

def crear_dataframe(datos: dict) -> DataFrame:
    """dataframe({col: [valores], ...}) -> DataFrame"""
    if not isinstance(datos, dict):
        raise ErrorDatos(
            "dataframe() espera un diccionario {columna: lista_de_valores}")
    for k, v in datos.items():
        if not isinstance(k, str):
            raise ErrorDatos("las claves del diccionario deben ser texto")
        if not isinstance(v, list):
            raise ErrorDatos(
                f"el valor de la columna '{k}' debe ser una lista")
    return DataFrame.desde_diccionario(datos)


def desde_tabla(tabla: Tabla) -> DataFrame:
    """Convierte una Tabla a DataFrame."""
    if not isinstance(tabla, Tabla):
        raise ErrorDatos("desde_tabla() espera una Tabla")
    return DataFrame.desde_tabla(tabla)


# =====================================================================
# INSPECCION
# =====================================================================

def head(df: DataFrame, n: int = 5) -> DataFrame:
    """Primeras n filas."""
    _validar_df(df, "head")
    return DataFrame(df.columnas, df.filas[:n])


def tail(df: DataFrame, n: int = 5) -> DataFrame:
    """Ultimas n filas."""
    _validar_df(df, "tail")
    return DataFrame(df.columnas, df.filas[-n:])


def info(df: DataFrame) -> str:
    """Resumen estructural: tipos, nulos y memoria estimada."""
    _validar_df(df, "info")
    lineas = [
        f"DataFrame: {df.num_filas} filas x {df.num_columnas} columnas",
        f"{'Columna':<25} {'Tipo':<12} {'No-nulos':>10} {'Nulos':>8}",
        "-" * 57,
    ]
    for col in df.columnas:
        valores    = df.columna(col)
        no_nulos   = sum(1 for v in valores if v is not None)
        nulos      = df.num_filas - no_nulos
        tipo       = df.tipo_columna(col)
        lineas.append(f"{col:<25} {tipo:<12} {no_nulos:>10} {nulos:>8}")
    return "\n".join(lineas)


def sample(df: DataFrame, n: int, semilla: int = None) -> DataFrame:
    """
    Muestra aleatoria de n filas sin reemplazo.

    Usa el generador propio de matematicas.py (no el modulo random de
    Python). El metodo es el barajado de Fisher-Yates parcial: se
    intercambia cada posicion i con una posicion al azar entre i y el
    final, y las primeras n posiciones quedan como la muestra.
    Con la misma semilla se obtiene siempre la misma muestra.
    """
    _validar_df(df, "sample")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ErrorDatos("sample(): n debe ser un entero mayor o igual a 0")
    if n > df.num_filas:
        raise ErrorDatos(
            f"sample(): se pidieron {n} filas pero el DataFrame "
            f"solo tiene {df.num_filas}")
    if semilla is not None and (isinstance(semilla, bool)
                                or not isinstance(semilla, int)):
        raise ErrorDatos("sample(): la semilla debe ser un numero entero")

    generador = GENERADOR if semilla is None else GeneradorAleatorio(semilla)
    posiciones = list(range(df.num_filas))
    for i in range(n):
        j = i + int(generador.siguiente() * (len(posiciones) - i))
        posiciones[i], posiciones[j] = posiciones[j], posiciones[i]

    filas = [df.filas[i] for i in sorted(posiciones[:n])]
    return DataFrame(df.columnas, filas)

# =====================================================================
# TRANSFORMACION
# =====================================================================

def apply_col(df: DataFrame, columna: str, funcion) -> DataFrame:
    """
    Aplica una funcion de Terra a cada elemento de una columna.
    La funcion recibe un valor y devuelve un valor.

    Se usa desde el visitante pasando un callable Python que
    envuelve la llamada al visitante Terra.
    """
    _validar_df(df, "aplicar")
    j      = df.indice(columna)
    nuevas = [funcion(fila[j]) for fila in df.filas]
    filas  = [fila[:j] + [v] + fila[j+1:]
              for fila, v in zip(df.filas, nuevas)]
    return DataFrame(df.columnas, filas)


def melt(df: DataFrame, columnas_id: list,
         nombre_variable: str = "variable",
         nombre_valor: str = "valor") -> DataFrame:
    """
    Convierte formato ancho a largo (unpivot).

    columnas_id: columnas que se mantienen fijas.
    El resto de columnas se apilan en dos columnas:
    nombre_variable (nombre de la columna original) y
    nombre_valor (valor de la celda).
    """
    _validar_df(df, "melt")
    for c in columnas_id:
        df.indice(c)   # valida existencia

    cols_valor = [c for c in df.columnas if c not in columnas_id]
    if not cols_valor:
        raise ErrorDatos("melt(): no quedan columnas para apilar")

    idx_id = [df.columnas.index(c) for c in columnas_id]
    filas  = []
    for fila in df.filas:
        base = [fila[j] for j in idx_id]
        for col in cols_valor:
            j = df.columnas.index(col)
            filas.append(base + [col, fila[j]])

    return DataFrame(columnas_id + [nombre_variable, nombre_valor], filas)


def pivot(df: DataFrame, indice: str,
          columna: str, valor: str) -> DataFrame:
    """
    Tabla dinamica: convierte filas en columnas.

    indice  : columna cuyos valores se convierten en filas del resultado
    columna : columna cuyos valores se convierten en cabeceras
    valor   : columna cuyos valores se colocan en las celdas
    """
    _validar_df(df, "pivot")
    df.indice(indice); df.indice(columna); df.indice(valor)

    j_idx = df.columnas.index(indice)
    j_col = df.columnas.index(columna)
    j_val = df.columnas.index(valor)

    # Recolectar filas e índices únicos (en orden de aparición)
    indices_unicos  = []
    columnas_unicas = []
    tabla_dict      = {}   # (idx_val, col_val) -> valor

    for fila in df.filas:
        i_val = fila[j_idx]
        c_val = fila[j_col]
        v_val = fila[j_val]

        if i_val not in indices_unicos:
            indices_unicos.append(i_val)
        if c_val not in columnas_unicas:
            columnas_unicas.append(c_val)
        tabla_dict[(i_val, c_val)] = v_val

    col_nombres = [indice] + [str(c) for c in columnas_unicas]
    filas = []
    for i_val in indices_unicos:
        fila = [i_val] + [tabla_dict.get((i_val, c), None)
                          for c in columnas_unicas]
        filas.append(fila)

    return DataFrame(col_nombres, filas)


def replace_val(df: DataFrame, columna: str,
                viejo, nuevo) -> DataFrame:
    """Reemplaza todas las ocurrencias de viejo por nuevo en una columna."""
    _validar_df(df, "reemplazar")
    j     = df.indice(columna)
    filas = [fila[:j] + [nuevo if fila[j] == viejo else fila[j]] + fila[j+1:]
             for fila in df.filas]
    return DataFrame(df.columnas, filas)


def duplicados(df: DataFrame) -> DataFrame:
    """Devuelve solo las filas duplicadas (la primera copia incluida)."""
    _validar_df(df, "duplicados_df")
    vistos = set()
    dup    = []
    for fila in df.filas:
        clave = tuple(fila)
        if clave in vistos:
            dup.append(fila)
        else:
            vistos.add(clave)
    return DataFrame(df.columnas, dup)


# =====================================================================
# MERGE (JOIN)
# =====================================================================

def merge(izq: DataFrame, der: DataFrame,
          en: str, tipo: str = "inner") -> DataFrame:
    """
    Une dos DataFrames por una columna comun.

    tipo: "inner" | "left" | "right" | "outer"
    """
    _validar_df(izq, "merge (izquierdo)")
    _validar_df(der, "merge (derecho)")
    izq.indice(en); der.indice(en)

    j_izq = izq.columnas.index(en)
    j_der = der.columnas.index(en)

    # Columnas del resultado: todas de izq + las de der que no sean 'en'
    cols_der_extra = [c for c in der.columnas if c != en]
    # Renombrar conflictos
    cols_res = list(izq.columnas)
    for c in cols_der_extra:
        nombre = c + "_der" if c in izq.columnas else c
        cols_res.append(nombre)

    # Indexar der por el valor de la columna clave
    indice_der = {}
    for fila in der.filas:
        clave = fila[j_der]
        indice_der.setdefault(clave, []).append(fila)

    filas     = []
    claves_izq = set()

    for fila_i in izq.filas:
        clave = fila_i[j_izq]
        claves_izq.add(clave)
        coincidencias = indice_der.get(clave, [])

        if coincidencias:
            for fila_d in coincidencias:
                extra = [fila_d[der.columnas.index(c)] for c in cols_der_extra]
                filas.append(list(fila_i) + extra)
        elif tipo in ("left", "outer"):
            filas.append(list(fila_i) + [None] * len(cols_der_extra))

    if tipo in ("right", "outer"):
        for fila_d in der.filas:
            clave = fila_d[j_der]
            if clave not in claves_izq:
                nulos_izq = [None] * (len(izq.columnas) - 1)
                extra     = [fila_d[der.columnas.index(c)] for c in cols_der_extra]
                # Reconstruye la fila izquierda con nulos excepto la clave
                fila_res  = []
                for col in izq.columnas:
                    fila_res.append(clave if col == en else None)
                fila_res += extra
                filas.append(fila_res)

    return DataFrame(cols_res, filas)


# =====================================================================
# CORRELACION
# =====================================================================

def corr(df: DataFrame) -> DataFrame:
    """
    Matriz de correlacion de Pearson entre todas las columnas numericas.
    Los nulos se ignoran en cada par de columnas.
    """
    _validar_df(df, "correlacion")
    cols_num = [c for c in df.columnas if df.es_numerica(c)]
    if len(cols_num) < 2:
        raise ErrorDatos(
            "correlacion() necesita al menos 2 columnas numericas")

    def pearson(xs, ys):
        pares  = [(x, y) for x, y in zip(xs, ys)
                  if x is not None and y is not None]
        if len(pares) < 2:
            return None
        n      = len(pares)
        mx     = sum(p[0] for p in pares) / n
        my     = sum(p[1] for p in pares) / n
        num    = sum((p[0]-mx)*(p[1]-my) for p in pares)
        den_x  = sum((p[0]-mx)**2 for p in pares) ** 0.5
        den_y  = sum((p[1]-my)**2 for p in pares) ** 0.5
        if den_x == 0 or den_y == 0:
            return None
        return num / (den_x * den_y)

    datos = {c: df.columna(c) for c in cols_num}
    filas = []
    for c1 in cols_num:
        fila = [c1]
        for c2 in cols_num:
            fila.append(pearson(datos[c1], datos[c2]))
        filas.append(fila)

    return DataFrame(["columna"] + cols_num, filas)


# =====================================================================
# VALIDACION
# =====================================================================

def _validar_df(obj, funcion: str):
    if not isinstance(obj, (Tabla, DataFrame)):
        raise ErrorDatos(
            f"{funcion}() espera un DataFrame o Tabla, "
            f"se recibio {type(obj).__name__}")
