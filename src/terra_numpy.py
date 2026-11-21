"""
terra_numpy.py
==============
Implementacion de arrays n-dimensionales para Terra, inspirada en numpy.
Escrita desde cero, sin dependencias externas.

Un Array es una lista de listas (o lista plana para 1D) con:
    - operaciones aritmeticas elemento a elemento
    - operaciones de algebra lineal (producto matricial, transpuesta,
      determinante, traza, norma)
    - operaciones de reduccion (suma, media, min, max por eje)
    - reshape, flatten, zeros, ones, rango (arange), linspace
    - broadcasting escalar
"""

from src.tabla import ErrorDatos


# =====================================================================
# CLASE ARRAY
# =====================================================================

class Array:
    """
    Array n-dimensional de Terra.

    Atributos:
        datos  : lista plana de valores en orden fila-mayor
        forma  : tupla (filas, cols) para 2D o (n,) para 1D
    """

    def __init__(self, datos: list, forma: tuple):
        self.datos = datos
        self.forma = forma

    # ---- propiedades ----

    @property
    def ndim(self) -> int:
        return len(self.forma)

    @property
    def size(self) -> int:
        resultado = 1
        for d in self.forma:
            resultado *= d
        return resultado

    @property
    def es_1d(self) -> bool:
        return self.ndim == 1

    @property
    def es_2d(self) -> bool:
        return self.ndim == 2

    # ---- acceso ----

    def _idx(self, i: int, j: int = None) -> int:
        if self.es_1d:
            return i
        return i * self.forma[1] + j

    def obtener(self, i: int, j: int = None):
        return self.datos[self._idx(i, j)]

    def fila(self, i: int) -> list:
        if self.es_1d:
            return [self.datos[i]]
        c = self.forma[1]
        return self.datos[i*c:(i+1)*c]

    def como_listas(self) -> list:
        """Devuelve el array como lista de listas (2D) o lista (1D)."""
        if self.es_1d:
            return list(self.datos)
        c = self.forma[1]
        return [self.datos[i*c:(i+1)*c] for i in range(self.forma[0])]

    # ---- representacion ----

    def __repr__(self):
        return f"Array{self.forma}"

    def como_texto(self) -> str:
        if self.es_1d:
            vals = " ".join(f"{v:>10.4f}" if isinstance(v, float)
                            else f"{v:>10}" for v in self.datos)
            return f"Array({self.forma[0]},)\n[{vals}]"
        lineas = [f"Array{self.forma}"]
        c = self.forma[1]
        for i in range(self.forma[0]):
            fila = self.datos[i*c:(i+1)*c]
            vals = " ".join(f"{v:>10.4f}" if isinstance(v, float)
                            else f"{v:>10}" for v in fila)
            lineas.append(f"  [{vals}]")
        return "\n".join(lineas)


# =====================================================================
# CREACION
# =====================================================================

def _aplanar(lista: list) -> tuple:
    """Aplana una lista de listas y determina la forma."""
    if not lista:
        return [], (0,)
    if isinstance(lista[0], list):
        filas = len(lista)
        cols  = len(lista[0])
        for i, fila in enumerate(lista):
            if len(fila) != cols:
                raise ErrorDatos(
                    f"array(): la fila {i} tiene {len(fila)} elementos "
                    f"pero se esperaban {cols}")
            for v in fila:
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise ErrorDatos(
                        "array(): todos los elementos deben ser numeros")
        datos = [float(v) for fila in lista for v in fila]
        return datos, (filas, cols)
    else:
        for v in lista:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ErrorDatos(
                    "array(): todos los elementos deben ser numeros")
        return [float(v) for v in lista], (len(lista),)


def array(lista: list) -> Array:
    """Crea un Array desde una lista (1D) o lista de listas (2D)."""
    if not isinstance(lista, list):
        raise ErrorDatos("array() espera una lista o lista de listas")
    datos, forma = _aplanar(lista)
    return Array(datos, forma)


def zeros(filas: int, cols: int = None) -> Array:
    """Array de ceros."""
    if cols is None:
        return Array([0.0] * filas, (filas,))
    return Array([0.0] * (filas * cols), (filas, cols))


def ones(filas: int, cols: int = None) -> Array:
    """Array de unos."""
    if cols is None:
        return Array([1.0] * filas, (filas,))
    return Array([1.0] * (filas * cols), (filas, cols))


def identidad(n: int) -> Array:
    """Matriz identidad n x n."""
    datos = [1.0 if i == j else 0.0
             for i in range(n) for j in range(n)]
    return Array(datos, (n, n))


def rango(inicio: float, fin: float, paso: float = 1.0) -> Array:
    """Equivalente a numpy.arange."""
    if paso == 0:
        raise ErrorDatos("rango(): el paso no puede ser cero")
    datos = []
    v = inicio
    while (paso > 0 and v < fin) or (paso < 0 and v > fin):
        datos.append(float(v))
        v += paso
    return Array(datos, (len(datos),))


def linspace(inicio: float, fin: float, n: int) -> Array:
    """n valores equiespaciados entre inicio y fin (inclusive)."""
    if n < 2:
        raise ErrorDatos("linspace(): n debe ser >= 2")
    paso  = (fin - inicio) / (n - 1)
    datos = [float(inicio + i * paso) for i in range(n)]
    return Array(datos, (n,))


# =====================================================================
# OPERACIONES ELEMENTO A ELEMENTO
# =====================================================================

def _op_escalar(a: Array, escalar, op: str) -> Array:
    ops = {
        "+": lambda x, s: x + s,
        "-": lambda x, s: x - s,
        "*": lambda x, s: x * s,
        "/": lambda x, s: x / s if s != 0 else (_ for _ in ()).throw(
            ErrorDatos("array: division por cero")),
        "^": lambda x, s: x ** s,
    }
    f = ops[op]
    return Array([f(v, escalar) for v in a.datos], a.forma)


def _op_elem(a: Array, b: Array, op: str) -> Array:
    if a.forma != b.forma:
        raise ErrorDatos(
            f"operacion elemento a elemento: las formas {a.forma} y "
            f"{b.forma} no coinciden")
    ops = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
    }
    f = ops[op]
    return Array([f(x, y) for x, y in zip(a.datos, b.datos)], a.forma)


def sumar_arrays(a: Array, b) -> Array:
    if isinstance(b, (int, float)):
        return _op_escalar(a, float(b), "+")
    return _op_elem(a, b, "+")


def restar_arrays(a: Array, b) -> Array:
    if isinstance(b, (int, float)):
        return _op_escalar(a, float(b), "-")
    return _op_elem(a, b, "-")


def multiplicar_arrays(a: Array, b) -> Array:
    """Multiplicacion elemento a elemento (no matricial)."""
    if isinstance(b, (int, float)):
        return _op_escalar(a, float(b), "*")
    return _op_elem(a, b, "*")


def dividir_arrays(a: Array, b) -> Array:
    if isinstance(b, (int, float)):
        if b == 0:
            raise ErrorDatos("division de array por cero")
        return _op_escalar(a, float(b), "/")
    return _op_elem(a, b, "/")


def potencia_array(a: Array, exp: float) -> Array:
    return _op_escalar(a, exp, "^")


def raiz_array(a: Array) -> Array:
    for v in a.datos:
        if v < 0:
            raise ErrorDatos(
                "raiz_array(): no se puede calcular la raiz de un numero negativo")
    return Array([v ** 0.5 for v in a.datos], a.forma)


def abs_array(a: Array) -> Array:
    return Array([abs(v) for v in a.datos], a.forma)


# =====================================================================
# ALGEBRA LINEAL
# =====================================================================

def producto_matricial(a: Array, b: Array) -> Array:
    """Multiplicación matricial A @ B entre dos Arrays 2D."""
    if not isinstance(a, Array) or not isinstance(b, Array):
        raise ErrorDatos("producto_matricial(): se necesitan dos arrays")
    if not a.es_2d or not b.es_2d:
        raise ErrorDatos(
            "producto_matricial(): ambos arrays deben ser 2D; "
            "crea matrices con array([[fila1], [fila2], ...])")
    if a.forma[1] != b.forma[0]:
        raise ErrorDatos(
            f"producto_matricial(): las formas {a.forma} y {b.forma} "
            f"son incompatibles (columnas de A != filas de B)")
    m, k, n = a.forma[0], a.forma[1], b.forma[1]
    datos   = []
    for i in range(m):
        for j in range(n):
            s = sum(a.datos[i * k + p] * b.datos[p * n + j]
                    for p in range(k))
            datos.append(s)
    return Array(datos, (m, n))


def transpuesta(a: Array) -> Array:
    """Transpone un Array 2D."""
    if not isinstance(a, Array) or not a.es_2d:
        raise ErrorDatos(
            "transpuesta(): se necesita un array 2D; "
            "crea uno con array([[fila1], [fila2], ...])")
    m, n  = a.forma
    datos = [a.datos[i * n + j] for j in range(n) for i in range(m)]
    return Array(datos, (n, m))


def traza(a: Array) -> float:
    """Suma de la diagonal principal de un Array 2D cuadrado."""
    if not isinstance(a, Array) or not a.es_2d:
        raise ErrorDatos(
            "traza(): se necesita un array 2D cuadrado")
    if a.forma[0] != a.forma[1]:
        raise ErrorDatos(
            f"traza(): la matriz debe ser cuadrada "
            f"(se recibio {a.forma[0]}x{a.forma[1]})")
    n = a.forma[0]
    return sum(a.datos[i * n + i] for i in range(n))



def determinante(a: Array) -> float:
    """Determinante de un Array 2D cuadrado por eliminación gaussiana."""
    if not isinstance(a, Array) or not a.es_2d:
        raise ErrorDatos(
            "determinante(): se necesita un array 2D cuadrado")
    if a.forma[0] != a.forma[1]:
        raise ErrorDatos(
            f"determinante(): la matriz debe ser cuadrada "
            f"(se recibio {a.forma[0]}x{a.forma[1]})")
    n = a.forma[0]
    # Copia de la matriz como lista de filas mutables
    M     = [a.datos[i * n:(i + 1) * n][:] for i in range(n)]
    signo = 1.0
    for col in range(n):
        max_fila = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[max_fila][col]) < 1e-12:
            return 0.0
        if max_fila != col:
            M[col], M[max_fila] = M[max_fila], M[col]
            signo *= -1
        piv = M[col][col]
        for fila in range(col + 1, n):
            factor = M[fila][col] / piv
            M[fila] = [M[fila][j] - factor * M[col][j] for j in range(n)]
    return signo * _producto_diagonal(M, n)


def _producto_diagonal(M, n):
    p = 1.0
    for i in range(n):
        p *= M[i][i]
    return p


def norma(a: Array) -> float:
    """Norma euclidiana (L2) de cualquier Array."""
    if not isinstance(a, Array) or len(a.datos) == 0:
        raise ErrorDatos(
            "norma(): se necesita un array no vacío")
    return sum(v ** 2 for v in a.datos) ** 0.5


def dot(a: Array, b: Array) -> float:
    """Producto punto de dos Arrays 1D."""
    if not isinstance(a, Array) or not isinstance(b, Array):
        raise ErrorDatos("dot(): se necesitan dos arrays")
    if not a.es_1d or not b.es_1d:
        raise ErrorDatos(
            "dot(): ambos arrays deben ser 1D; "
            "para matrices usa producto_mat()")
    if a.forma[0] != b.forma[0]:
        raise ErrorDatos(
            f"dot(): longitudes {a.forma[0]} y {b.forma[0]} no coinciden")
    return sum(x * y for x, y in zip(a.datos, b.datos))


# =====================================================================
# REDUCCION
# =====================================================================

def suma_array(a: Array, eje: int = None):
    """Suma de todos los elementos o por eje (0=columnas, 1=filas)."""
    if eje is None:
        return sum(a.datos)
    if not a.es_2d:
        raise ErrorDatos("suma_array(eje): el array debe ser 2D")
    m, n = a.forma
    if eje == 0:   # resultado: vector de longitud n (una suma por columna)
        datos = [sum(a.datos[i*n+j] for i in range(m)) for j in range(n)]
        return Array(datos, (n,))
    if eje == 1:   # resultado: vector de longitud m (una suma por fila)
        datos = [sum(a.datos[i*n+j] for j in range(n)) for i in range(m)]
        return Array(datos, (m,))
    raise ErrorDatos("suma_array(): eje debe ser 0 o 1")


def media_array(a: Array, eje: int = None):
    if eje is None:
        return sum(a.datos) / len(a.datos)
    s = suma_array(a, eje)
    divisor = a.forma[1] if eje == 0 else a.forma[0]
    return Array([v / divisor for v in s.datos], s.forma)


def min_array(a: Array):
    return min(a.datos)


def max_array(a: Array):
    return max(a.datos)


# =====================================================================
# RESHAPE Y UTILIDADES
# =====================================================================

def reshape(a: Array, filas: int, cols: int) -> Array:
    if filas * cols != a.size:
        raise ErrorDatos(
            f"reshape(): {a.size} elementos no caben en forma "
            f"({filas}, {cols})")
    return Array(list(a.datos), (filas, cols))


def flatten(a: Array) -> Array:
    return Array(list(a.datos), (a.size,))


def concatenar(a: Array, b: Array, eje: int = 0) -> Array:
    """Concatena dos arrays por eje."""
    if a.ndim != b.ndim:
        raise ErrorDatos("concatenar(): los arrays deben tener el mismo ndim")
    if a.es_1d:
        return Array(list(a.datos) + list(b.datos),
                     (a.forma[0] + b.forma[0],))
    if eje == 0:
        if a.forma[1] != b.forma[1]:
            raise ErrorDatos(
                f"concatenar(eje=0): el numero de columnas debe coincidir "
                f"({a.forma[1]} vs {b.forma[1]})")
        return Array(list(a.datos) + list(b.datos),
                     (a.forma[0] + b.forma[0], a.forma[1]))
    if eje == 1:
        if a.forma[0] != b.forma[0]:
            raise ErrorDatos(
                f"concatenar(eje=1): el numero de filas debe coincidir "
                f"({a.forma[0]} vs {b.forma[0]})")
        m, na, nb = a.forma[0], a.forma[1], b.forma[1]
        datos = []
        for i in range(m):
            datos += a.datos[i*na:(i+1)*na]
            datos += b.datos[i*nb:(i+1)*nb]
        return Array(datos, (m, na+nb))
    raise ErrorDatos("concatenar(): eje debe ser 0 o 1")