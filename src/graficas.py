"""
graficas.py
===========
Las cinco graficas de Terra dibujadas sobre el lienzo propio y guardadas
como PNG: barras, lineas, histograma, dispersion y caja.

Todas comparten el mismo "marco":

    +---------------------------------------------------+
    |                     TITULO                        |
    |      +-------------------------------+  leyenda   |
    |  eje |  area de dibujo               |  ■ serie 1 |
    |   y  |  (con cuadricula punteada)    |  ■ serie 2 |
    |      +-------------------------------+            |
    |              marcas del eje x                     |
    |                  nombre del eje x                 |
    +---------------------------------------------------+

La parte mas importante es pasar de un valor de los datos a un pixel:
si el eje y va de y_min a y_max y el area de dibujo va del pixel
"abajo" al pixel "arriba", un valor v queda en

    pixel = abajo - (v - y_min) / (y_max - y_min) * (abajo - arriba)

(se resta porque en la imagen la y crece hacia abajo).

Los valores nulos se ignoran en todas las graficas.
"""

from src.lienzo import (Lienzo, PALETA, NEGRO, GRIS_OSCURO, GRIS_CLARO)
from src.tabla import ErrorDatos

ANCHO = 800
ALTO = 500
MARGEN_IZQ = 90
MARGEN_DER = 30
MARGEN_ARRIBA = 60
MARGEN_ABAJO = 75
ANCHO_LEYENDA = 150


# =====================================================================
# UTILIDADES NUMERICAS (sin el modulo math)
# =====================================================================

def _piso(x: float) -> int:
    """Mayor entero <= x. int() corta hacia cero, asi que en negativos
    con decimales hay que restar 1."""
    entero = int(x)
    return entero if x >= 0 or x == entero else entero - 1


def _techo(x: float) -> int:
    """Menor entero >= x."""
    return -_piso(-x)


def _potencia_de_diez(x: float) -> float:
    """La potencia de 10 mas grande que es <= x (x > 0)."""
    potencia = 1.0
    while potencia * 10 <= x:
        potencia *= 10
    while potencia > x:
        potencia /= 10
    return potencia


def marcas_bonitas(minimo: float, maximo: float, cantidad: int = 5):
    """
    Escoge las marcas de un eje: numeros "redondos" (pasos de 1, 2 o 5
    por una potencia de 10) que cubren desde minimo hasta maximo.

    Por ejemplo, de 38.7 a 92.5 da 30, 40, 50, ..., 100 en vez de marcas
    raras como 38.7, 49.5, 60.2...

    Devuelve (lista_de_marcas, paso).
    """
    if minimo == maximo:
        minimo, maximo = minimo - 1, maximo + 1
    paso_bruto = (maximo - minimo) / cantidad
    potencia = _potencia_de_diez(paso_bruto)
    for multiplo in (1, 2, 5, 10):
        if multiplo * potencia >= paso_bruto:
            paso = multiplo * potencia
            break
    inicio = _piso(minimo / paso + 1e-9) * paso
    fin = _techo(maximo / paso - 1e-9) * paso
    marcas = []
    valor = inicio
    while valor <= fin + paso / 2:
        marcas.append(round(valor, 10))
        valor += paso
    return marcas, paso


def formatear_numero(valor: float, paso: float = 1.0) -> str:
    """
    Texto corto para una marca del eje. Usa tantos decimales como
    necesite el paso (paso 0.25 -> 2 decimales) y abrevia los numeros
    grandes con K (miles) y M (millones).
    """
    if abs(valor) >= 1_000_000:
        return _quitar_ceros(f"{valor / 1_000_000:.2f}") + "M"
    if abs(valor) >= 10_000:
        return _quitar_ceros(f"{valor / 1_000:.1f}") + "K"
    decimales = 0
    while paso * (10 ** decimales) < 1 and decimales < 6:
        decimales += 1
    if paso * (10 ** decimales) != int(paso * (10 ** decimales)):
        decimales += 1
    return f"{valor:.{decimales}f}"


def _quitar_ceros(texto: str) -> str:
    return texto.rstrip("0").rstrip(".") if "." in texto else texto


def cuartil(ordenados: list, p: float) -> float:
    """
    Percentil p (entre 0 y 1) con interpolacion lineal, sobre una lista
    YA ORDENADA. La posicion es p * (n - 1); si cae entre dos datos se
    toma el punto intermedio proporcional.
    """
    posicion = p * (len(ordenados) - 1)
    abajo = _piso(posicion)
    arriba = _techo(posicion)
    fraccion = posicion - abajo
    return ordenados[abajo] + (ordenados[arriba] - ordenados[abajo]) * fraccion


def _numeros(valores, que: str) -> list:
    """Quita los nulos y verifica que lo demas sean numeros."""
    limpios = []
    for v in valores:
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ErrorDatos(f"{que} deben ser numeros, se encontro '{v}'")
        limpios.append(v)
    return limpios


# =====================================================================
# EL MARCO COMUN
# =====================================================================

class _Marco:
    """
    Lienzo con titulo, ejes, cuadricula y leyenda ya dibujados, y las
    funciones para convertir valores de los datos a pixeles.
    """

    def __init__(self, titulo, eje_x, eje_y, y_min, y_max,
                 leyendas=None, x_min=None, x_max=None):
        self.lienzo = Lienzo(ANCHO, ALTO)
        self.leyendas = leyendas or []
        derecha = MARGEN_DER + (ANCHO_LEYENDA if self.leyendas else 0)
        self.izq = MARGEN_IZQ
        self.der = ANCHO - derecha
        self.arriba = MARGEN_ARRIBA
        self.abajo = ALTO - MARGEN_ABAJO

        self.marcas_y, self.paso_y = marcas_bonitas(y_min, y_max)
        self.y_min, self.y_max = self.marcas_y[0], self.marcas_y[-1]

        self.numerico_x = x_min is not None
        if self.numerico_x:
            self.marcas_x, self.paso_x = marcas_bonitas(x_min, x_max)
            self.x_min, self.x_max = self.marcas_x[0], self.marcas_x[-1]

        self._dibujar_titulo(titulo)
        self._dibujar_eje_y(eje_y)
        if self.numerico_x:
            self._dibujar_eje_x_numerico()
        self._dibujar_nombre_x(eje_x)

    # --- conversion de valores a pixeles ---

    def py(self, valor: float) -> float:
        proporcion = (valor - self.y_min) / (self.y_max - self.y_min)
        return self.abajo - proporcion * (self.abajo - self.arriba)

    def px(self, valor: float) -> float:
        proporcion = (valor - self.x_min) / (self.x_max - self.x_min)
        return self.izq + proporcion * (self.der - self.izq)

    # --- partes del marco ---

    def _dibujar_titulo(self, titulo):
        if not titulo:
            return
        escala = 2
        while (Lienzo.ancho_texto(titulo, escala) > ANCHO - 20
               and escala > 1):
            escala -= 1
        texto = _recortar(titulo, ANCHO - 20, escala)
        ancho = Lienzo.ancho_texto(texto, escala)
        self.lienzo.texto((ANCHO - ancho) / 2, 20, texto, NEGRO, escala)

    def _dibujar_eje_y(self, nombre):
        for marca in self.marcas_y:
            y = self.py(marca)
            self.lienzo.linea_punteada(self.izq, y, self.der, y, GRIS_CLARO)
            texto = formatear_numero(marca, self.paso_y)
            ancho = Lienzo.ancho_texto(texto)
            self.lienzo.texto(self.izq - 8 - ancho, y - 3, texto, GRIS_OSCURO)
            self.lienzo.linea(self.izq - 4, y, self.izq, y, NEGRO)
        self.lienzo.linea(self.izq, self.arriba, self.izq, self.abajo, NEGRO)
        if nombre:
            texto = _recortar(nombre, self.abajo - self.arriba, 2)
            alto = Lienzo.ancho_texto(texto, 2)
            centro = (self.arriba + self.abajo) / 2
            self.lienzo.texto_vertical(14, centro + alto / 2, texto,
                                       NEGRO, 2)

    def _dibujar_eje_x_numerico(self):
        for marca in self.marcas_x:
            x = self.px(marca)
            self.lienzo.linea_punteada(x, self.arriba, x, self.abajo,
                                       GRIS_CLARO)
            texto = formatear_numero(marca, self.paso_x)
            ancho = Lienzo.ancho_texto(texto)
            self.lienzo.texto(x - ancho / 2, self.abajo + 10, texto,
                              GRIS_OSCURO)
            self.lienzo.linea(x, self.abajo, x, self.abajo + 4, NEGRO)
        self.dibujar_linea_base()

    def dibujar_linea_base(self):
        self.lienzo.linea(self.izq, self.abajo, self.der, self.abajo, NEGRO)

    def _dibujar_nombre_x(self, nombre):
        if not nombre:
            return
        texto = _recortar(nombre, self.der - self.izq, 2)
        ancho = Lienzo.ancho_texto(texto, 2)
        centro = (self.izq + self.der) / 2
        self.lienzo.texto(centro - ancho / 2, ALTO - 30, texto, NEGRO, 2)

    def etiquetas_categorias(self, nombres, centros):
        """Nombres debajo de cada barra o caja. Si no caben derechos, se
        escriben verticales."""
        espacio = (self.der - self.izq) / max(len(nombres), 1)
        caben = all(Lienzo.ancho_texto(str(n)) <= espacio - 4
                    for n in nombres)
        for nombre, x in zip(nombres, centros):
            texto = str(nombre)
            if caben:
                ancho = Lienzo.ancho_texto(texto)
                self.lienzo.texto(x - ancho / 2, self.abajo + 10, texto,
                                  GRIS_OSCURO)
            else:
                texto = _recortar(texto, MARGEN_ABAJO - 40, 1)
                self.lienzo.texto_vertical(
                    x - 3, self.abajo + 8 + Lienzo.ancho_texto(texto),
                    texto, GRIS_OSCURO)

    def dibujar_leyenda(self):
        if not self.leyendas:
            return
        x = self.der + 20
        y = self.arriba
        for i, nombre in enumerate(self.leyendas):
            color = PALETA[i % len(PALETA)]
            self.lienzo.rectangulo(x, y, x + 10, y + 10, color)
            texto = _recortar(str(nombre), ANCHO_LEYENDA - 30, 1)
            self.lienzo.texto(x + 16, y + 2, texto, NEGRO)
            y += 18

    def guardar(self, ruta):
        self.dibujar_leyenda()
        self.lienzo.guardar_png(ruta)


def _recortar(texto: str, ancho_maximo: float, escala: int) -> str:
    """Corta el texto con '..' si no cabe en el ancho dado."""
    texto = str(texto)
    if Lienzo.ancho_texto(texto, escala) <= ancho_maximo:
        return texto
    while texto and Lienzo.ancho_texto(texto + "..", escala) > ancho_maximo:
        texto = texto[:-1]
    return texto + ".."


def _rango(valores: list):
    if not valores:
        raise ErrorDatos("no hay datos numericos para graficar")
    return min(valores), max(valores)


# =====================================================================
# LAS CINCO GRAFICAS
# =====================================================================

def grafica_barras(categorias, valores, ruta, titulo="", eje_x="",
                   eje_y="", leyenda=None):
    """Una barra por categoria. El eje y siempre incluye el cero, para
    que el tamano de las barras se pueda comparar."""
    if len(categorias) != len(valores):
        raise ErrorDatos("barras: debe haber un valor por cada categoria")
    pares = [(c, v) for c, v in zip(categorias, valores) if v is not None]
    numeros = _numeros([v for _, v in pares], "los valores de las barras")
    minimo, maximo = _rango(numeros)
    marco = _Marco(titulo, eje_x, eje_y, min(0, minimo), max(0, maximo),
                   leyendas=[leyenda] if leyenda else None)

    n = len(pares)
    espacio = (marco.der - marco.izq) / n
    ancho_barra = max(espacio * 0.7, 1)
    centros = []
    cero = marco.py(0)
    for i, (_, valor) in enumerate(pares):
        centro = marco.izq + espacio * (i + 0.5)
        centros.append(centro)
        marco.lienzo.rectangulo(centro - ancho_barra / 2, marco.py(valor),
                                centro + ancho_barra / 2, cero, PALETA[0])
    marco.lienzo.linea(marco.izq, cero, marco.der, cero, NEGRO)
    marco.dibujar_linea_base()
    marco.etiquetas_categorias([c for c, _ in pares], centros)
    marco.guardar(ruta)


def grafica_lineas(xs, series, ruta, titulo="", eje_x="", eje_y="",
                   leyendas=None):
    """
    Una o varias lineas. series es una lista de listas (una por serie),
    todas del mismo largo que xs. Los puntos se ordenan por x antes de
    unirlos, para que la linea no vaya y vuelva.
    """
    xs_validos = _numeros(xs, "los valores del eje x")
    todos_y = []
    for serie in series:
        if len(serie) != len(xs):
            raise ErrorDatos("lineas: cada serie debe tener tantos valores "
                             "como el eje x")
        todos_y += _numeros(serie, "los valores del eje y")
    x_min, x_max = _rango(xs_validos)
    y_min, y_max = _rango(todos_y)
    marco = _Marco(titulo, eje_x, eje_y, y_min, y_max, leyendas=leyendas,
                   x_min=x_min, x_max=x_max)

    for i, serie in enumerate(series):
        color = PALETA[i % len(PALETA)]
        puntos = sorted((x, y) for x, y in zip(xs, serie)
                        if x is not None and y is not None)
        for (x0, y0), (x1, y1) in zip(puntos, puntos[1:]):
            marco.lienzo.linea(marco.px(x0), marco.py(y0),
                               marco.px(x1), marco.py(y1), color, 2)
        for x, y in puntos:
            marco.lienzo.circulo(marco.px(x), marco.py(y), 3, color)
    marco.guardar(ruta)


def grafica_dispersion(xs, ys, ruta, titulo="", eje_x="", eje_y="",
                       grupos=None, leyendas=None):
    """
    Un punto por cada pareja (x, y). Si se da grupos (una etiqueta por
    punto), cada grupo sale de un color distinto con su leyenda.
    """
    if len(xs) != len(ys):
        raise ErrorDatos("dispersion: x e y deben tener el mismo largo")
    pares = [(x, y, g) for x, y, g in
             zip(xs, ys, grupos if grupos else [None] * len(xs))
             if x is not None and y is not None]
    x_min, x_max = _rango(_numeros([p[0] for p in pares], "los valores x"))
    y_min, y_max = _rango(_numeros([p[1] for p in pares], "los valores y"))

    if grupos:
        leyendas = []
        for _, _, g in pares:
            if g not in leyendas:
                leyendas.append(g)
    marco = _Marco(titulo, eje_x, eje_y, y_min, y_max, leyendas=leyendas,
                   x_min=x_min, x_max=x_max)
    for x, y, g in pares:
        indice = leyendas.index(g) if grupos else 0
        marco.lienzo.circulo(marco.px(x), marco.py(y), 4,
                             PALETA[indice % len(PALETA)])
    marco.guardar(ruta)


def grafica_histograma(valores, ruta, bins=10, titulo="", eje_x="",
                       eje_y="frecuencia", leyenda=None):
    """
    Divide el rango de los datos en 'bins' intervalos del mismo ancho y
    dibuja cuantas observaciones caen en cada uno. El ultimo intervalo
    incluye el maximo.
    """
    numeros = _numeros(valores, "los valores del histograma")
    minimo, maximo = _rango(numeros)
    if minimo == maximo:
        minimo, maximo = minimo - 0.5, maximo + 0.5
    ancho_bin = (maximo - minimo) / bins
    conteos = [0] * bins
    for v in numeros:
        indice = int((v - minimo) / ancho_bin)
        conteos[min(indice, bins - 1)] += 1

    marco = _Marco(titulo, eje_x, eje_y, 0, max(conteos),
                   leyendas=[leyenda] if leyenda else None,
                   x_min=minimo, x_max=maximo)
    for i, conteo in enumerate(conteos):
        if conteo == 0:
            continue
        izquierda = marco.px(minimo + i * ancho_bin)
        derecha = marco.px(minimo + (i + 1) * ancho_bin)
        marco.lienzo.rectangulo(izquierda + 1, marco.py(conteo),
                                derecha - 1, marco.py(0), PALETA[0])
    marco.dibujar_linea_base()
    marco.guardar(ruta)
    return conteos


def grafica_caja(grupos, nombres, ruta, titulo="", eje_x="", eje_y=""):
    """
    Diagrama de caja (box plot) para uno o varios grupos de datos:

        - la caja va del cuartil 1 (25 %) al cuartil 3 (75 %),
        - la raya del medio es la mediana,
        - los "bigotes" llegan hasta el dato mas lejano que este a menos
          de 1.5 veces el rango intercuartil (Q3 - Q1) de la caja,
        - los datos mas alla de los bigotes son atipicos y se dibujan
          como puntos sueltos.
    """
    listas = [sorted(_numeros(g, "los valores de la caja")) for g in grupos]
    if any(len(lista) == 0 for lista in listas):
        raise ErrorDatos("caja: cada grupo necesita al menos un dato")
    y_min = min(lista[0] for lista in listas)
    y_max = max(lista[-1] for lista in listas)
    marco = _Marco(titulo, eje_x, eje_y, y_min, y_max)

    espacio = (marco.der - marco.izq) / len(listas)
    ancho_caja = min(espacio * 0.5, 120)
    centros = []
    resumenes = []
    for i, datos in enumerate(listas):
        color = PALETA[i % len(PALETA)]
        centro = marco.izq + espacio * (i + 0.5)
        centros.append(centro)
        q1 = cuartil(datos, 0.25)
        mediana = cuartil(datos, 0.5)
        q3 = cuartil(datos, 0.75)
        rango_ic = q3 - q1
        bigote_bajo = min(v for v in datos if v >= q1 - 1.5 * rango_ic)
        bigote_alto = max(v for v in datos if v <= q3 + 1.5 * rango_ic)
        atipicos = [v for v in datos if v < bigote_bajo or v > bigote_alto]
        resumenes.append({"q1": q1, "mediana": mediana, "q3": q3,
                          "bigote_bajo": bigote_bajo,
                          "bigote_alto": bigote_alto,
                          "atipicos": atipicos})

        izq = centro - ancho_caja / 2
        der = centro + ancho_caja / 2
        marco.lienzo.linea(centro, marco.py(bigote_alto),
                           centro, marco.py(q3), NEGRO)
        marco.lienzo.linea(centro, marco.py(q1),
                           centro, marco.py(bigote_bajo), NEGRO)
        for extremo in (bigote_bajo, bigote_alto):
            marco.lienzo.linea(centro - ancho_caja / 4, marco.py(extremo),
                               centro + ancho_caja / 4, marco.py(extremo),
                               NEGRO)
        marco.lienzo.rectangulo(izq, marco.py(q3), der, marco.py(q1), color)
        marco.lienzo.marco(izq, marco.py(q3), der, marco.py(q1), NEGRO)
        marco.lienzo.linea(izq, marco.py(mediana), der, marco.py(mediana),
                           NEGRO, 3)
        for v in atipicos:
            marco.lienzo.circulo(centro, marco.py(v), 3, NEGRO)

    marco.dibujar_linea_base()
    marco.etiquetas_categorias(nombres, centros)
    marco.guardar(ruta)
    return resumenes
