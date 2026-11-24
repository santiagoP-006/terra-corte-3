"""
lienzo.py
=========
Un "lienzo" de pixeles para dibujar graficas, escrito desde cero.

El lienzo es una cuadricula de ancho x alto pixeles guardada en un
bytearray: 3 bytes (rojo, verde, azul) por pixel, fila por fila.
Sobre esa cuadricula se dibujan rectangulos, lineas, circulos y texto,
y al final se guarda como PNG con el modulo png.py.

Coordenadas: (0, 0) es la esquina superior izquierda. x crece hacia la
derecha y y crece hacia ABAJO, como en casi todos los formatos de imagen.

El texto usa una fuente de mapa de bits de 5 x 7 pixeles definida a mano
al final de este archivo. Solo tiene mayusculas, digitos y algunos
signos: las minusculas se dibujan en mayuscula y las tildes se quitan.
"""

from src import png

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS_OSCURO = (90, 90, 90)
GRIS_CLARO = (225, 225, 225)

# Colores para las series de datos (se escogieron para que se distingan
# entre si y tambien en escala de grises)
PALETA = [
    (31, 119, 180),    # azul
    (255, 127, 14),    # naranja
    (44, 160, 44),     # verde
    (214, 39, 40),     # rojo
    (148, 103, 189),   # morado
    (140, 86, 75),     # cafe
    (227, 119, 194),   # rosado
    (23, 190, 207),    # cian
]

ANCHO_LETRA = 5
ALTO_LETRA = 7
SEPARACION_LETRA = 1


class Lienzo:
    """Cuadricula de pixeles RGB sobre la que se dibuja."""

    def __init__(self, ancho: int, alto: int, fondo=BLANCO):
        self.ancho = ancho
        self.alto = alto
        self.datos = bytearray(bytes(fondo) * (ancho * alto))

    # -----------------------------------------------------------------
    # Primitivas
    # -----------------------------------------------------------------

    def pixel(self, x: int, y: int, color) -> None:
        """Pinta un pixel. Lo que cae fuera del lienzo se ignora."""
        if 0 <= x < self.ancho and 0 <= y < self.alto:
            i = (y * self.ancho + x) * 3
            self.datos[i:i + 3] = bytes(color)

    def color_en(self, x: int, y: int) -> tuple:
        """Color del pixel (x, y). Sirve para las pruebas."""
        i = (y * self.ancho + x) * 3
        return tuple(self.datos[i:i + 3])

    def rectangulo(self, x0, y0, x1, y1, color) -> None:
        """
        Rectangulo relleno entre las esquinas (x0, y0) y (x1, y1),
        incluidas. Se pinta fila por fila copiando un trozo de bytes,
        que es mucho mas rapido que pintar pixel por pixel.
        """
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        x0, x1 = max(x0, 0), min(x1, self.ancho - 1)
        y0, y1 = max(y0, 0), min(y1, self.alto - 1)
        if x0 > x1 or y0 > y1:
            return
        trozo = bytes(color) * (x1 - x0 + 1)
        for y in range(y0, y1 + 1):
            i = (y * self.ancho + x0) * 3
            self.datos[i:i + len(trozo)] = trozo

    def marco(self, x0, y0, x1, y1, color) -> None:
        """Contorno de un rectangulo (sin relleno)."""
        self.rectangulo(x0, y0, x1, y0, color)
        self.rectangulo(x0, y1, x1, y1, color)
        self.rectangulo(x0, y0, x0, y1, color)
        self.rectangulo(x1, y0, x1, y1, color)

    def linea(self, x0, y0, x1, y1, color, grosor: int = 1) -> None:
        """
        Linea recta con el algoritmo de Bresenham.

        La idea: se avanza de a un pixel en la direccion donde la linea
        cambia mas, y se lleva un "error" acumulado que dice cuando hay
        que subir o bajar un pixel en la otra direccion. Solo usa sumas y
        restas de enteros. Para lineas gruesas se pinta un cuadradito de
        grosor x grosor en cada punto.
        """
        x0, y0 = int(round(x0)), int(round(y0))
        x1, y1 = int(round(x1)), int(round(y1))
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        paso_x = 1 if x0 < x1 else -1
        paso_y = 1 if y0 < y1 else -1
        error = dx + dy
        mitad = grosor // 2
        while True:
            if grosor == 1:
                self.pixel(x0, y0, color)
            else:
                self.rectangulo(x0 - mitad, y0 - mitad,
                                x0 - mitad + grosor - 1,
                                y0 - mitad + grosor - 1, color)
            if x0 == x1 and y0 == y1:
                break
            doble = 2 * error
            if doble >= dy:
                error += dy
                x0 += paso_x
            if doble <= dx:
                error += dx
                y0 += paso_y

    def linea_punteada(self, x0, y0, x1, y1, color, trazo: int = 4) -> None:
        """Linea horizontal o vertical punteada (para la cuadricula)."""
        x0, y0 = int(round(x0)), int(round(y0))
        x1, y1 = int(round(x1)), int(round(y1))
        if y0 == y1:
            for x in range(min(x0, x1), max(x0, x1) + 1):
                if (x // trazo) % 2 == 0:
                    self.pixel(x, y0, color)
        else:
            for y in range(min(y0, y1), max(y0, y1) + 1):
                if (y // trazo) % 2 == 0:
                    self.pixel(x0, y, color)

    def circulo(self, cx, cy, radio: int, color) -> None:
        """
        Circulo relleno: se pinta cada pixel del cuadrado que lo encierra
        cuya distancia al centro sea menor o igual al radio. Se compara
        con el cuadrado de la distancia para no necesitar raiz cuadrada.
        """
        cx, cy = int(round(cx)), int(round(cy))
        r2 = radio * radio
        for dy in range(-radio, radio + 1):
            for dx in range(-radio, radio + 1):
                if dx * dx + dy * dy <= r2:
                    self.pixel(cx + dx, cy + dy, color)

    # -----------------------------------------------------------------
    # Texto
    # -----------------------------------------------------------------

    @staticmethod
    def ancho_texto(texto: str, escala: int = 1) -> int:
        """Ancho en pixeles que ocupara un texto."""
        if not texto:
            return 0
        return (len(texto) * (ANCHO_LETRA + SEPARACION_LETRA)
                - SEPARACION_LETRA) * escala

    @staticmethod
    def alto_texto(escala: int = 1) -> int:
        return ALTO_LETRA * escala

    def texto(self, x, y, texto: str, color=NEGRO, escala: int = 1) -> None:
        """
        Escribe un texto horizontal con la esquina superior izquierda en
        (x, y). Cada pixel de la fuente se dibuja como un cuadrado de
        escala x escala, asi que escala=2 da letras del doble de grandes.
        """
        x, y = int(round(x)), int(round(y))
        for caracter in texto:
            for fila, patron in enumerate(_glifo(caracter)):
                for columna, marca in enumerate(patron):
                    if marca == "#":
                        px = x + columna * escala
                        py = y + fila * escala
                        self.rectangulo(px, py, px + escala - 1,
                                        py + escala - 1, color)
            x += (ANCHO_LETRA + SEPARACION_LETRA) * escala

    def texto_vertical(self, x, y, texto: str, color=NEGRO,
                       escala: int = 1) -> None:
        """
        Escribe un texto girado 90 grados, para leerse de abajo hacia
        arriba (como el nombre del eje y). (x, y) es la esquina inferior
        izquierda del texto. Girar es cambiar los papeles de fila y
        columna: la columna de la letra se vuelve la altura y la fila se
        vuelve la distancia hacia la derecha.
        """
        x, y = int(round(x)), int(round(y))
        for caracter in texto:
            for fila, patron in enumerate(_glifo(caracter)):
                for columna, marca in enumerate(patron):
                    if marca == "#":
                        px = x + fila * escala
                        py = y - columna * escala
                        self.rectangulo(px, py - escala + 1,
                                        px + escala - 1, py, color)
            y -= (ANCHO_LETRA + SEPARACION_LETRA) * escala

    # -----------------------------------------------------------------
    # Salida
    # -----------------------------------------------------------------

    def guardar_png(self, ruta: str) -> None:
        png.escribir_png(ruta, self.ancho, self.alto, self.datos)


# =====================================================================
# FUENTE DE 5 x 7 PIXELES
# =====================================================================
# Cada letra son 7 filas de 5 caracteres: '#' es pixel pintado y '.'
# es vacio. Asi se puede "ver" la letra leyendo el codigo.

_FUENTE_TEXTO = {
    "A": ".###. #...# #...# ##### #...# #...# #...#",
    "B": "####. #...# #...# ####. #...# #...# ####.",
    "C": ".###. #...# #.... #.... #.... #...# .###.",
    "D": "###.. #..#. #...# #...# #...# #..#. ###..",
    "E": "##### #.... #.... ####. #.... #.... #####",
    "F": "##### #.... #.... ####. #.... #.... #....",
    "G": ".###. #...# #.... #.### #...# #...# .####",
    "H": "#...# #...# #...# ##### #...# #...# #...#",
    "I": ".###. ..#.. ..#.. ..#.. ..#.. ..#.. .###.",
    "J": "..### ...#. ...#. ...#. ...#. #..#. .##..",
    "K": "#...# #..#. #.#.. ##... #.#.. #..#. #...#",
    "L": "#.... #.... #.... #.... #.... #.... #####",
    "M": "#...# ##.## #.#.# #.#.# #...# #...# #...#",
    "N": "#...# #...# ##..# #.#.# #..## #...# #...#",
    "Ñ": ".###. ..... #...# ##..# #.#.# #..## #...#",
    "O": ".###. #...# #...# #...# #...# #...# .###.",
    "P": "####. #...# #...# ####. #.... #.... #....",
    "Q": ".###. #...# #...# #...# #.#.# #..#. .##.#",
    "R": "####. #...# #...# ####. #.#.. #..#. #...#",
    "S": ".#### #.... #.... .###. ....# ....# ####.",
    "T": "##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
    "U": "#...# #...# #...# #...# #...# #...# .###.",
    "V": "#...# #...# #...# #...# #...# .#.#. ..#..",
    "W": "#...# #...# #...# #.#.# #.#.# #.#.# .#.#.",
    "X": "#...# #...# .#.#. ..#.. .#.#. #...# #...#",
    "Y": "#...# #...# .#.#. ..#.. ..#.. ..#.. ..#..",
    "Z": "##### ....# ...#. ..#.. .#... #.... #####",
    "0": ".###. #...# #..## #.#.# ##..# #...# .###.",
    "1": "..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.",
    "2": ".###. #...# ....# ...#. ..#.. .#... #####",
    "3": "##### ...#. ..#.. ...#. ....# #...# .###.",
    "4": "...#. ..##. .#.#. #..#. ##### ...#. ...#.",
    "5": "##### #.... ####. ....# ....# #...# .###.",
    "6": "..##. .#... #.... ####. #...# #...# .###.",
    "7": "##### ....# ...#. ..#.. .#... .#... .#...",
    "8": ".###. #...# #...# .###. #...# #...# .###.",
    "9": ".###. #...# #...# .#### ....# ...#. .##..",
    " ": "..... ..... ..... ..... ..... ..... .....",
    ".": "..... ..... ..... ..... ..... .##.. .##..",
    ",": "..... ..... ..... ..... .##.. ..#.. .#...",
    ":": "..... .##.. .##.. ..... .##.. .##.. .....",
    "-": "..... ..... ..... ##### ..... ..... .....",
    "+": "..... ..#.. ..#.. ##### ..#.. ..#.. .....",
    "=": "..... ..... ##### ..... ##### ..... .....",
    "_": "..... ..... ..... ..... ..... ..... #####",
    "/": "....# ...#. ...#. ..#.. .#... .#... #....",
    "%": "##..# ##.#. ...#. ..#.. .#... .#.## #..##",
    "(": "...#. ..#.. .#... .#... .#... ..#.. ...#.",
    ")": ".#... ..#.. ...#. ...#. ...#. ..#.. .#...",
    "'": "..#.. ..#.. ..... ..... ..... ..... .....",
    "!": "..#.. ..#.. ..#.. ..#.. ..#.. ..... ..#..",
    "?": ".###. #...# ....# ...#. ..#.. ..... ..#..",
    "#": ".#.#. ##### .#.#. .#.#. .#.#. ##### .#.#.",
    "<": "...#. ..#.. .#... #.... .#... ..#.. ...#.",
    ">": ".#... ..#.. ...#. ....# ...#. ..#.. .#...",
}

# Cada texto se parte en sus 7 filas una sola vez, al cargar el modulo
FUENTE = {letra: dibujo.split() for letra, dibujo in _FUENTE_TEXTO.items()}

# Letras con tilde y otros caracteres que se dibujan como uno parecido
_EQUIVALENTES = {
    "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ü": "U",
    "₂": "2", "²": "2", "³": "3", "°": "'",
}


def _glifo(caracter: str) -> list:
    """Dibujo de 7 filas para un caracter; '?' si la fuente no lo tiene."""
    letra = caracter.upper()
    letra = _EQUIVALENTES.get(letra, letra)
    return FUENTE.get(letra, FUENTE["?"])
