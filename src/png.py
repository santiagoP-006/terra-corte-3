"""
png.py
======
Escritor (y lector de verificacion) de imagenes PNG, escrito desde cero.

No usa zlib, struct ni ninguna libreria: todo lo que necesita un PNG
se calcula aqui a mano.

Como esta hecho un archivo PNG
------------------------------
    1. Firma de 8 bytes que identifica el archivo como PNG.
    2. Una serie de bloques ("chunks"). Cada bloque tiene:
           longitud (4 bytes) | tipo (4 letras) | datos | CRC-32 (4 bytes)
       El CRC-32 es una suma de verificacion: si un byte se dana, el
       visor de imagenes lo detecta.
    3. Los bloques que usamos son:
           IHDR  ancho, alto y tipo de color (RGB, 8 bits por canal)
           IDAT  los pixeles, empaquetados en formato zlib
           IEND  marca el final del archivo

Los pixeles en formato zlib
---------------------------
zlib normalmente comprime, pero tambien permite guardar los datos en
"bloques almacenados", es decir, sin comprimir. Eso hacemos aqui: el
archivo queda mas pesado, pero cualquier visor lo abre igual y no hace
falta implementar el algoritmo de compresion. Un flujo zlib es:

    2 bytes de encabezado | bloques almacenados | Adler-32 (4 bytes)

Cada bloque almacenado guarda hasta 65 535 bytes. Adler-32 es otra suma
de verificacion, distinta del CRC, que zlib exige al final.

Antes de cada fila de pixeles va un byte de "filtro". Usamos el filtro 0
(ninguno): la fila se guarda tal cual.

Uso
---
    escribir_png("salidas/grafica.png", ancho, alto, datos_rgb)

    datos_rgb es un bytearray de ancho * alto * 3 bytes: los pixeles fila
    por fila, de izquierda a derecha, cada uno como rojo, verde, azul.
"""

from src.tabla import ErrorDatos

FIRMA_PNG = bytes([137, 80, 78, 71, 13, 10, 26, 10])   # \x89 P N G \r \n \x1a \n
MAXIMO_BLOQUE = 65535        # bytes que caben en un bloque almacenado de zlib
MODULO_ADLER = 65521         # el primo mas grande menor que 2^16


# =====================================================================
# SUMAS DE VERIFICACION
# =====================================================================

def _tabla_crc() -> list:
    """
    Precalcula el CRC de los 256 valores posibles de un byte.

    El CRC-32 divide los datos (vistos como un polinomio en binario)
    entre un polinomio fijo y se queda con el residuo. Hacer esa division
    bit por bit es lento, asi que se calcula una vez el resultado para
    cada byte posible y luego solo se consulta la tabla.
    0xEDB88320 es el polinomio estandar de PNG escrito "al reves".
    """
    tabla = []
    for n in range(256):
        c = n
        for _ in range(8):
            if c & 1:
                c = 0xEDB88320 ^ (c >> 1)
            else:
                c = c >> 1
        tabla.append(c)
    return tabla


_TABLA_CRC = _tabla_crc()


def crc32(datos: bytes) -> int:
    """CRC-32 de una secuencia de bytes, tal como lo define PNG."""
    c = 0xFFFFFFFF
    tabla = _TABLA_CRC
    for byte in datos:
        c = tabla[(c ^ byte) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def adler32(datos: bytes) -> int:
    """
    Adler-32: dos sumas acumuladas modulo 65521.

    a suma los bytes; b suma los valores que va tomando a. El resultado
    es b en los 16 bits altos y a en los 16 bits bajos.
    """
    a, b = 1, 0
    # Se reduce modulo cada 5552 bytes (el maximo que cabe sin que b se
    # desborde en la definicion original); en Python no hay desborde,
    # pero asi los numeros se mantienen pequenos y es mas rapido.
    for inicio in range(0, len(datos), 5552):
        for byte in datos[inicio:inicio + 5552]:
            a += byte
            b += a
        a %= MODULO_ADLER
        b %= MODULO_ADLER
    return (b << 16) | a


# =====================================================================
# ARMADO DEL ARCHIVO
# =====================================================================

def _entero_4(n: int) -> bytes:
    """Entero sin signo de 4 bytes, el byte mas significativo primero."""
    return n.to_bytes(4, "big")


def _bloque(tipo: bytes, datos: bytes) -> bytes:
    """Un chunk de PNG: longitud, tipo, datos y CRC del tipo + datos."""
    return (_entero_4(len(datos)) + tipo + datos
            + _entero_4(crc32(tipo + datos)))


def _zlib_sin_comprimir(datos: bytes) -> bytes:
    """
    Envuelve los datos en un flujo zlib valido, sin comprimirlos.

    Encabezado 0x78 0x01: metodo "deflate" con ventana de 32 KB y sin
    diccionario. (0x7801 es multiplo de 31, como exige zlib.)
    Cada bloque almacenado empieza con un byte que dice si es el ultimo
    (1) o no (0), seguido de su longitud y la longitud negada, ambas de
    2 bytes con el byte menos significativo primero.
    """
    salida = bytearray([0x78, 0x01])
    total = len(datos)
    inicio = 0
    while True:
        trozo = datos[inicio:inicio + MAXIMO_BLOQUE]
        inicio += len(trozo)
        es_ultimo = inicio >= total
        largo = len(trozo)
        salida.append(1 if es_ultimo else 0)
        salida += largo.to_bytes(2, "little")
        salida += (largo ^ 0xFFFF).to_bytes(2, "little")
        salida += trozo
        if es_ultimo:
            break
    salida += _entero_4(adler32(datos))
    return bytes(salida)


def codificar_png(ancho: int, alto: int, datos_rgb) -> bytes:
    """Devuelve los bytes de un PNG RGB de ancho x alto."""
    if ancho < 1 or alto < 1:
        raise ErrorDatos("la imagen debe medir al menos 1 x 1 pixeles")
    if len(datos_rgb) != ancho * alto * 3:
        raise ErrorDatos(
            f"se esperaban {ancho * alto * 3} bytes de pixeles "
            f"({ancho} x {alto} x 3), se recibieron {len(datos_rgb)}")

    # Cada fila lleva delante el byte de filtro 0
    bytes_por_fila = ancho * 3
    filas = bytearray()
    for y in range(alto):
        filas.append(0)
        filas += datos_rgb[y * bytes_por_fila:(y + 1) * bytes_por_fila]

    encabezado = (_entero_4(ancho) + _entero_4(alto)
                  + bytes([8,      # 8 bits por canal
                           2,      # tipo de color 2 = RGB
                           0,      # compresion deflate (la unica que existe)
                           0,      # filtrado estandar
                           0]))    # sin entrelazado

    return (FIRMA_PNG
            + _bloque(b"IHDR", encabezado)
            + _bloque(b"IDAT", _zlib_sin_comprimir(bytes(filas)))
            + _bloque(b"IEND", b""))


def escribir_png(ruta: str, ancho: int, alto: int, datos_rgb) -> None:
    """Guarda la imagen en disco como archivo PNG."""
    contenido = codificar_png(ancho, alto, datos_rgb)
    try:
        with open(ruta, "wb") as archivo:
            archivo.write(contenido)
    except FileNotFoundError:
        raise ErrorDatos(
            f"no se pudo guardar '{ruta}': la carpeta no existe")
    except OSError as e:
        raise ErrorDatos(f"no se pudo guardar '{ruta}': {e.strerror}")


# =====================================================================
# LECTURA (para verificar en las pruebas)
# =====================================================================

def leer_png(ruta: str):
    """
    Lee un PNG escrito por este modulo y verifica que este bien formado.

    Comprueba la firma, el CRC de cada bloque y el Adler-32, y devuelve
    (ancho, alto, datos_rgb). Solo entiende PNG RGB sin comprimir, que es
    lo que produce escribir_png; sirve para que las pruebas confirmen que
    el archivo es valido sin depender de ninguna libreria.
    """
    with open(ruta, "rb") as archivo:
        contenido = archivo.read()

    if contenido[:8] != FIRMA_PNG:
        raise ErrorDatos(f"'{ruta}' no es un PNG: la firma no coincide")

    posicion = 8
    ancho = alto = None
    idat = bytearray()
    while posicion < len(contenido):
        largo = int.from_bytes(contenido[posicion:posicion + 4], "big")
        tipo = contenido[posicion + 4:posicion + 8]
        datos = contenido[posicion + 8:posicion + 8 + largo]
        crc = int.from_bytes(
            contenido[posicion + 8 + largo:posicion + 12 + largo], "big")
        if crc32(tipo + datos) != crc:
            raise ErrorDatos(f"CRC incorrecto en el bloque {tipo.decode()}")
        if tipo == b"IHDR":
            ancho = int.from_bytes(datos[0:4], "big")
            alto = int.from_bytes(datos[4:8], "big")
        elif tipo == b"IDAT":
            idat += datos
        elif tipo == b"IEND":
            break
        posicion += 12 + largo

    # Desempacar los bloques almacenados de zlib
    crudo = bytearray()
    i = 2                                  # saltar el encabezado 0x78 0x01
    while True:
        es_ultimo = idat[i] & 1
        largo = int.from_bytes(idat[i + 1:i + 3], "little")
        crudo += idat[i + 5:i + 5 + largo]
        i += 5 + largo
        if es_ultimo:
            break
    if int.from_bytes(idat[i:i + 4], "big") != adler32(bytes(crudo)):
        raise ErrorDatos("Adler-32 incorrecto en los datos de la imagen")

    # Quitar el byte de filtro de cada fila
    bytes_por_fila = ancho * 3
    datos_rgb = bytearray()
    for y in range(alto):
        fila = y * (bytes_por_fila + 1)
        datos_rgb += crudo[fila + 1:fila + 1 + bytes_por_fila]
    return ancho, alto, datos_rgb
