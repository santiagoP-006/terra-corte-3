"""
visualizacion.py
================
Puente entre la sentencia 'graficar' del lenguaje y las graficas.

    graficar barras resumen
        eje_x pais
        eje_y emisiones
        titulo "Emisiones por pais"
        leyenda "CO2 total"
        guardar como "salidas/emisiones.png"

El evaluador (visitante_evaluador.py) solo lee el arbol y arma un
diccionario con las opciones; este modulo hace el resto: revisa que las
columnas existan y sean del tipo correcto, saca los datos de la tabla y
llama a graficas.py (PNG) o a terra_plot.py (dibujo en la terminal).
Igual que operaciones.py, no conoce el arbol de ANTLR y lanza ErrorDatos;
el evaluador le agrega la linea.

Que pide cada tipo de grafica
-----------------------------
    barras      eje_x: columna de categorias   eje_y: columna numerica
                (si una categoria se repite, se suman sus valores)
    lineas      eje_x y eje_y numericas
    dispersion  eje_x y eje_y numericas
    histograma  eje_x (o eje_y) numerica: la columna que se reparte
    caja        eje_y numerica; eje_x opcional: una caja por categoria

La opcion 'leyenda'
-------------------
    - Si el texto es el nombre de una columna y la grafica es de lineas,
      dispersion o caja, los datos se separan por esa columna: una linea,
      un color o una caja por cada valor (por ejemplo, una por pais).
    - Si no, el texto se muestra como nombre de la serie en la leyenda.

Si no se escribe 'guardar como', la grafica se dibuja en la terminal.
"""

from src import graficas
from src import terra_plot as plt
from src.tabla import Tabla, ErrorDatos

TIPOS = ("barras", "lineas", "histograma", "dispersion", "caja")


def graficar(tipo: str, tabla, opciones: dict) -> str:
    """
    Dibuja la grafica. opciones puede tener: eje_x, eje_y, titulo,
    leyenda y ruta. Devuelve un mensaje para mostrar al usuario.
    """
    if tipo not in TIPOS:
        raise ErrorDatos(f"tipo de grafica desconocido: '{tipo}'")
    if not isinstance(tabla, Tabla):
        raise ErrorDatos("graficar necesita una tabla")
    if tabla.num_filas == 0:
        raise ErrorDatos("la tabla esta vacia: no hay nada que graficar")

    ruta = opciones.get("ruta")
    if ruta is not None and not ruta.lower().endswith(".png"):
        raise ErrorDatos(
            f"las graficas se guardan en formato PNG: la ruta '{ruta}' "
            f"debe terminar en .png")

    dibujar = {
        "barras": _barras,
        "lineas": _lineas,
        "histograma": _histograma,
        "dispersion": _dispersion,
        "caja": _caja,
    }[tipo]
    dibujar(tabla, opciones, ruta)
    if ruta is None:
        return None
    return f"Grafica de {tipo} guardada en {ruta}"


# =====================================================================
# AYUDANTES
# =====================================================================

def _exigir(opciones: dict, eje: str, tipo: str) -> str:
    if not opciones.get(eje):
        raise ErrorDatos(f"la grafica de {tipo} necesita la opcion '{eje}'")
    return opciones[eje]


def _columna(tabla, nombre: str) -> list:
    if not tabla.tiene_columna(nombre):
        disponibles = ", ".join(tabla.columnas)
        raise ErrorDatos(
            f"la columna '{nombre}' no existe; columnas disponibles: "
            f"{disponibles}")
    return tabla.columna(nombre)


def _columna_numerica(tabla, nombre: str) -> list:
    valores = _columna(tabla, nombre)
    for v in valores:
        if v is not None and (isinstance(v, bool)
                              or not isinstance(v, (int, float))):
            raise ErrorDatos(
                f"la columna '{nombre}' debe ser numerica para esta "
                f"grafica, pero tiene el valor '{v}'")
    return valores


def _grupos_por_leyenda(tabla, opciones):
    """Si la leyenda es el nombre de una columna, devuelve sus valores."""
    leyenda = opciones.get("leyenda")
    if leyenda and tabla.tiene_columna(leyenda):
        return _columna(tabla, leyenda)
    return None


def _en_orden(valores) -> list:
    """Valores distintos en el orden en que aparecen por primera vez."""
    vistos = []
    for v in valores:
        if v not in vistos:
            vistos.append(v)
    return vistos


def _ordenar_si_son_numeros(valores) -> list:
    """Los grupos numericos (0, 1, 2...) se muestran en orden; los de
    texto, en el orden en que aparecen en la tabla."""
    if all(isinstance(v, (int, float)) and not isinstance(v, bool)
           for v in valores):
        return sorted(valores)
    return valores


def _bins_sturges(n: int) -> int:
    """
    Numero de intervalos del histograma con la regla de Sturges:
    1 + log2(n). log2(n) redondeado hacia abajo es la cantidad de bits
    de n menos uno, asi que no hace falta el modulo math.
    """
    return max(1, n.bit_length())


# =====================================================================
# CADA TIPO DE GRAFICA
# =====================================================================

def _barras(tabla, opciones, ruta):
    columna_x = _exigir(opciones, "eje_x", "barras")
    columna_y = _exigir(opciones, "eje_y", "barras")
    categorias = _columna(tabla, columna_x)
    valores = _columna_numerica(tabla, columna_y)

    totales = {}
    orden = []
    for categoria, valor in zip(categorias, valores):
        if valor is None:
            continue
        if categoria not in totales:
            totales[categoria] = 0
            orden.append(categoria)
        totales[categoria] += valor
    nombres = [str(c) for c in orden]
    alturas = [totales[c] for c in orden]

    titulo = opciones.get("titulo", "")
    if ruta is None:
        plt.grafica_barras_ascii(nombres, alturas, titulo, columna_y)
    else:
        graficas.grafica_barras(nombres, alturas, ruta, titulo, columna_x,
                                columna_y, opciones.get("leyenda"))


def _lineas(tabla, opciones, ruta):
    columna_x = _exigir(opciones, "eje_x", "lineas")
    columna_y = _exigir(opciones, "eje_y", "lineas")
    xs = _columna_numerica(tabla, columna_x)
    ys = _columna_numerica(tabla, columna_y)
    titulo = opciones.get("titulo", "")
    grupos = _grupos_por_leyenda(tabla, opciones)

    if grupos is None:
        _sin_x_repetidos(xs, columna_x)
        series = [ys]
        nombres = [opciones["leyenda"]] if opciones.get("leyenda") else None
    else:
        # una serie por grupo: en las filas de otros grupos va nulo
        nombres = _en_orden(grupos)
        series = [[y if g == nombre else None for y, g in zip(ys, grupos)]
                  for nombre in nombres]
        nombres = [str(n) for n in nombres]

    if ruta is None:
        pares = sorted((x, y) for x, y in zip(xs, series[0])
                       if x is not None and y is not None)
        plt.grafica_lineas_ascii([p[0] for p in pares],
                                 [p[1] for p in pares],
                                 titulo, columna_x, columna_y)
    else:
        graficas.grafica_lineas(xs, series, ruta, titulo, columna_x,
                                columna_y, nombres)


def _sin_x_repetidos(xs, columna_x):
    """
    Una linea une un punto por cada x. Si un x se repite (por ejemplo,
    el mismo anio para varios paises), la linea iria y volveria en zigzag
    y la grafica enganaria: se pide separar las series con 'leyenda'.
    """
    vistos = set()
    for x in xs:
        if x is None:
            continue
        if x in vistos:
            raise ErrorDatos(
                f"el eje x '{columna_x}' tiene valores repetidos ({x} "
                f"aparece mas de una vez): usa leyenda \"<columna>\" para "
                f"dibujar una linea por grupo, o agrupa los datos antes")
        vistos.add(x)


def _dispersion(tabla, opciones, ruta):
    columna_x = _exigir(opciones, "eje_x", "dispersion")
    columna_y = _exigir(opciones, "eje_y", "dispersion")
    xs = _columna_numerica(tabla, columna_x)
    ys = _columna_numerica(tabla, columna_y)
    titulo = opciones.get("titulo", "")
    grupos = _grupos_por_leyenda(tabla, opciones)

    if ruta is None:
        pares = [(x, y) for x, y in zip(xs, ys)
                 if x is not None and y is not None]
        plt.grafica_dispersion_ascii([p[0] for p in pares],
                                     [p[1] for p in pares],
                                     titulo, columna_x, columna_y)
    elif grupos is None:
        leyenda = opciones.get("leyenda")
        graficas.grafica_dispersion(xs, ys, ruta, titulo, columna_x,
                                    columna_y,
                                    leyendas=[leyenda] if leyenda else None)
    else:
        graficas.grafica_dispersion(xs, ys, ruta, titulo, columna_x,
                                    columna_y,
                                    grupos=[str(g) for g in grupos])


def _histograma(tabla, opciones, ruta):
    columna = opciones.get("eje_x") or opciones.get("eje_y")
    if not columna:
        raise ErrorDatos("la grafica de histograma necesita la opcion "
                         "'eje_x' con la columna a repartir")
    valores = _columna_numerica(tabla, columna)
    presentes = [v for v in valores if v is not None]
    bins = _bins_sturges(len(presentes))
    titulo = opciones.get("titulo", "")
    if ruta is None:
        plt.grafica_histograma_ascii(presentes, bins, titulo, columna)
    else:
        graficas.grafica_histograma(valores, ruta, bins, titulo, columna,
                                    "frecuencia", opciones.get("leyenda"))


def _caja(tabla, opciones, ruta):
    columna_y = _exigir(opciones, "eje_y", "caja")
    valores = _columna_numerica(tabla, columna_y)
    titulo = opciones.get("titulo", "")

    columna_grupo = opciones.get("eje_x")
    grupos = _columna(tabla, columna_grupo) if columna_grupo else \
        _grupos_por_leyenda(tabla, opciones)

    if grupos is None:
        nombres = [columna_y]
        listas = [valores]
    else:
        nombres = _ordenar_si_son_numeros(_en_orden(grupos))
        listas = [[v for v, g in zip(valores, grupos) if g == nombre]
                  for nombre in nombres]
        nombres = [str(n) for n in nombres]

    if ruta is None:
        for nombre, lista in zip(nombres, listas):
            plt.grafica_caja_ascii(lista, f"{titulo} - {nombre}".strip(" -"),
                                   columna_y)
    else:
        graficas.grafica_caja(listas, nombres, ruta, titulo,
                              columna_grupo or "", columna_y)
