"""
terra_plot.py
=============
Motor de graficas para Terra. Implementado desde cero.

Genera dos formatos:
    1. ASCII  — imprime la grafica en la terminal con caracteres de texto.
    2. SVG    — exporta un archivo SVG valido que se puede abrir en cualquier
                navegador o visor de imagenes.

Tipos de grafica soportados:
    - lineas      : serie temporal o relacion x-y
    - barras      : barras verticales por categoria
    - dispersion  : nube de puntos (scatter)
    - histograma  : distribucion de frecuencias
    - caja        : diagrama de caja y bigotes (box plot)
"""

from src.tabla import ErrorDatos


# =====================================================================
# CONFIGURACION
# =====================================================================

ANCHO_ASCII  = 60    # columnas de la grafica ASCII
ALTO_ASCII   = 20    # filas de la grafica ASCII
ANCHO_SVG    = 600   # pixeles
ALTO_SVG     = 400   # pixeles
MARGEN       = 60    # margen en pixeles para ejes


# =====================================================================
# UTILIDADES
# =====================================================================

def _rango(valores: list) -> tuple:
    no_nulos = [v for v in valores if v is not None]
    if not no_nulos:
        return 0.0, 1.0
    mn, mx = min(no_nulos), max(no_nulos)
    if mn == mx:
        mn -= 1; mx += 1
    return float(mn), float(mx)


def _escalar_ascii(valor, minv, maxv, pasos) -> int:
    if maxv == minv:
        return 0
    return int((valor - minv) / (maxv - minv) * (pasos - 1))


def _escalar_svg(valor, minv, maxv, largo, invertir=False) -> float:
    if maxv == minv:
        return largo / 2
    t = (valor - minv) / (maxv - minv)
    return largo * (1 - t) if invertir else largo * t


def _etiqueta(valor) -> str:
    if isinstance(valor, float):
        return f"{valor:.2f}"
    return str(valor)


# =====================================================================
# GRAFICAS ASCII
# =====================================================================

def _lienzo_ascii(alto, ancho) -> list:
    return [[" "] * ancho for _ in range(alto)]


def _dibujar_ejes(lienzo, alto, ancho):
    for i in range(alto):
        lienzo[i][0] = "│"
    for j in range(ancho):
        lienzo[alto-1][j] = "─"
    lienzo[alto-1][0] = "└"


def _imprimir_lienzo(lienzo, titulo=""):
    if titulo:
        print(f"\n  {titulo}")
        print("  " + "─" * len(titulo))
    for fila in lienzo:
        print("  " + "".join(fila))


def grafica_lineas_ascii(xs: list, ys: list, titulo: str = "",
                         etiqueta_x: str = "x", etiqueta_y: str = "y"):
    if len(xs) != len(ys):
        raise ErrorDatos("grafica_lineas(): xs e ys deben tener la misma longitud")

    ancho = ANCHO_ASCII
    alto  = ALTO_ASCII
    lienzo = _lienzo_ascii(alto, ancho)
    _dibujar_ejes(lienzo, alto, ancho)

    min_y, max_y = _rango(ys)
    n = len(xs)

    puntos = []
    for i in range(n):
        if ys[i] is None:
            continue
        col = int(i / max(n - 1, 1) * (ancho - 2)) + 1
        fil = alto - 2 - _escalar_ascii(ys[i], min_y, max_y, alto - 2)
        fil = max(0, min(alto - 2, fil))
        puntos.append((fil, col))

    # Trazar linea entre puntos consecutivos
    for k in range(len(puntos) - 1):
        f1, c1 = puntos[k]
        f2, c2 = puntos[k+1]
        pasos  = max(abs(f2-f1), abs(c2-c1), 1)
        for s in range(pasos + 1):
            ff = f1 + int((f2-f1) * s / pasos)
            cc = c1 + int((c2-c1) * s / pasos)
            if 0 <= ff < alto-1 and 0 < cc < ancho:
                lienzo[ff][cc] = "·"

    for f, c in puntos:
        if 0 <= f < alto-1 and 0 < c < ancho:
            lienzo[f][c] = "●"

    # Etiquetas de ejes
    lienzo[0][1] = _etiqueta(max_y)[0]
    lienzo[alto-2][1] = _etiqueta(min_y)[0]

    _imprimir_lienzo(lienzo, titulo)
    print(f"  {etiqueta_x}: {_etiqueta(min(xs))} → {_etiqueta(max(xs))}")
    print(f"  {etiqueta_y}: {_etiqueta(min_y)} → {_etiqueta(max_y)}")


def grafica_barras_ascii(categorias: list, valores: list,
                         titulo: str = "", etiqueta_y: str = "valor"):
    if len(categorias) != len(valores):
        raise ErrorDatos("grafica_barras(): categorias y valores deben tener la misma longitud")

    ancho  = ANCHO_ASCII
    alto   = ALTO_ASCII
    lienzo = _lienzo_ascii(alto, ancho)
    _dibujar_ejes(lienzo, alto, ancho)

    min_v, max_v = 0.0, _rango(valores)[1]
    n     = len(categorias)
    ancho_barra = max(1, (ancho - 2) // n)

    for i, (cat, val) in enumerate(zip(categorias, valores)):
        if val is None:
            continue
        col_ini = 1 + i * ancho_barra
        altura  = _escalar_ascii(val, min_v, max_v, alto - 2)
        for h in range(altura):
            fil = alto - 2 - h
            for dc in range(ancho_barra - 1):
                c = col_ini + dc
                if 0 < c < ancho:
                    lienzo[fil][c] = "█"
        # Etiqueta categoria
        etiq = str(cat)[:ancho_barra-1]
        for k, ch in enumerate(etiq):
            c = col_ini + k
            if 0 < c < ancho:
                lienzo[alto-1][c] = ch

    _imprimir_lienzo(lienzo, titulo)
    print(f"  {etiqueta_y}: 0 → {_etiqueta(max_v)}")


def grafica_dispersion_ascii(xs: list, ys: list,
                             titulo: str = "",
                             etiqueta_x: str = "x",
                             etiqueta_y: str = "y"):
    ancho  = ANCHO_ASCII
    alto   = ALTO_ASCII
    lienzo = _lienzo_ascii(alto, ancho)
    _dibujar_ejes(lienzo, alto, ancho)

    min_x, max_x = _rango(xs)
    min_y, max_y = _rango(ys)

    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        col = _escalar_ascii(x, min_x, max_x, ancho - 1) + 1
        fil = alto - 2 - _escalar_ascii(y, min_y, max_y, alto - 2)
        fil = max(0, min(alto-2, fil))
        col = max(1, min(ancho-1, col))
        lienzo[fil][col] = "×"

    _imprimir_lienzo(lienzo, titulo)
    print(f"  {etiqueta_x}: {_etiqueta(min_x)} → {_etiqueta(max_x)}")
    print(f"  {etiqueta_y}: {_etiqueta(min_y)} → {_etiqueta(max_y)}")


def grafica_histograma_ascii(valores: list, bins: int = 10,
                             titulo: str = "",
                             etiqueta_x: str = "valor"):
    no_nulos = [v for v in valores if v is not None]
    if not no_nulos:
        raise ErrorDatos("histograma(): no hay valores no nulos")

    min_v, max_v = _rango(no_nulos)
    paso  = (max_v - min_v) / bins
    conteo = [0] * bins
    for v in no_nulos:
        idx = min(int((v - min_v) / paso), bins - 1)
        conteo[idx] += 1

    categorias = [f"{min_v + i*paso:.1f}" for i in range(bins)]
    grafica_barras_ascii(categorias, conteo, titulo, "frecuencia")
    print(f"  {etiqueta_x}: {_etiqueta(min_v)} → {_etiqueta(max_v)}  |  bins={bins}")


def grafica_caja_ascii(valores: list, titulo: str = "",
                       etiqueta: str = "valor"):
    no_nulos = sorted(v for v in valores if v is not None)
    if len(no_nulos) < 4:
        raise ErrorDatos("grafica_caja(): se necesitan al menos 4 valores")

    def cuartil(datos, q):
        idx = (len(datos) - 1) * q
        lo  = int(idx)
        hi  = lo + 1
        if hi >= len(datos):
            return datos[lo]
        return datos[lo] + (datos[hi] - datos[lo]) * (idx - lo)

    q1  = cuartil(no_nulos, 0.25)
    med = cuartil(no_nulos, 0.50)
    q3  = cuartil(no_nulos, 0.75)
    iqr = q3 - q1
    bmin = max(min(no_nulos), q1 - 1.5 * iqr)
    bmax = min(max(no_nulos), q3 + 1.5 * iqr)

    ancho = ANCHO_ASCII
    min_v, max_v = _rango(no_nulos)

    def pos(v):
        return _escalar_ascii(v, min_v, max_v, ancho - 2) + 1

    linea = [" "] * ancho
    # Bigotes
    p_bmin = pos(bmin); p_q1 = pos(q1)
    p_med  = pos(med);  p_q3 = pos(q3); p_bmax = pos(bmax)

    for j in range(p_bmin, p_bmax + 1):
        if 0 < j < ancho:
            linea[j] = "─"
    for j in range(p_q1, p_q3 + 1):
        if 0 < j < ancho:
            linea[j] = "█"
    for p, ch in [(p_bmin, "├"), (p_q1, "│"), (p_med, "◆"),
                  (p_q3, "│"), (p_bmax, "┤")]:
        if 0 < p < ancho:
            linea[p] = ch

    if titulo:
        print(f"\n  {titulo}")
    print("  " + "".join(linea))
    print(f"  min={_etiqueta(bmin)}  Q1={_etiqueta(q1)}  "
          f"med={_etiqueta(med)}  Q3={_etiqueta(q3)}  "
          f"max={_etiqueta(bmax)}")


# =====================================================================
# EXPORTACION SVG
# =====================================================================

def _svg_header(ancho, alto, titulo) -> list:
    lineas = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{ancho}" height="{alto}" '
        f'style="background:#fff;font-family:monospace">',
    ]
    if titulo:
        lineas.append(
            f'  <text x="{ancho//2}" y="20" text-anchor="middle" '
            f'font-size="14" font-weight="bold">{titulo}</text>')
    return lineas


def _svg_ejes(ancho, alto, margen, min_x, max_x, min_y, max_y,
              etiq_x, etiq_y) -> list:
    ax = margen; ay = margen
    bx = ancho - margen // 2; by = alto - margen

    lineas = [
        f'  <!-- ejes -->',
        f'  <line x1="{ax}" y1="{ay}" x2="{ax}" y2="{by}" '
        f'stroke="#333" stroke-width="1.5"/>',
        f'  <line x1="{ax}" y1="{by}" x2="{bx}" y2="{by}" '
        f'stroke="#333" stroke-width="1.5"/>',
        f'  <text x="{ax-5}" y="{ay}" text-anchor="end" '
        f'font-size="10">{_etiqueta(max_y)}</text>',
        f'  <text x="{ax-5}" y="{by}" text-anchor="end" '
        f'font-size="10">{_etiqueta(min_y)}</text>',
        f'  <text x="{ax}" y="{by+15}" font-size="10">{_etiqueta(min_x)}</text>',
        f'  <text x="{bx}" y="{by+15}" text-anchor="end" '
        f'font-size="10">{_etiqueta(max_x)}</text>',
        f'  <text x="{(ax+bx)//2}" y="{alto-5}" text-anchor="middle" '
        f'font-size="11">{etiq_x}</text>',
        f'  <text x="12" y="{(ay+by)//2}" text-anchor="middle" '
        f'font-size="11" transform="rotate(-90,12,{(ay+by)//2})">{etiq_y}</text>',
    ]
    return lineas


def _px(valor, minv, maxv, px_min, px_max, invertir=False) -> float:
    if maxv == minv:
        return (px_min + px_max) / 2
    t = (valor - minv) / (maxv - minv)
    return px_max - t * (px_max - px_min) if invertir \
        else px_min + t * (px_max - px_min)


COLORES = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
           "#9C27B0", "#00BCD4", "#795548", "#607D8B"]


def exportar_lineas_svg(xs: list, ys_lista: list, etiquetas: list,
                        titulo: str, etiq_x: str, etiq_y: str,
                        ruta: str):
    """
    Exporta una grafica de lineas a SVG.
    ys_lista es una lista de listas (una por serie).
    """
    if not isinstance(ys_lista[0], list):
        ys_lista = [ys_lista]
        etiquetas = etiquetas or ["serie"]

    ancho, alto, m = ANCHO_SVG, ALTO_SVG, MARGEN
    ax, bx = m, ancho - m // 2
    ay, by = m + 10, alto - m

    todos_y = [v for ys in ys_lista for v in ys if v is not None]
    min_x, max_x = _rango(xs)
    min_y, max_y = _rango(todos_y)

    lineas = _svg_header(ancho, alto, titulo)
    lineas += _svg_ejes(ancho, alto, m, min_x, max_x, min_y, max_y,
                        etiq_x, etiq_y)

    for idx, ys in enumerate(ys_lista):
        color = COLORES[idx % len(COLORES)]
        puntos_svg = []
        for x, y in zip(xs, ys):
            if y is None:
                continue
            px = _px(x, min_x, max_x, ax, bx)
            py = _px(y, min_y, max_y, ay, by, invertir=True)
            puntos_svg.append(f"{px:.1f},{py:.1f}")

        if puntos_svg:
            lineas.append(
                f'  <polyline points="{" ".join(puntos_svg)}" '
                f'fill="none" stroke="{color}" stroke-width="2"/>')
            # Puntos
            for pt in puntos_svg:
                px_s, py_s = pt.split(",")
                lineas.append(
                    f'  <circle cx="{px_s}" cy="{py_s}" r="3" fill="{color}"/>')

        # Leyenda
        ley_x = bx - 120
        ley_y = ay + idx * 18
        etiq  = etiquetas[idx] if idx < len(etiquetas) else f"serie{idx+1}"
        lineas.append(
            f'  <line x1="{ley_x}" y1="{ley_y}" x2="{ley_x+20}" '
            f'y2="{ley_y}" stroke="{color}" stroke-width="2"/>')
        lineas.append(
            f'  <text x="{ley_x+24}" y="{ley_y+4}" font-size="11">{etiq}</text>')

    lineas.append("</svg>")
    _escribir_svg(ruta, lineas)


def exportar_barras_svg(categorias: list, valores: list,
                        titulo: str, etiq_x: str, etiq_y: str,
                        ruta: str):
    ancho, alto, m = ANCHO_SVG, ALTO_SVG, MARGEN
    ax, bx = m, ancho - m // 2
    ay, by = m + 10, alto - m

    min_v, max_v = 0.0, _rango(valores)[1]
    n            = len(categorias)
    ancho_barra  = (bx - ax) / n * 0.7
    espacio      = (bx - ax) / n

    lineas = _svg_header(ancho, alto, titulo)
    lineas += _svg_ejes(ancho, alto, m, 0, n - 1, min_v, max_v,
                        etiq_x, etiq_y)

    for i, (cat, val) in enumerate(zip(categorias, valores)):
        if val is None:
            continue
        color = COLORES[i % len(COLORES)]
        cx    = ax + i * espacio + espacio * 0.15
        py    = _px(val, min_v, max_v, ay, by, invertir=True)
        altura = by - py
        lineas.append(
            f'  <rect x="{cx:.1f}" y="{py:.1f}" '
            f'width="{ancho_barra:.1f}" height="{altura:.1f}" '
            f'fill="{color}" opacity="0.8"/>')
        # Etiqueta
        etiq = str(cat)[:8]
        lineas.append(
            f'  <text x="{cx + ancho_barra/2:.1f}" y="{by+14}" '
            f'text-anchor="middle" font-size="9">{etiq}</text>')
        lineas.append(
            f'  <text x="{cx + ancho_barra/2:.1f}" y="{py-3:.1f}" '
            f'text-anchor="middle" font-size="9">{_etiqueta(val)}</text>')

    lineas.append("</svg>")
    _escribir_svg(ruta, lineas)


def exportar_dispersion_svg(xs: list, ys: list,
                            titulo: str, etiq_x: str, etiq_y: str,
                            ruta: str, color: str = "#2196F3"):
    ancho, alto, m = ANCHO_SVG, ALTO_SVG, MARGEN
    ax, bx = m, ancho - m // 2
    ay, by = m + 10, alto - m

    min_x, max_x = _rango(xs)
    min_y, max_y = _rango(ys)

    lineas = _svg_header(ancho, alto, titulo)
    lineas += _svg_ejes(ancho, alto, m, min_x, max_x, min_y, max_y,
                        etiq_x, etiq_y)

    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        px = _px(x, min_x, max_x, ax, bx)
        py = _px(y, min_y, max_y, ay, by, invertir=True)
        lineas.append(
            f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="4" '
            f'fill="{color}" opacity="0.7"/>')

    lineas.append("</svg>")
    _escribir_svg(ruta, lineas)


def exportar_histograma_svg(valores: list, bins: int,
                            titulo: str, etiq_x: str, ruta: str):
    no_nulos = [v for v in valores if v is not None]
    if not no_nulos:
        raise ErrorDatos("histograma_svg(): no hay valores no nulos")

    min_v, max_v = _rango(no_nulos)
    paso  = (max_v - min_v) / bins
    conteo = [0] * bins
    for v in no_nulos:
        idx = min(int((v - min_v) / paso), bins - 1)
        conteo[idx] += 1

    categorias = [min_v + i * paso for i in range(bins)]
    exportar_barras_svg(categorias, conteo, titulo, etiq_x, "frecuencia", ruta)


def _escribir_svg(ruta: str, lineas: list):
    try:
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas) + "\n")
    except (FileNotFoundError, PermissionError) as e:
        raise ErrorDatos(f"no se pudo guardar el SVG en '{ruta}': {e}")