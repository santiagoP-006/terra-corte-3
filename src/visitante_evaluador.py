"""
visitante_evaluador.py
======================
Evaluador semantico de Terra: implementa el patron Visitor sobre el
arbol de analisis generado por ANTLR.

Responsabilidades de este modulo (nucleo del lenguaje):
    - Memoria de variables con inmutabilidad (fijo) y mutabilidad (var)
    - Tipado dinamico y fuerte: ninguna conversion implicita entre tipos
    - Validacion de tipos en todos los operadores
    - Estructuras de control: si / sino si / sino, mientras, para ... en
    - Funciones incorporadas de matematicas y estadistica
    - Salida con mostrar()

Las operaciones de datos (cargar, tuberias, guardar, graficar) se
reconocen sintacticamente pero su evaluacion corresponde a los cortes
siguientes: aqui producen un error semantico explicito.

Tipos de Terra y su representacion en Python:
    numero    -> int
    decimal   -> float
    texto     -> str
    booleano  -> bool
    nulo      -> None
    lista     -> list
"""

import sys
import os
from src import regresion as reg
from src import arbol_decision as ad
from src import terra_pandas as pd
from src import terra_numpy  as np
from src import terra_plot   as plt
from src.terra_pandas import DataFrame
from src.terra_numpy  import Array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gramaticas.TerraParser import TerraParser
from gramaticas.TerraVisitor import TerraVisitor
from src.tabla import Tabla, ErrorDatos, leer_csv, escribir_csv
from src import operaciones as ops
from src import biblioteca

class RetornoSenal(Exception):
    """
    Senal interna que transporta el valor de 'retornar' hasta la llamada.

    No es un error: se usa como mecanismo de salto. Cuando una funcion
    ejecuta 'retornar', esta excepcion sube por la pila de visitas
    atravesando ciclos y condicionales, y la captura _invocar().
    """

    def __init__(self, valor):
        self.valor = valor
        super().__init__("retorno de funcion")

class ErrorSemantico(Exception):
    """Error detectado durante la evaluacion del arbol."""

    def __init__(self, mensaje, linea=None):
        self.linea = linea
        if linea is not None:
            mensaje = f"[linea {linea}] {mensaje}"
        super().__init__(mensaje)


# =====================================================================
# UTILIDADES DE TIPO
# =====================================================================

def nombre_tipo(valor) -> str:
    """
    Devuelve el nombre del tipo Terra de un valor de Python.

    El orden importa: bool va antes que int (en Python bool es subclase
    de int) y DataFrame va antes que Tabla (DataFrame hereda de Tabla).
    """
    if isinstance(valor, bool):
        return "booleano"
    if isinstance(valor, int):
        return "numero"
    if isinstance(valor, float):
        return "decimal"
    if isinstance(valor, str):
        return "texto"
    if isinstance(valor, list):
        return "lista"
    if valor is None:
        return "nulo"
    if isinstance(valor, DataFrame):
        return "dataframe"
    if isinstance(valor, Tabla):
        return "tabla"
    if isinstance(valor, Array):
        return "array"
    if isinstance(valor, dict) and "tipo" in valor:
        return "modelo"
    if isinstance(valor, dict) and "exactitud" in valor:
        return "evaluacion"
    if biblioteca.es_funcion(valor):
        return "funcion"
    return "desconocido"


def es_numero(valor) -> bool:
    """
    True si el valor es numerico.

    La comprobacion de bool va primero porque en Python bool es
    subclase de int: sin ella, verdadero + 1 seria valido y Terra
    perderia el tipado fuerte.
    """
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def formatear(valor, anidado=False) -> str:
    """
    Convierte un valor a su representacion textual en Terra.

    Cada tipo tiene su propia forma de mostrarse. Antes habia codigo
    despues de un 'return' que nunca se ejecutaba; por eso los arrays
    salian como 'Array(2, 2)' y los modelos como diccionarios de Python.
    """
    if valor is None:
        return "nulo"
    if isinstance(valor, bool):
        return "verdadero" if valor else "falso"
    if isinstance(valor, list):
        return "[" + ", ".join(formatear(v, True) for v in valor) + "]"
    if isinstance(valor, str):
        return f'"{valor}"' if anidado else valor
    if isinstance(valor, Tabla):          # incluye DataFrame
        return valor.como_texto()
    if isinstance(valor, Array):
        return valor.como_texto()
    if isinstance(valor, dict) and "tipo" in valor:
        return _formatear_modelo(valor)
    if isinstance(valor, dict) and "exactitud" in valor:
        return (f"Evaluacion: exactitud={round(valor['exactitud'] * 100, 2)}%"
                f"  ({valor['correctas']}/{valor['total']})")
    return str(valor)

def _formatear_modelo(modelo: dict) -> str:
    """Representacion legible de un modelo de regresion."""
    if modelo["tipo"] == "simple":
        a = round(modelo["intercepto"], 6)
        b = round(modelo["pendiente"],  6)
        r2 = round(modelo["r_cuadrado"], 4)
        return (f"Modelo lineal simple (n={modelo['n']})\n"
                f"  y = {a} + {b} * x\n"
                f"  R^2 = {r2}")
    if modelo["tipo"] == "multiple":
        coefs = [round(c, 6) for c in modelo["coeficientes"]]
        r2    = round(modelo["r_cuadrado"], 4)
        terms = " + ".join(f"{coefs[i+1]} * x{i+1}"
                           for i in range(len(coefs) - 1))
        return (f"Modelo lineal multiple (n={modelo['n']}, "
                f"p={modelo['p']})\n"
                f"  y = {coefs[0]} + {terms}\n"
                f"  R^2 = {r2}")
    return str(modelo)

# =====================================================================
# VISITANTE
# =====================================================================

class VisitanteEvaluador(TerraVisitor):
    """
    Recorre el arbol de analisis y ejecuta el programa.

    Attributes:
        memoria (dict): nombre -> {"valor": ..., "mutable": bool}
        modo_interactivo (bool): si es True, las expresiones sueltas
            imprimen su resultado automaticamente.
    """

    LIMITE_RECURSION = 200

    def __init__(self, modo_interactivo=False):
        self.memoria = {}
        self.funciones = {}
        self.profundidad = 0
        self.modo_interactivo = modo_interactivo
    # -----------------------------------------------------------------
    # ESTRUCTURA
    # -----------------------------------------------------------------

    def visitPrograma(self, ctx: TerraParser.ProgramaContext):
        """Ejecuta cada sentencia del programa en orden."""
        for hijo in ctx.children or []:
            if isinstance(hijo, TerraParser.SentenciaContext):
                self.visit(hijo)
        return None

    def visitBloque(self, ctx: TerraParser.BloqueContext):
        """Ejecuta las sentencias de un bloque, ignorando lineas vacias."""
        for hijo in ctx.children or []:
            if isinstance(hijo, TerraParser.SentenciaContext):
                self.visit(hijo)
        return None

    def visitSentencia(self, ctx: TerraParser.SentenciaContext):
        return self.visit(ctx.getChild(0))

    # -----------------------------------------------------------------
    # VARIABLES
    # -----------------------------------------------------------------

    def visitDeclaracion_variable(self, ctx):
        """
        Declara una variable.

        'fijo' crea una variable inmutable y 'var' una mutable. Ninguna
        de las dos puede redeclararse: el scope es unico y explicito.
        """
        nombre = ctx.IDENTIFICADOR().getText()
        linea = ctx.start.line

        if nombre in self.memoria:
            raise ErrorSemantico(
                f"la variable '{nombre}' ya fue declarada", linea)

        valor = self.visit(ctx.expresion())
        self.memoria[nombre] = {
            "valor": valor,
            "mutable": ctx.VAR() is not None,
        }
        return None

    def visitAsignacion(self, ctx: TerraParser.AsignacionContext):
        """Reasigna una variable mutable ya declarada."""
        nombre = ctx.IDENTIFICADOR().getText()
        linea = ctx.start.line

        if nombre not in self.memoria:
            raise ErrorSemantico(
                f"la variable '{nombre}' no ha sido declarada", linea)
        if not self.memoria[nombre]["mutable"]:
            raise ErrorSemantico(
                f"'{nombre}' se declaro con 'fijo' y no puede reasignarse",
                linea)

        self.memoria[nombre]["valor"] = self.visit(ctx.expresion())
        return None

    def visitExpresion_stmt(self, ctx):
        """
        Evalua una expresion suelta.

        En modo interactivo imprime el resultado con el prefijo '>',
        como hace cualquier REPL.
        """
        valor = self.visit(ctx.expresion())
        if self.modo_interactivo and valor is not None:
            print(">", formatear(valor))
        return valor

    def visitMostrar_stmt(self, ctx: TerraParser.Mostrar_stmtContext):
        """Imprime uno o varios valores separados por espacio."""
        partes = [formatear(self.visit(e)) for e in ctx.expresion()]
        print(" ".join(partes))
        return None

    # -----------------------------------------------------------------
    # CONTROL DE FLUJO
    # -----------------------------------------------------------------

    def _condicion(self, ctx_expresion) -> bool:
        """Evalua una condicion y exige que sea booleana."""
        valor = self.visit(ctx_expresion)
        if not isinstance(valor, bool):
            raise ErrorSemantico(
                f"la condicion debe ser booleana, se recibio "
                f"{nombre_tipo(valor)}", ctx_expresion.start.line)
        return valor

    def visitSi_stmt(self, ctx: TerraParser.Si_stmtContext):
        """
        Ejecuta el primer bloque cuya condicion sea verdadera.

        Las condiciones (si y sino si) llegan en ctx.expresion() y los
        bloques en ctx.bloque(), alineados por indice. Si hay un bloque
        mas que condiciones, el ultimo corresponde al 'sino'.
        """
        condiciones = ctx.expresion()
        bloques = ctx.bloque()

        for i, condicion in enumerate(condiciones):
            if self._condicion(condicion):
                self.visit(bloques[i])
                return None

        if len(bloques) > len(condiciones):
            self.visit(bloques[-1])
        return None

    def visitMientras_stmt(self, ctx: TerraParser.Mientras_stmtContext):
        """Repite el bloque mientras la condicion sea verdadera."""
        limite = 1_000_000
        vueltas = 0
        while self._condicion(ctx.expresion()):
            self.visit(ctx.bloque())
            vueltas += 1
            if vueltas > limite:
                raise ErrorSemantico(
                    "el ciclo 'mientras' supero el limite de iteraciones; "
                    "revisa la condicion de salida", ctx.start.line)
        return None

    def visitPara_stmt(self, ctx: TerraParser.Para_stmtContext):
        """
        Recorre una lista o un texto.

        La variable del ciclo es local: se declara al entrar y se retira
        al terminar, sin sobrescribir una variable del mismo nombre.
        """
        nombre = ctx.IDENTIFICADOR().getText()
        linea = ctx.start.line
        iterable = self.visit(ctx.expresion())

        if not isinstance(iterable, (list, str)):
            raise ErrorSemantico(
                f"solo se puede recorrer una lista o un texto, se recibio "
                f"{nombre_tipo(iterable)}", linea)

        previo = self.memoria.get(nombre)
        for elemento in iterable:
            self.memoria[nombre] = {"valor": elemento, "mutable": True}
            self.visit(ctx.bloque())

        if previo is not None:
            self.memoria[nombre] = previo
        else:
            self.memoria.pop(nombre, None)
        return None

    # -----------------------------------------------------------------
    # DOMINIO: PENDIENTE PARA LOS CORTES SIGUIENTES
    # -----------------------------------------------------------------

    def visitGuardar_stmt(self, ctx: TerraParser.Guardar_stmtContext):
        """
        guardar tabla como "ruta.csv"
        Escribe una Tabla en un archivo CSV.
        """
        linea = ctx.start.line
        tabla = self.visit(ctx.expr(0))
        ruta = self.visit(ctx.expr(1))

        if not isinstance(tabla, Tabla):
            raise ErrorSemantico(
                f"'guardar' necesita una tabla, se recibio "
                f"{nombre_tipo(tabla)}", linea)
        if not isinstance(ruta, str):
            raise ErrorSemantico(
                f"la ruta de 'guardar' debe ser texto, se recibio "
                f"{nombre_tipo(ruta)}", linea)

        try:
            separador = ","
            if ctx.SEPARADOR():
                separador = self.visit(ctx.expr(2))
                if not isinstance(separador, str):
                    raise ErrorSemantico(
                        f"el separador debe ser texto, se recibio "
                        f"{nombre_tipo(separador)}", linea)
            escribir_csv(tabla, ruta, separador)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        return None

    def visitGraficar_stmt(self, ctx):
        raise ErrorSemantico(
            "'graficar' se reconoce sintacticamente pero su ejecucion "
            "corresponde al tercer corte", ctx.start.line)

    def visitCarga(self, ctx: TerraParser.CargaContext):
        """
        cargar "ruta.csv" [con separador ";"] [con encabezado falso]
        Lee un archivo CSV y devuelve una Tabla con tipos inferidos.
        Las opciones pueden ir en cualquier orden, pero sin repetirse.
        """
        linea = ctx.start.line
        ruta = self.visit(ctx.expr())
        if not isinstance(ruta, str):
            raise ErrorSemantico(
                f"la ruta de 'cargar' debe ser texto, se recibio "
                f"{nombre_tipo(ruta)}", linea)

        separador = ","
        encabezado = True
        usadas = set()

        for opcion in ctx.opcion_carga():
            nombre = "separador" if opcion.SEPARADOR() else "encabezado"
            if nombre in usadas:
                raise ErrorSemantico(
                    f"la opcion '{nombre}' aparece repetida en 'cargar'",
                    linea)
            usadas.add(nombre)
            valor = self.visit(opcion.expr())

            if nombre == "separador":
                if not isinstance(valor, str):
                    raise ErrorSemantico(
                        f"el separador debe ser texto, se recibio "
                        f"{nombre_tipo(valor)}", linea)
                separador = valor
            else:
                if not isinstance(valor, bool):
                    raise ErrorSemantico(
                        f"la opcion 'encabezado' debe ser verdadero o "
                        f"falso, se recibio {nombre_tipo(valor)}", linea)
                encabezado = valor

        try:
            return leer_csv(ruta, separador, encabezado)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
    # -----------------------------------------------------------------
    # EXPRESIONES
    # -----------------------------------------------------------------

    def visitExpresion(self, ctx: TerraParser.ExpresionContext):
        return self.visit(ctx.tuberia())

    def visitTuberia(self, ctx: TerraParser.TuberiaContext):
        """
        Sin '|>' es una expresion normal.
        Con '|>' el valor inicial debe ser una tabla; cada operacion
        recibe la tabla que produjo la anterior y devuelve una nueva.
        """
        valor = self.visit(ctx.expr())
        operaciones = ctx.operacion()
        if not operaciones:
            return valor

        if not isinstance(valor, Tabla):
            raise ErrorSemantico(
                f"una tuberia '|>' debe empezar con una tabla, se recibio "
                f"{nombre_tipo(valor)}", ctx.start.line)

        for operacion in operaciones:
            valor = self._aplicar_operacion(valor, operacion)
        if isinstance(valor, ops.Agrupada):
            raise ErrorSemantico(
                "la tuberia termina en 'agrupar': falta 'resumir'",
                ctx.start.line)
        return valor

    def _columnas(self, ctx_lista):
        """Nombres de una lista_columnas: [a, b, c] -> ['a', 'b', 'c']."""
        return [t.getText() for t in ctx_lista.IDENTIFICADOR()]

    def _nombres_usados(self, ctx_expr):
        """
        Conjunto de identificadores que aparecen en una expresion.

        Recorre el subarbol una sola vez y guarda el resultado, para
        que filtrar, crear y resumir declaren solo las columnas que la
        expresion realmente usa, y no todas las de la tabla.
        """
        if not hasattr(self, "_cache_nombres"):
            self._cache_nombres = {}
        if ctx_expr not in self._cache_nombres:
            nombres = set()
            pendientes = [ctx_expr]
            while pendientes:
                nodo = pendientes.pop()
                if hasattr(nodo, "symbol"):
                    if nodo.symbol.type == TerraParser.IDENTIFICADOR:
                        nombres.add(nodo.getText())
                else:
                    pendientes.extend(nodo.getChildren())
            self._cache_nombres[ctx_expr] = nombres
        return self._cache_nombres[ctx_expr]
    def _evaluar_en_fila(self, tabla, i, ctx_expr):
        """
        Evalua una expresion usando la fila i de la tabla.

        Las columnas de la fila se declaran como variables temporales
        (no reasignables) y al terminar se restaura la memoria tal como
        estaba. Asi 'filtrar donde co2 > 50' puede usar 'co2' como si
        fuera una variable.
        """
        respaldo = {}
        usados = self._nombres_usados(ctx_expr)
        for nombre, valor in zip(tabla.columnas, tabla.filas[i]):
            if nombre not in usados:
                continue
            respaldo[nombre] = self.memoria.get(nombre)
            self.memoria[nombre] = {"valor": valor, "mutable": False}
        try:
            return self.visit(ctx_expr)
        finally:
            for nombre, previo in respaldo.items():
                if previo is None:
                    self.memoria.pop(nombre, None)
                else:
                    self.memoria[nombre] = previo

    def _evaluar_en_grupo(self, tabla, indices, ctx_expr):
        """
        Evalua una agregacion sobre un grupo de filas.

        Cada columna se declara como la LISTA de sus valores en el
        grupo, sin nulos. Asi suma(co2) o media(co2) reutilizan las
        funciones incorporadas de Terra. contar() devuelve el numero
        de filas del grupo. Al terminar se restaura la memoria.
        """
        respaldo = {}
        usados = self._nombres_usados(ctx_expr)
        for j, nombre in enumerate(tabla.columnas):
            if nombre not in usados:
                continue
            valores = [tabla.filas[i][j] for i in indices
                       if tabla.filas[i][j] is not None]
            respaldo[nombre] = self.memoria.get(nombre)
            self.memoria[nombre] = {"valor": valores, "mutable": False}

        tamano_previo = getattr(self, "_tamano_grupo", None)
        self._tamano_grupo = len(indices)
        try:
            return self.visit(ctx_expr)
        finally:
            self._tamano_grupo = tamano_previo
            for nombre, previo in respaldo.items():
                if previo is None:
                    self.memoria.pop(nombre, None)
                else:
                    self.memoria[nombre] = previo

    def _resumir(self, origen, agregaciones, linea):
        """
        resumir nombre = expresion, ...
        Con 'agrupar' antes: una fila por grupo, con las columnas clave.
        Sin 'agrupar': una sola fila que resume toda la tabla.
        """
        if isinstance(origen, ops.Agrupada):
            tabla, claves, grupos = origen.tabla, origen.claves, origen.grupos
        else:
            tabla, claves = origen, []
            grupos = [((), list(range(origen.num_filas)))]

        nombres = [a.IDENTIFICADOR().getText() for a in agregaciones]
        filas = []
        for clave, indices in grupos:
            fila = list(clave)
            for nombre, agregacion in zip(nombres, agregaciones):
                valor = self._evaluar_en_grupo(tabla, indices,
                                               agregacion.expr())
                if (valor is not None
                        and not isinstance(valor, ops.TIPOS_CELDA)):
                    raise ErrorSemantico(
                        f"la agregacion '{nombre}' debe producir un solo "
                        f"valor (por ejemplo con suma, media o contar), "
                        f"se obtuvo {nombre_tipo(valor)}",
                        agregacion.start.line)
                fila.append(valor)
            filas.append(fila)

        return Tabla(claves + nombres, filas)
    def _aplicar_operacion(self, tabla, op):
        """Ejecuta una operacion de la tuberia sobre una tabla."""
        linea = op.start.line
        if isinstance(tabla, ops.Agrupada) and not op.RESUMIR():
            raise ErrorSemantico(
                "despues de 'agrupar' debe venir 'resumir'", linea)
        try:
            if op.SELECCIONAR():
                return ops.seleccionar(
                    tabla, self._columnas(op.lista_columnas()))

            if op.FILTRAR():
                def condicion(i):
                    resultado = self._evaluar_en_fila(tabla, i, op.expr())
                    if not isinstance(resultado, bool):
                        raise ErrorSemantico(
                            f"la condicion de 'filtrar' debe ser booleana, "
                            f"se obtuvo {nombre_tipo(resultado)}", linea)
                    return resultado
                return ops.filtrar(tabla, condicion)

            if op.CREAR():
                nombre = op.IDENTIFICADOR(0).getText()
                return ops.crear(
                    tabla, nombre,
                    lambda i: self._evaluar_en_fila(tabla, i, op.expr()))

            if op.RENOMBRAR():
                return ops.renombrar(tabla,
                                     op.IDENTIFICADOR(0).getText(),
                                     op.IDENTIFICADOR(1).getText())

            if op.ORDENAR():
                return ops.ordenar(tabla, op.IDENTIFICADOR(0).getText(),
                                   descendente=op.DESC() is not None)

            if op.ELIMINAR() and op.DUPLICADOS():
                return ops.eliminar_duplicados(tabla)

            if op.ELIMINAR() and op.NULOS():
                columnas = (self._columnas(op.lista_columnas())
                            if op.lista_columnas() else None)
                return ops.eliminar_nulos(tabla, columnas)

            if op.RELLENAR():
                valor = self.visit(op.expr())
                return ops.rellenar_nulos(
                    tabla, op.IDENTIFICADOR(0).getText(), valor)

            if op.CONVERTIR():
                return ops.convertir(tabla, op.IDENTIFICADOR(0).getText(),
                                     op.tipo_dato().getText())

            if op.LIMITAR():
                n = self.visit(op.expr())
                if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                    raise ErrorSemantico(
                        "'limitar' necesita un numero entero mayor o "
                        "igual a 0", linea)
                return ops.limitar(tabla, n)

            if op.AGRUPAR():
                return ops.agrupar(
                    tabla, self._columnas(op.lista_columnas()))

            if op.RESUMIR():
                return self._resumir(tabla, op.agregacion(), linea)
            raise ErrorSemantico(
                f"operacion desconocida '{op.getText()}'", linea)

        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
    def visitExpr(self, ctx: TerraParser.ExprContext):
        return self.visit(ctx.o_logico())

    def visitO_logico(self, ctx: TerraParser.O_logicoContext):
        """Disyuncion logica. Ambos operandos deben ser booleanos."""
        operandos = ctx.y_logico()
        resultado = self.visit(operandos[0])
        if len(operandos) == 1:
            return resultado

        self._exigir_booleano(resultado, "||", ctx.start.line)
        for i in range(1, len(operandos)):
            derecho = self.visit(operandos[i])
            self._exigir_booleano(derecho, "||", ctx.start.line)
            resultado = resultado or derecho
        return resultado

    def visitY_logico(self, ctx: TerraParser.Y_logicoContext):
        """Conjuncion logica. Ambos operandos deben ser booleanos."""
        operandos = ctx.igualdad()
        resultado = self.visit(operandos[0])
        if len(operandos) == 1:
            return resultado

        self._exigir_booleano(resultado, "&&", ctx.start.line)
        for i in range(1, len(operandos)):
            derecho = self.visit(operandos[i])
            self._exigir_booleano(derecho, "&&", ctx.start.line)
            resultado = resultado and derecho
        return resultado

    def visitIgualdad(self, ctx: TerraParser.IgualdadContext):
        """
        Igualdad y desigualdad.

        Exige que ambos lados sean del mismo tipo, con una excepcion:
        numero y decimal se comparan entre si porque ambos son valores
        numericos.
        """
        operandos = ctx.relacional()
        resultado = self.visit(operandos[0])

        for i in range(1, len(operandos)):
            operador = ctx.getChild(2 * i - 1).getText()
            derecho = self.visit(operandos[i])
            linea = ctx.start.line

            compatibles = (nombre_tipo(resultado) == nombre_tipo(derecho)
                           or (es_numero(resultado) and es_numero(derecho)))
            if not compatibles:
                raise ErrorSemantico(
                    f"no se puede comparar {nombre_tipo(resultado)} con "
                    f"{nombre_tipo(derecho)} usando '{operador}'", linea)

            resultado = (resultado == derecho if operador == "=="
                         else resultado != derecho)
        return resultado

    def visitRelacional(self, ctx: TerraParser.RelacionalContext):
        """Comparaciones de orden. Solo entre valores numericos."""
        operandos = ctx.pertenencia()
        resultado = self.visit(operandos[0])

        for i in range(1, len(operandos)):
            operador = ctx.getChild(2 * i - 1).getText()
            derecho = self.visit(operandos[i])
            linea = ctx.start.line

            if not (es_numero(resultado) and es_numero(derecho)):
                raise ErrorSemantico(
                    f"'{operador}' solo compara numeros, se recibio "
                    f"{nombre_tipo(resultado)} y {nombre_tipo(derecho)}",
                    linea)

            if operador == ">":
                resultado = resultado > derecho
            elif operador == "<":
                resultado = resultado < derecho
            elif operador == ">=":
                resultado = resultado >= derecho
            else:
                resultado = resultado <= derecho
        return resultado

    def visitPertenencia(self, ctx: TerraParser.PertenenciaContext):
        """Operador 'en': pertenencia a una lista o subcadena de un texto."""
        operandos = ctx.suma()
        izquierdo = self.visit(operandos[0])
        if len(operandos) == 1:
            return izquierdo

        derecho = self.visit(operandos[1])
        linea = ctx.start.line

        if isinstance(derecho, list):
            return izquierdo in derecho
        if isinstance(derecho, str):
            if not isinstance(izquierdo, str):
                raise ErrorSemantico(
                    "para buscar dentro de un texto el valor buscado debe "
                    f"ser texto, se recibio {nombre_tipo(izquierdo)}", linea)
            return izquierdo in derecho

        raise ErrorSemantico(
            f"'en' opera sobre listas o textos, se recibio "
            f"{nombre_tipo(derecho)}", linea)

    def visitSuma(self, ctx: TerraParser.SumaContext):
        """
        Suma y resta.

        '+' tambien concatena textos y une listas; '-' solo opera sobre
        numeros. No hay conversion implicita: "a" + 1 es un error.
        """
        operandos = ctx.termino()
        resultado = self.visit(operandos[0])

        for i in range(1, len(operandos)):
            operador = ctx.getChild(2 * i - 1).getText()
            derecho = self.visit(operandos[i])
            linea = ctx.start.line

            if operador == "+":
                if es_numero(resultado) and es_numero(derecho):
                    resultado = resultado + derecho
                elif isinstance(resultado, str) and isinstance(derecho, str):
                    resultado = resultado + derecho
                elif isinstance(resultado, list) and isinstance(derecho, list):
                    resultado = resultado + derecho
                else:
                    raise ErrorSemantico(
                        f"no se puede sumar {nombre_tipo(resultado)} con "
                        f"{nombre_tipo(derecho)}", linea)
            else:
                self._exigir_numeros(resultado, derecho, "-", linea)
                resultado = resultado - derecho
        return resultado

    def visitTermino(self, ctx: TerraParser.TerminoContext):
        """Multiplicacion, division y modulo. Solo entre numeros."""
        operandos = ctx.potencia()
        resultado = self.visit(operandos[0])

        for i in range(1, len(operandos)):
            operador = ctx.getChild(2 * i - 1).getText()
            derecho = self.visit(operandos[i])
            linea = ctx.start.line

            self._exigir_numeros(resultado, derecho, operador, linea)

            if operador == "*":
                resultado = resultado * derecho
            elif operador == "/":
                if derecho == 0:
                    raise ErrorSemantico("division por cero", linea)
                resultado = resultado / derecho
            else:
                if derecho == 0:
                    raise ErrorSemantico("modulo por cero", linea)
                resultado = resultado % derecho
        return resultado

    def visitPotencia(self, ctx: TerraParser.PotenciaContext):
        """Potencia, asociativa a la derecha: 2 ^ 3 ^ 2 es 2 ^ (3 ^ 2)."""
        base = self.visit(ctx.unario())
        if ctx.potencia() is None:
            return base

        exponente = self.visit(ctx.potencia())
        self._exigir_numeros(base, exponente, "^", ctx.start.line)
        return base ** exponente

    def visitUnario(self, ctx: TerraParser.UnarioContext):
        """Negativo aritmetico y negacion logica."""
        if ctx.postfijo() is not None:
            return self.visit(ctx.postfijo())

        operador = ctx.getChild(0).getText()
        valor = self.visit(ctx.unario())
        linea = ctx.start.line

        if operador == "-":
            if not es_numero(valor):
                raise ErrorSemantico(
                    f"el negativo solo aplica a numeros, se recibio "
                    f"{nombre_tipo(valor)}", linea)
            return -valor

        if not isinstance(valor, bool):
            raise ErrorSemantico(
                f"'!' solo aplica a booleanos, se recibio "
                f"{nombre_tipo(valor)}", linea)
        return not valor

    def visitPostfijo(self, ctx: TerraParser.PostfijoContext):
        """Indexacion de listas y textos: lista[0], texto[2]."""
        valor = self.visit(ctx.primario())
        linea = ctx.start.line

        for indice_ctx in ctx.expr():
            indice = self.visit(indice_ctx)

            if not isinstance(valor, (list, str)):
                raise ErrorSemantico(
                    f"solo se puede indexar una lista o un texto, se "
                    f"recibio {nombre_tipo(valor)}", linea)
            if isinstance(indice, bool) or not isinstance(indice, int):
                raise ErrorSemantico(
                    f"el indice debe ser un numero entero, se recibio "
                    f"{nombre_tipo(indice)}", linea)
            if indice < 0 or indice >= len(valor):
                raise ErrorSemantico(
                    f"indice {indice} fuera de rango (longitud "
                    f"{len(valor)})", linea)

            valor = valor[indice]
        return valor

    def visitPrimario(self, ctx: TerraParser.PrimarioContext):
        """Literales, variables, listas, llamadas y parentesis."""
        linea = ctx.start.line

        if ctx.NUMERO() is not None:
            return int(ctx.NUMERO().getText())
        if ctx.DECIMAL() is not None:
            return float(ctx.DECIMAL().getText())
        if ctx.TEXTO() is not None:
            crudo = ctx.TEXTO().getText()[1:-1]
            # Secuencias de escape: \n salto de linea, \t tabulador,
            # \" comilla y \\ barra invertida
            return (crudo.replace("\\\\", "\x00")
                    .replace('\\"', '"')
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\x00", "\\"))
        if ctx.BOOLEANO() is not None:
            return ctx.BOOLEANO().getText() == "verdadero"
        if ctx.NULO() is not None:
            return None
        if ctx.carga() is not None:
            return self.visit(ctx.carga())
        if ctx.llamada() is not None:
            return self.visit(ctx.llamada())
        if ctx.lista() is not None:
            return self.visit(ctx.lista())
        if ctx.expresion() is not None:
            return self.visit(ctx.expresion())

        if ctx.funcion_anonima() is not None:
            return self.visit(ctx.funcion_anonima())

        nombre = ctx.IDENTIFICADOR().getText()
        if nombre not in self.memoria:
            # El nombre de una funcion escrito sin parentesis es un valor
            # de tipo funcion: asi se puede pasar a mapear, filtro, etc.
            if nombre in self.funciones or nombre in self._incorporadas():
                return biblioteca.FuncionNombrada(nombre)
            raise ErrorSemantico(
                f"la variable '{nombre}' no ha sido declarada", linea)
        return self.memoria[nombre]["valor"]

    def visitFuncion_anonima(self, ctx):
        """
        funcion(x, y) => expresion
        Crea un valor de tipo funcion. El cuerpo no se evalua aqui, sino
        cada vez que alguien llama a la funcion.
        """
        parametros = [t.getText() for t in ctx.IDENTIFICADOR()]
        if len(set(parametros)) != len(parametros):
            raise ErrorSemantico(
                "un parametro aparece repetido en la funcion anonima",
                ctx.start.line)
        return biblioteca.FuncionAnonima(parametros, ctx.expr())

    def visitLista(self, ctx: TerraParser.ListaContext):
        return [self.visit(e) for e in ctx.expr()]

    # -----------------------------------------------------------------
    # FUNCIONES INCORPORADAS
    # -----------------------------------------------------------------

    def _incorporadas(self):
        """
        Catalogo de funciones que trae el lenguaje.

        Se arma una sola vez y se guarda, porque visitLlamada lo consulta
        en cada llamada (en un 'filtrar' eso es una vez por fila). A las
        funciones propias del evaluador se suman las de src/biblioteca.py:
        matematicas, matrices y las de los pasos siguientes.
        """
        if getattr(self, "_catalogo", None) is None:
            self._catalogo = {
                "longitud": self._f_longitud,
                "suma": self._f_suma,
                "describir": self._f_describir,
                "contar": self._f_contar,
                "media": self._f_media,
                "mediana": self._f_mediana,
                "minimo": self._f_minimo,
                "maximo": self._f_maximo,
                "desviacion": self._f_desviacion,
                "raiz": self._f_raiz,
                "absoluto": self._f_absoluto,
                "redondear": self._f_redondear,
                "a_numero": self._f_a_numero,
                "a_decimal": self._f_a_decimal,
                "a_texto": self._f_a_texto,
                "agregar": self._f_agregar,
                "insertar": self._f_insertar,
                "quitar": self._f_quitar,
                "regresion_lineal": self._f_regresion_lineal,
                "regresion_multiple": self._f_regresion_multiple,
                "predecir": self._f_predecir,
                "predecir_multiple": self._f_predecir_multiple,
                "coeficientes": self._f_coeficientes,
                "r_cuadrado": self._f_r_cuadrado,
                "arbol_decision":    self._f_arbol_decision,
                "clasificar":        self._f_clasificar,
                "clasificar_lote":   self._f_clasificar_lote,
                "evaluar_arbol":     self._f_evaluar_arbol,
                "exactitud":         self._f_exactitud,
                "mostrar_arbol":     self._f_mostrar_arbol,

                # --- pandas ---
                "dataframe":        self._f_dataframe,
                "desde_tabla":      self._f_desde_tabla,
                "head":             self._f_head,
                "tail":             self._f_tail,
                "info_df":          self._f_info_df,
                "sample":           self._f_sample,
                "merge":            self._f_merge,
                "melt":             self._f_melt,
                "pivot":            self._f_pivot,
                "reemplazar":       self._f_reemplazar,
                "duplicados_df":    self._f_duplicados_df,
                "correlacion":      self._f_correlacion,

                # --- numpy ---
                "array":            self._f_array,
                "zeros":            self._f_zeros,
                "ones":             self._f_ones,
                "identidad":        self._f_identidad,
                "rango_array":      self._f_rango_array,
                "linspace":         self._f_linspace,
                "reshape":          self._f_reshape,
                "flatten":          self._f_flatten,
                "concatenar":       self._f_concatenar,
                "transpuesta":      self._f_transpuesta,
                "producto_mat":     self._f_producto_mat,
                "dot":              self._f_dot,
                "determinante":     self._f_determinante,
                "traza":            self._f_traza,
                "norma":            self._f_norma,
                "suma_array":       self._f_suma_array,
                "media_array":      self._f_media_array,
                "min_array":        self._f_min_array,
                "max_array":        self._f_max_array,
                "raiz_array":       self._f_raiz_array,
                "abs_array":        self._f_abs_array,
                # --- operaciones aritméticas de arrays ---
                "sumar_arrays":      self._f_sumar_arrays,
                "restar_arrays":     self._f_restar_arrays,
                "multiplicar_arr":   self._f_multiplicar_arr,
                "dividir_arr":       self._f_dividir_arr,
                "potencia_arr":      self._f_potencia_arr,

                # --- plot ---
                "graficar_lineas":      self._f_graficar_lineas,
                "graficar_barras":      self._f_graficar_barras,
                "graficar_dispersion":  self._f_graficar_dispersion,
                "graficar_histograma":  self._f_graficar_histograma,
                "graficar_caja":        self._f_graficar_caja,
                "exportar_lineas":      self._f_exportar_lineas,
                "exportar_barras":      self._f_exportar_barras,
                "exportar_dispersion":  self._f_exportar_dispersion,
                "exportar_histograma":  self._f_exportar_histograma,
            }
            for nombre, funcion in biblioteca.funciones(self).items():
                self._catalogo[nombre] = self._envolver(funcion)
        return self._catalogo

    def _envolver(self, funcion):
        """
        Adapta una funcion de la biblioteca: los ErrorDatos que lance se
        convierten en ErrorSemantico con la linea de la llamada.
        """
        def llamar(args, linea):
            try:
                return funcion(args, linea)
            except ErrorDatos as e:
                raise ErrorSemantico(str(e), linea)
        return llamar

    def visitLlamada(self, ctx: TerraParser.LlamadaContext):
        """
        Resuelve una llamada a funcion.

        Primero busca entre las incorporadas y despues entre las que
        definio el usuario. Ese orden garantiza que una definicion
        propia nunca oculte una del lenguaje.
        """
        nombre = ctx.IDENTIFICADOR().getText()
        linea = ctx.start.line
        argumentos = [self.visit(a) for a in ctx.expr()]

        incorporadas = self._incorporadas()
        if nombre in incorporadas:
            return incorporadas[nombre](argumentos, linea)
        if nombre in self.funciones:
            return self._invocar(nombre, argumentos, linea)

                # Una variable que guarda una funcion tambien se puede llamar
        if nombre in self.memoria and biblioteca.es_funcion(
                self.memoria[nombre]["valor"]):
            try:
                return biblioteca.llamar_funcion(
                    self, self.memoria[nombre]["valor"], argumentos, linea)
            except ErrorDatos as e:
                raise ErrorSemantico(str(e), linea)
        raise ErrorSemantico(f"la funcion '{nombre}' no existe", linea)

    def visitFuncion_decl(self, ctx):
        """
        Registra una funcion definida por el usuario.

        No ejecuta el cuerpo: solo guarda sus parametros y el nodo del
        bloque para visitarlo cuando alguien la llame.
        """
        identificadores = ctx.IDENTIFICADOR()
        nombre = identificadores[0].getText()
        parametros = [t.getText() for t in identificadores[1:]]
        linea = ctx.start.line

        if nombre in self._incorporadas():
            raise ErrorSemantico(
                f"'{nombre}' es una funcion incorporada del lenguaje y no "
                f"puede redefinirse", linea)
        if nombre in self.funciones:
            raise ErrorSemantico(
                f"la funcion '{nombre}' ya fue definida", linea)
        if len(set(parametros)) != len(parametros):
            raise ErrorSemantico(
                f"la funcion '{nombre}' repite el nombre de un parametro",
                linea)

        self.funciones[nombre] = {
            "parametros": parametros,
            "cuerpo": ctx.bloque(),
        }
        return None

    def _invocar(self, nombre, argumentos, linea):
        """
        Ejecuta una funcion del usuario y devuelve su resultado.

        El cuerpo se evalua con una memoria propia que contiene solo los
        parametros: la funcion no ve las variables del programa principal
        ni puede modificarlas. Al terminar se restaura la memoria
        anterior, de modo que la llamada no deja rastro.

        Una funcion sin 'retornar' devuelve nulo.
        """
        definicion = self.funciones[nombre]
        parametros = definicion["parametros"]

        if len(argumentos) != len(parametros):
            raise ErrorSemantico(
                f"la funcion '{nombre}' espera {len(parametros)} "
                f"argumento(s) y recibio {len(argumentos)}", linea)

        if self.profundidad >= self.LIMITE_RECURSION:
            raise ErrorSemantico(
                f"la funcion '{nombre}' supero el limite de "
                f"{self.LIMITE_RECURSION} llamadas anidadas; revisa el caso "
                f"base de la recursion", linea)

        memoria_anterior = self.memoria
        self.memoria = {
            p: {"valor": v, "mutable": True}
            for p, v in zip(parametros, argumentos)
        }
        self.profundidad += 1

        try:
            self.visit(definicion["cuerpo"])
            resultado = None
        except RetornoSenal as senal:
            resultado = senal.valor
        except RecursionError:
            raise ErrorSemantico(
                f"la funcion '{nombre}' se llamo demasiadas veces de forma "
                f"anidada y se agoto la pila; revisa el caso base de la "
                f"recursion", linea)
        finally:
            self.memoria = memoria_anterior
            self.profundidad -= 1

        return resultado

    def visitRetornar_stmt(self, ctx):
        """Interrumpe la funcion actual y entrega un valor a la llamada."""
        if self.profundidad == 0:
            raise ErrorSemantico(
                "'retornar' solo puede usarse dentro de una funcion",
                ctx.start.line)

        valor = (self.visit(ctx.expresion())
                 if ctx.expresion() is not None else None)
        raise RetornoSenal(valor)

    def _lista_numerica(self, argumentos, linea, funcion):
        """Valida que el argumento sea una lista de numeros y la devuelve."""
        if len(argumentos) != 1 or not isinstance(argumentos[0], list):
            raise ErrorSemantico(
                f"{funcion}() espera una lista de numeros", linea)
        datos = argumentos[0]
        if not datos:
            raise ErrorSemantico(
                f"{funcion}() no puede operar sobre una lista vacia", linea)
        for valor in datos:
            if not es_numero(valor):
                raise ErrorSemantico(
                    f"{funcion}() solo acepta numeros, encontro "
                    f"{nombre_tipo(valor)}", linea)
        return datos

    def _un_numero(self, argumentos, linea, funcion):
        """Valida que haya exactamente un argumento numerico."""
        if len(argumentos) != 1 or not es_numero(argumentos[0]):
            raise ErrorSemantico(
                f"{funcion}() espera un numero", linea)
        return argumentos[0]

    def _f_longitud(self, args, linea):
        if len(args) != 1 or not isinstance(args[0], (list, str)):
            raise ErrorSemantico("longitud() espera una lista o un texto",
                                 linea)
        return len(args[0])

    def _f_describir(self, args, linea):
        """describir(tabla) -> tabla con estadisticas de cada columna."""
        if len(args) != 1 or not isinstance(args[0], Tabla):
            recibido = ", ".join(nombre_tipo(a) for a in args) or "nada"
            raise ErrorSemantico(
                f"describir() necesita una tabla, se recibio {recibido}",
                linea)
        return ops.describir(args[0])

    def _f_contar(self, args, linea):
        """
        contar()      -> numero de filas del grupo (solo en 'resumir')
        contar(lista) -> cantidad de elementos de la lista
        """
        if not args:
            tamano = getattr(self, "_tamano_grupo", None)
            if tamano is None:
                raise ErrorSemantico(
                    "contar() sin argumentos solo puede usarse dentro "
                    "de 'resumir'", linea)
            return tamano
        if len(args) == 1 and isinstance(args[0], list):
            return len(args[0])
        raise ErrorSemantico(
            "contar() recibe una lista o ningun argumento", linea)
    def _f_suma(self, args, linea):
        return sum(self._lista_numerica(args, linea, "suma"))

    def _f_media(self, args, linea):
        datos = self._lista_numerica(args, linea, "media")
        return sum(datos) / len(datos)

    def _f_mediana(self, args, linea):
        datos = sorted(self._lista_numerica(args, linea, "mediana"))
        n = len(datos)
        medio = n // 2
        if n % 2 == 1:
            return datos[medio]
        return (datos[medio - 1] + datos[medio]) / 2

    def _f_minimo(self, args, linea):
        return min(self._lista_numerica(args, linea, "minimo"))

    def _f_maximo(self, args, linea):
        return max(self._lista_numerica(args, linea, "maximo"))

    def _f_desviacion(self, args, linea):
        """Desviacion estandar muestral, calculada sin librerias."""
        datos = self._lista_numerica(args, linea, "desviacion")
        if len(datos) < 2:
            raise ErrorSemantico(
                "desviacion() necesita al menos dos valores", linea)
        promedio = sum(datos) / len(datos)
        varianza = sum((x - promedio) ** 2 for x in datos) / (len(datos) - 1)
        return varianza ** 0.5

    def _f_raiz(self, args, linea):
        valor = self._un_numero(args, linea, "raiz")
        if valor < 0:
            raise ErrorSemantico(
                "raiz() no acepta numeros negativos", linea)
        return valor ** 0.5

    def _f_absoluto(self, args, linea):
        return abs(self._un_numero(args, linea, "absoluto"))

    def _f_redondear(self, args, linea):
        """redondear(x) o redondear(x, decimales)."""
        if not args or not es_numero(args[0]):
            raise ErrorSemantico("redondear() espera un numero", linea)
        if len(args) == 1:
            return round(args[0])
        if isinstance(args[1], bool) or not isinstance(args[1], int):
            raise ErrorSemantico(
                "el segundo argumento de redondear() debe ser un entero",
                linea)
        return round(args[0], args[1])

    def _f_a_numero(self, args, linea):
        """Conversion explicita a numero entero."""
        if len(args) != 1:
            raise ErrorSemantico("a_numero() espera un argumento", linea)
        valor = args[0]
        if es_numero(valor):
            return int(valor)
        if isinstance(valor, str):
            try:
                return int(float(valor))
            except ValueError:
                raise ErrorSemantico(
                    f"no se puede convertir \"{valor}\" a numero", linea)
        raise ErrorSemantico(
            f"no se puede convertir {nombre_tipo(valor)} a numero", linea)

    def _f_a_decimal(self, args, linea):
        """Conversion explicita a decimal."""
        if len(args) != 1:
            raise ErrorSemantico("a_decimal() espera un argumento", linea)
        valor = args[0]
        if es_numero(valor):
            return float(valor)
        if isinstance(valor, str):
            try:
                return float(valor)
            except ValueError:
                raise ErrorSemantico(
                    f"no se puede convertir \"{valor}\" a decimal", linea)
        raise ErrorSemantico(
            f"no se puede convertir {nombre_tipo(valor)} a decimal", linea)

    def _f_a_texto(self, args, linea):
        """Conversion explicita a texto."""
        if len(args) != 1:
            raise ErrorSemantico("a_texto() espera un argumento", linea)
        return formatear(args[0])

    # -----------------------------------------------------------------
    # VALIDACIONES COMPARTIDAS
    # -----------------------------------------------------------------

    def _exigir_numeros(self, izquierdo, derecho, operador, linea):
        """Aborta si alguno de los dos operandos no es numerico."""
        if not (es_numero(izquierdo) and es_numero(derecho)):
            raise ErrorSemantico(
                f"'{operador}' solo opera sobre numeros, se recibio "
                f"{nombre_tipo(izquierdo)} y {nombre_tipo(derecho)}", linea)

    def _exigir_booleano(self, valor, operador, linea):
        """Aborta si el valor no es booleano."""
        if not isinstance(valor, bool):
            raise ErrorSemantico(
                f"'{operador}' solo opera sobre booleanos, se recibio "
                f"{nombre_tipo(valor)}", linea)

    def _f_agregar(self, args, linea):
        """
        agregar(lista, elemento) -> lista nueva con el elemento al final.
        No muta la lista original; devuelve una copia extendida.
        """
        if len(args) != 2 or not isinstance(args[0], list):
            raise ErrorSemantico(
                "agregar() espera una lista y un elemento", linea)
        return args[0] + [args[1]]


    def _f_insertar(self, args, linea):
        """
        insertar(lista, indice, elemento) -> lista nueva con el elemento
        en la posicion indicada. Los elementos siguientes se desplazan.
        El indice puede ser 0 (inicio) hasta longitud(lista) (final).
        """
        if len(args) != 3 or not isinstance(args[0], list):
            raise ErrorSemantico(
                "insertar() espera una lista, un indice y un elemento", linea)
        lista, indice, elemento = args
        if isinstance(indice, bool) or not isinstance(indice, int):
            raise ErrorSemantico(
                "el indice de insertar() debe ser un numero entero", linea)
        if indice < 0 or indice > len(lista):
            raise ErrorSemantico(
                f"indice {indice} fuera de rango para insertar "
                f"(longitud {len(lista)})", linea)
        return lista[:indice] + [elemento] + lista[indice:]


    def _f_quitar(self, args, linea):
        """
        quitar(lista, indice) -> lista nueva sin el elemento en esa posicion.
        """
        if len(args) != 2 or not isinstance(args[0], list):
            raise ErrorSemantico(
                "quitar() espera una lista y un indice", linea)
        lista, indice = args
        if isinstance(indice, bool) or not isinstance(indice, int):
            raise ErrorSemantico(
                "el indice de quitar() debe ser un numero entero", linea)
        if indice < 0 or indice >= len(lista):
            raise ErrorSemantico(
                f"indice {indice} fuera de rango (longitud {len(lista)})", linea)
        return lista[:indice] + lista[indice + 1:]  
    # -----------------------------------------------------------------
    # Predicciones y regresiones
    # -----------------------------------------------------------------
    def _f_regresion_lineal(self, args, linea):
        """
        regresion_lineal(xs, ys) -> modelo
        Entrena una regresion lineal simple y devuelve el modelo como diccionario.
        """
        if len(args) != 2:
            raise ErrorSemantico(
                "regresion_lineal() espera dos listas: xs e ys", linea)
        try:
            return reg.entrenar_simple(args[0], args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_regresion_multiple(self, args, linea):
        """
        regresion_multiple(X, ys) -> modelo
        X es una lista de listas (cada sublista es una observacion).
        """
        if len(args) != 2:
            raise ErrorSemantico(
                "regresion_multiple() espera una matriz X y una lista ys", linea)
        try:
            return reg.entrenar_multiple(args[0], args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_predecir(self, args, linea):
        """
        predecir(modelo, x) -> decimal
        Aplica el modelo simple a un valor de X.
        """
        if len(args) != 2:
            raise ErrorSemantico(
                "predecir() espera un modelo y un valor x", linea)
        try:
            return reg.predecir_simple(args[0], args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_predecir_multiple(self, args, linea):
        """
        predecir_multiple(modelo, [x1, x2, ...]) -> decimal
        Aplica el modelo multiple a un vector de predictores.
        """
        if len(args) != 2:
            raise ErrorSemantico(
                "predecir_multiple() espera un modelo y una lista de predictores",
                linea)
        try:
            return reg.predecir_multiple(args[0], args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_coeficientes(self, args, linea):
        """
        coeficientes(modelo) -> lista
        Devuelve [intercepto, pendiente] para simple,
        o [b0, b1, b2, ...] para multiple.
        """
        if len(args) != 1 or not isinstance(args[0], dict):
            raise ErrorSemantico(
                "coeficientes() espera un modelo entrenado", linea)
        modelo = args[0]
        if modelo.get("tipo") == "simple":
            return [modelo["intercepto"], modelo["pendiente"]]
        if modelo.get("tipo") == "multiple":
            return modelo["coeficientes"]
        raise ErrorSemantico(
            "coeficientes() recibio un valor que no es un modelo", linea)


    def _f_r_cuadrado(self, args, linea):
        """
        r_cuadrado(modelo) -> decimal entre 0 y 1
        """
        if len(args) != 1 or not isinstance(args[0], dict):
            raise ErrorSemantico(
                "r_cuadrado() espera un modelo entrenado", linea)
        modelo = args[0]
        if "r_cuadrado" not in modelo:
            raise ErrorSemantico(
                "r_cuadrado() recibio un valor que no es un modelo", linea)
        return modelo["r_cuadrado"]
    
    def _f_arbol_decision(self, args, linea):
        """
        arbol_decision(X, ys)
        arbol_decision(X, ys, max_profundidad)
        arbol_decision(X, ys, max_profundidad, min_muestras)
        """
        if len(args) < 2 or len(args) > 4:
            raise ErrorSemantico(
                "arbol_decision() espera entre 2 y 4 argumentos: "
                "X, ys [, max_profundidad] [, min_muestras]", linea)

        X   = args[0]
        ys  = args[1]
        max_prof     = args[2] if len(args) > 2 else 5
        min_muestras = args[3] if len(args) > 3 else 2

        for param, nombre in [(max_prof, "max_profundidad"),
                            (min_muestras, "min_muestras")]:
            if isinstance(param, bool) or not isinstance(param, int) or param < 1:
                raise ErrorSemantico(
                    f"arbol_decision(): {nombre} debe ser un entero >= 1",
                    linea)
        try:
            return ad.entrenar_arbol(X, ys, max_prof, min_muestras)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_clasificar(self, args, linea):
        """clasificar(modelo, [x1, x2, ...]) -> clase predicha"""
        if len(args) != 2 or not isinstance(args[1], list):
            raise ErrorSemantico(
                "clasificar() espera un modelo y una lista de predictores",
                linea)
        try:
            return ad.predecir_clase(args[0], args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_clasificar_lote(self, args, linea):
        """clasificar_lote(modelo, X) -> lista de clases predichas"""
        if len(args) != 2 or not isinstance(args[1], list):
            raise ErrorSemantico(
                "clasificar_lote() espera un modelo y una lista de observaciones",
                linea)
        try:
            return ad.predecir_lote(args[0], args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_evaluar_arbol(self, args, linea):
        """evaluar_arbol(modelo, X, ys) -> diccionario con metricas"""
        if len(args) != 3:
            raise ErrorSemantico(
                "evaluar_arbol() espera un modelo, una matriz X y una lista ys",
                linea)
        try:
            return ad.evaluar_modelo(args[0], args[1], args[2])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_exactitud(self, args, linea):
        """exactitud(resultado_evaluar) -> decimal entre 0 y 1"""
        if len(args) != 1 or not isinstance(args[0], dict):
            raise ErrorSemantico(
                "exactitud() espera el resultado de evaluar_arbol()", linea)
        if "exactitud" not in args[0]:
            raise ErrorSemantico(
                "exactitud() recibio un valor que no es un resultado de "
                "evaluacion", linea)
        return args[0]["exactitud"]


    def _f_mostrar_arbol(self, args, linea):
        """
        mostrar_arbol(modelo)
        mostrar_arbol(modelo, [nombre_col1, nombre_col2, ...])
        Imprime el arbol en texto y devuelve nulo.
        """
        if len(args) < 1 or len(args) > 2:
            raise ErrorSemantico(
                "mostrar_arbol() espera un modelo y opcionalmente "
                "una lista de nombres de columnas", linea)
        try:
            ad._validar_modelo(args[0])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)

        nombres = args[1] if len(args) == 2 else None
        if nombres is not None and not isinstance(nombres, list):
            raise ErrorSemantico(
                "el segundo argumento de mostrar_arbol() debe ser una lista "
                "de textos", linea)

        modelo = args[0]
        internos, hojas = ad._contar_nodos(modelo["raiz"])
        prof_real       = ad._profundidad_real(modelo["raiz"])

        encabezado = (f"Arbol de decision "
                    f"(n={modelo['n']}, p={modelo['p']}, "
                    f"clases={modelo['clases']})\n"
                    f"Profundidad real: {prof_real}  |  "
                    f"Nodos internos: {internos}  |  "
                    f"Hojas: {hojas}\n"
                    + "-" * 60)
        arbol_txt = ad._imprimir_arbol(modelo["raiz"], nombres,
                                    prefijo="", es_izquierda=True)
        print(encabezado)
        print(arbol_txt)
        return None


# --- Implementacion de funciones de pandas ---
    def _f_dataframe(self, args, linea):
        if len(args) != 1 or not isinstance(args[0], dict):
            raise ErrorSemantico(
                "dataframe() espera un diccionario {columna: lista}", linea)
        # En Terra los diccionarios no existen como tipo nativo, pero los
        # modelos son dicts. Se usa desde Terra pasando listas paralelas:
        # dataframe([col1, col2], [[v1,v2,...], [v1,v2,...]])
        raise ErrorSemantico(
            "usa dataframe_cols(nombres, listas) para crear un DataFrame", linea)


    def _f_dataframe_cols(self, args, linea):
        """dataframe_cols([col1, col2], [[vals1], [vals2]]) -> DataFrame"""
        if (len(args) != 2
                or not isinstance(args[0], list)
                or not isinstance(args[1], list)):
            raise ErrorSemantico(
                "dataframe_cols() espera una lista de nombres y una lista "
                "de listas de valores", linea)
        nombres = args[0]
        listas  = args[1]
        if len(nombres) != len(listas):
            raise ErrorSemantico(
                "dataframe_cols(): el numero de nombres y de listas debe coincidir",
                linea)
        try:
            datos = {nombres[i]: listas[i] for i in range(len(nombres))}
            return pd.crear_dataframe(datos)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_desde_tabla(self, args, linea):
        if len(args) != 1 or not isinstance(args[0], Tabla):
            raise ErrorSemantico("desde_tabla() espera una Tabla", linea)
        return pd.desde_tabla(args[0])


    def _f_head(self, args, linea):
        n = args[1] if len(args) == 2 else 5
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ErrorSemantico("head() espera un entero >= 1", linea)
        try:
            return pd.head(args[0], n)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_tail(self, args, linea):
        n = args[1] if len(args) == 2 else 5
        try:
            return pd.tail(args[0], n)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_info_df(self, args, linea):
        if len(args) != 1:
            raise ErrorSemantico("info_df() espera un DataFrame", linea)
        try:
            print(pd.info(args[0]))
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        return None


    def _f_sample(self, args, linea):
        if len(args) < 2:
            raise ErrorSemantico("sample() espera un DataFrame y n", linea)
        semilla = args[2] if len(args) == 3 else None
        try:
            return pd.sample(args[0], args[1], semilla)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_merge(self, args, linea):
        tipo = args[3] if len(args) == 4 else "inner"
        if not isinstance(tipo, str) or tipo not in ("inner","left","right","outer"):
            raise ErrorSemantico(
                "merge(): el tipo debe ser 'inner', 'left', 'right' u 'outer'",
                linea)
        try:
            return pd.merge(args[0], args[1], args[2], tipo)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_melt(self, args, linea):
        if len(args) < 2:
            raise ErrorSemantico(
                "melt() espera un DataFrame y una lista de columnas id", linea)
        nom_var = args[2] if len(args) > 2 else "variable"
        nom_val = args[3] if len(args) > 3 else "valor"
        try:
            return pd.melt(args[0], args[1], nom_var, nom_val)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_pivot(self, args, linea):
        if len(args) != 4:
            raise ErrorSemantico(
                "pivot() espera DataFrame, indice, columna, valor", linea)
        try:
            return pd.pivot(args[0], args[1], args[2], args[3])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_reemplazar(self, args, linea):
        if len(args) != 4:
            raise ErrorSemantico(
                "reemplazar() espera DataFrame, columna, viejo, nuevo", linea)
        try:
            return pd.replace_val(args[0], args[1], args[2], args[3])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_duplicados_df(self, args, linea):
        if len(args) != 1:
            raise ErrorSemantico("duplicados_df() espera un DataFrame", linea)
        try:
            return pd.duplicados(args[0])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)


    def _f_correlacion(self, args, linea):
        if len(args) != 1:
            raise ErrorSemantico("correlacion() espera un DataFrame", linea)
        try:
            return pd.corr(args[0])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)

# -- Implementacion de funciones de numpy ---
    def _wrap_np(self, f, args, linea, nombre):
        try:
            return f(*args)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        except TypeError:
            raise ErrorSemantico(f"{nombre}(): argumentos incorrectos", linea)

    def _f_array(self, args, linea):
        if len(args) != 1 or not isinstance(args[0], list):
            raise ErrorSemantico("array() espera una lista", linea)
        return self._wrap_np(np.array, args, linea, "array")

    def _f_zeros(self, args, linea):
        return self._wrap_np(np.zeros, args, linea, "zeros")

    def _f_ones(self, args, linea):
        return self._wrap_np(np.ones, args, linea, "ones")

    def _f_identidad(self, args, linea):
        return self._wrap_np(np.identidad, args, linea, "identidad")

    def _f_rango_array(self, args, linea):
        return self._wrap_np(np.rango, args, linea, "rango_array")

    def _f_linspace(self, args, linea):
        return self._wrap_np(np.linspace, args, linea, "linspace")

    def _f_reshape(self, args, linea):
        return self._wrap_np(np.reshape, args, linea, "reshape")

    def _f_flatten(self, args, linea):
        return self._wrap_np(np.flatten, args, linea, "flatten")

    def _f_concatenar(self, args, linea):
        return self._wrap_np(np.concatenar, args, linea, "concatenar")

    def _f_transpuesta(self, args, linea):
        return self._wrap_np(np.transpuesta, args, linea, "transpuesta")

    def _f_producto_mat(self, args, linea):
        return self._wrap_np(np.producto_matricial, args, linea, "producto_mat")

    def _f_dot(self, args, linea):
        return self._wrap_np(np.dot, args, linea, "dot")

    def _f_determinante(self, args, linea):
        return self._wrap_np(np.determinante, args, linea, "determinante")

    def _f_traza(self, args, linea):
        return self._wrap_np(np.traza, args, linea, "traza")

    def _f_norma(self, args, linea):
        return self._wrap_np(np.norma, args, linea, "norma")

    def _f_suma_array(self, args, linea):
        return self._wrap_np(np.suma_array, args, linea, "suma_array")

    def _f_media_array(self, args, linea):
        return self._wrap_np(np.media_array, args, linea, "media_array")

    def _f_min_array(self, args, linea):
        return self._wrap_np(np.min_array, args, linea, "min_array")

    def _f_max_array(self, args, linea):
        return self._wrap_np(np.max_array, args, linea, "max_array")

    def _f_raiz_array(self, args, linea):
        return self._wrap_np(np.raiz_array, args, linea, "raiz_array")

    def _f_abs_array(self, args, linea):
        return self._wrap_np(np.abs_array, args, linea, "abs_array")

# --- Implementacion de funciones de graficado ---
    def _f_graficar_lineas(self, args, linea):
        if len(args) < 2:
            raise ErrorSemantico(
                "graficar_lineas() espera xs, ys [, titulo] [, eje_x] [, eje_y]",
                linea)
        titulo = args[2] if len(args) > 2 else ""
        ej_x   = args[3] if len(args) > 3 else "x"
        ej_y   = args[4] if len(args) > 4 else "y"
        try:
            plt.grafica_lineas_ascii(args[0], args[1], titulo, ej_x, ej_y)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        return None


    def _f_graficar_barras(self, args, linea):
        if len(args) < 2:
            raise ErrorSemantico(
                "graficar_barras() espera categorias, valores [, titulo] [, eje_y]",
                linea)
        titulo = args[2] if len(args) > 2 else ""
        ej_y   = args[3] if len(args) > 3 else "valor"
        try:
            plt.grafica_barras_ascii(args[0], args[1], titulo, ej_y)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        return None


    def _f_graficar_dispersion(self, args, linea):
        if len(args) < 2:
            raise ErrorSemantico(
                "graficar_dispersion() espera xs, ys [, titulo]", linea)
        titulo = args[2] if len(args) > 2 else ""
        ej_x   = args[3] if len(args) > 3 else "x"
        ej_y   = args[4] if len(args) > 4 else "y"
        try:
            plt.grafica_dispersion_ascii(args[0], args[1], titulo, ej_x, ej_y)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        return None


    def _f_graficar_histograma(self, args, linea):
        if len(args) < 1:
            raise ErrorSemantico("graficar_histograma() espera una lista", linea)
        bins   = args[1] if len(args) > 1 else 10
        titulo = args[2] if len(args) > 2 else ""
        ej_x   = args[3] if len(args) > 3 else "valor"
        try:
            plt.grafica_histograma_ascii(args[0], bins, titulo, ej_x)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        return None


    def _f_graficar_caja(self, args, linea):
        if len(args) < 1:
            raise ErrorSemantico("graficar_caja() espera una lista", linea)
        titulo = args[1] if len(args) > 1 else ""
        etiq   = args[2] if len(args) > 2 else "valor"
        try:
            plt.grafica_caja_ascii(args[0], titulo, etiq)
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        return None


    def _f_exportar_lineas(self, args, linea):
        if len(args) < 3:
            raise ErrorSemantico(
                "exportar_lineas() espera xs, ys, ruta [, titulo] [, eje_x] [, eje_y]",
                linea)
        titulo = args[3] if len(args) > 3 else ""
        ej_x   = args[4] if len(args) > 4 else "x"
        ej_y   = args[5] if len(args) > 5 else "y"
        ys     = args[1] if isinstance(args[1][0], list) else [args[1]]
        etiq   = ["serie"]
        try:
            plt.exportar_lineas_svg(args[0], ys, etiq, titulo, ej_x, ej_y, args[2])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        return None


    def _f_exportar_barras(self, args, linea):
        if len(args) < 3:
            raise ErrorSemantico(
                "exportar_barras() espera categorias, valores, ruta [, titulo]",
                linea)
        titulo = args[3] if len(args) > 3 else ""
        ej_x   = args[4] if len(args) > 4 else "categoria"
        ej_y   = args[5] if len(args) > 5 else "valor"
        try:
            plt.exportar_barras_svg(args[0], args[1], titulo, ej_x, ej_y, args[2])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        return None


    def _f_exportar_dispersion(self, args, linea):
        if len(args) < 3:
            raise ErrorSemantico(
                "exportar_dispersion() espera xs, ys, ruta [, titulo]", linea)
        titulo = args[3] if len(args) > 3 else ""
        ej_x   = args[4] if len(args) > 4 else "x"
        ej_y   = args[5] if len(args) > 5 else "y"
        try:
            plt.exportar_dispersion_svg(args[0], args[1], titulo, ej_x, ej_y, args[2])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        return None


    def _f_exportar_histograma(self, args, linea):
        if len(args) < 2:
            raise ErrorSemantico(
                "exportar_histograma() espera lista, ruta [, bins] [, titulo]",
                linea)
        bins   = args[2] if len(args) > 2 else 10
        titulo = args[3] if len(args) > 3 else ""
        ej_x   = args[4] if len(args) > 4 else "valor"
        try:
            plt.exportar_histograma_svg(args[0], bins, titulo, ej_x, args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
        return None

    def _f_sumar_arrays(self, args, linea):
        if len(args) != 2:
            raise ErrorSemantico(
                "sumar_arrays() espera dos arrays o un array y un número", linea)
        try:
            return np.sumar_arrays(args[0], args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)

    def _f_restar_arrays(self, args, linea):
        if len(args) != 2:
            raise ErrorSemantico(
                "restar_arrays() espera dos arrays o un array y un número", linea)
        try:
            return np.restar_arrays(args[0], args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)

    def _f_multiplicar_arr(self, args, linea):
        if len(args) != 2:
            raise ErrorSemantico(
                "multiplicar_arr() espera dos arrays o un array y un número", linea)
        try:
            return np.multiplicar_arrays(args[0], args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)

    def _f_dividir_arr(self, args, linea):
        if len(args) != 2:
            raise ErrorSemantico(
                "dividir_arr() espera dos arrays o un array y un número", linea)
        try:
            return np.dividir_arrays(args[0], args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)

    def _f_potencia_arr(self, args, linea):
        if len(args) != 2:
            raise ErrorSemantico(
                "potencia_arr() espera un array y un exponente numérico", linea)
        if not isinstance(args[1], (int, float)) or isinstance(args[1], bool):
            raise ErrorSemantico(
                "potencia_arr(): el exponente debe ser un número", linea)
        try:
            return np.potencia_array(args[0], args[1])
        except ErrorDatos as e:
            raise ErrorSemantico(str(e), linea)
