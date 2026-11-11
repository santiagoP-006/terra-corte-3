# Terra

Lenguaje de dominio específico (DSL) para el análisis de datos ambientales y de sostenibilidad.

Terra permite describir, con una sintaxis corta y declarativa en español, un flujo completo de ciencia de datos: cargar un CSV, limpiarlo, transformarlo, agruparlo, calcular estadísticas, entrenar modelos de predicción y clasificación, y guardar los resultados.

```
fijo datos = cargar "datos/emisiones_ejemplo.csv"

fijo resumen = datos
    |> eliminar nulos
    |> crear co2_per_capita = co2 * 1000000 / poblacion
    |> agrupar por [pais]
    |> resumir promedio = redondear(media(co2_per_capita), 2),
               registros = contar()
    |> ordenar por promedio desc

mostrar(resumen)
guardar resumen como "salidas/resumen.csv"
```

Proyecto del curso **Lenguajes de Programación y Transducción**, Universidad Sergio Arboleda, semestre 2026-2.

---

## Características

- Gramática escrita en **ANTLR 4** y ejecución con el **patrón Visitor** en Python.
- **Sin librerías externas de datos ni de machine learning:** la estructura de tabla, el lector y escritor de CSV, todas las estadísticas, la regresión lineal y el árbol de decisión están implementados desde cero.
- **Tuberías (`|>`)**: cada operación recibe una tabla y devuelve una tabla nueva; la original nunca se modifica.
- **Tipado estricto:** `numero`, `decimal`, `texto`, `booleano`, `nulo`, `lista` y `tabla`, con errores claros cuando se mezclan tipos incompatibles.
- **Errores en español con número de línea**, en las etapas léxica, sintáctica y semántica.
- Variables, control de flujo, funciones definidas por el usuario y funciones incorporadas.
- **Manipulación de listas:** `agregar`, `insertar` y `quitar` como funciones de primera clase.
- **Regresión lineal simple y múltiple** implementada desde cero con mínimos cuadrados.
- **Árbol de decisión CART** para clasificación, implementado desde cero con índice de Gini.

---

## Estructura del proyecto

```
terra/
  gramaticas/
    Terra.g4                   # Gramática del lenguaje (se edita a mano)
    TerraLexer.py              # Generado por ANTLR - no editar
    TerraParser.py             # Generado por ANTLR - no editar
    TerraVisitor.py            # Generado por ANTLR - no editar
  src/
    terra.py                   # Punto de entrada: CLI y modo interactivo
    visitante_evaluador.py     # Semántica: Visitor y tabla de símbolos
    tabla.py                   # Estructura Tabla, lector y escritor CSV
    operaciones.py             # Operaciones de datos y estadística descriptiva
    regresion.py               # Regresión lineal simple y múltiple
    arbol_decision.py          # Árbol de decisión CART para clasificación
    manejador_errores.py       # Errores léxicos y sintácticos en español
    validador.py               # Validador interactivo de sintaxis
  ejemplos/                    # Programas de ejemplo (.terra)
  datos/                       # Conjuntos de datos
  salidas/                     # Archivos generados por los ejemplos
  tests/
    test_terra.py              # Pruebas del primer corte
    test_corte2.py             # Pruebas del segundo corte
    test_arbol_decision.py     # Pruebas del árbol de decisión
    test_regresion.py          # Pruebas de la regresión lineal
  docs/                        # Informes y gramática BNF
  generar.sh                   # Regenera lexer, parser y visitor
  arbol.sh                     # Muestra el árbol de análisis
  requisitos.txt               # Dependencias
```

La arquitectura separa las responsabilidades:

| Módulo | Responsabilidad |
|---|---|
| `Terra.g4` | Sintaxis: qué programas son válidos |
| `visitante_evaluador.py` | Semántica: recorre el árbol, mantiene la tabla de símbolos y valida tipos |
| `operaciones.py` | Manipulación de tablas; no conoce el árbol de ANTLR |
| `tabla.py` | Representación de los datos y lectura/escritura de archivos |
| `regresion.py` | Regresión lineal simple y múltiple por mínimos cuadrados |
| `arbol_decision.py` | Árbol de decisión CART con criterio de Gini |

---

## Requisitos

- Python 3.11 o superior
- Java (solo para generar el parser con ANTLR)
- `antlr4-python3-runtime` y `pytest` (en `requisitos.txt`)

---

## Instalación

**1. Clonar el repositorio**

```bash
git clone <url-del-repositorio>
cd terra
```

**2. Crear y activar el entorno virtual**

```bash
python3 -m venv venv
source venv/bin/activate
```

Hay que repetir el `source` en cada terminal nueva.

**3. Instalar dependencias**

```bash
pip install -r requisitos.txt
```

**4. Generar el parser** (solo si se modifica la gramática)

```bash
chmod +x generar.sh arbol.sh
./generar.sh
```

---

## Uso

| Comando | Efecto |
|---|---|
| `python src/terra.py archivo.terra` | Ejecuta el programa |
| `python src/terra.py` | Modo interactivo (REPL) |
| `python src/terra.py archivo.terra --arbol` | Imprime el árbol de análisis |
| `python src/terra.py archivo.terra --tokens` | Lista los tokens reconocidos |
| `python src/validador.py` | Validador interactivo de construcciones |
| `./arbol.sh archivo.terra -gui` | Árbol en una ventana, exportable a PNG |
| `pytest tests/ -q` | Ejecuta todas las pruebas |

Terra solo ejecuta archivos con extensión `.terra`.

---

## El lenguaje

### Variables y tipos

```
fijo PI = 3.14159          # constante: no se puede reasignar
var contador = 0           # variable: se puede reasignar
contador = contador + 1

fijo nombre = "Colombia"
fijo activo = verdadero
fijo valores = [86.2, 92.5, 79.8]
fijo vacio = nulo
```

### Operadores

| Tipo | Operadores |
|---|---|
| Aritméticos | `+  -  *  /  %  ^` |
| Relacionales | `>  <  >=  <=  ==  !=` |
| Lógicos | `&&  \|\|  !` |
| Pertenencia | `en` (por ejemplo `x en [1, 2, 3]`) |

### Control de flujo y funciones

```
si co2 > 80:
    mostrar("alto")
sino si co2 > 50:
    mostrar("medio")
sino:
    mostrar("bajo")
fin

para valor en [1, 2, 3]:
    mostrar(valor)
fin

funcion per_capita(emisiones, habitantes):
    retornar emisiones * 1000000 / habitantes
fin
```

Los bloques se cierran con `fin`.

---

## Manipulación de listas

Terra incluye tres funciones incorporadas para modificar listas. Todas devuelven una **lista nueva** sin mutar la original; el patrón correcto es reasignar la variable con `var`.

### `agregar(lista, elemento)`

Añade un elemento al final de la lista.

```
var nums = [1, 2, 3]
nums = agregar(nums, 4)
mostrar(nums)              # [1, 2, 3, 4]

var palabras = []
palabras = agregar(palabras, "bosque")
palabras = agregar(palabras, "oceano")
mostrar(palabras)          # [bosque, oceano]
```

### `insertar(lista, indice, elemento)`

Inserta un elemento en la posición indicada. Los elementos siguientes se desplazan. El índice puede ser `0` (inicio) hasta `longitud(lista)` (equivale a agregar al final).

```
var nums = [1, 3, 4]
nums = insertar(nums, 1, 2)
mostrar(nums)              # [1, 2, 3, 4]

nums = insertar(nums, 0, 0)
mostrar(nums)              # [0, 1, 2, 3, 4]
```

### `quitar(lista, indice)`

Elimina el elemento en la posición indicada y devuelve la lista resultante.

```
var nums = [1, 2, 3, 4]
nums = quitar(nums, 0)
mostrar(nums)              # [2, 3, 4]

nums = quitar(nums, longitud(nums) - 1)
mostrar(nums)              # [2, 3]
```

**Errores comunes:**

```
# indice fuera de rango
nums = insertar([1, 2], 5, 9)   # Error: indice 5 fuera de rango
nums = quitar([1, 2], 5)         # Error: indice 5 fuera de rango
```

---

## Trabajo con datos

### Cargar

```
fijo datos = cargar "datos/emisiones_ejemplo.csv"
fijo otros = cargar "datos/archivo.csv" con separador ";" con encabezado falso
```

- El separador por defecto es la coma y el encabezado está activado.
- Sin encabezado, las columnas se llaman `col1`, `col2`, ...
- El tipo de cada columna se infiere automáticamente: booleano, número, decimal o texto.
- Las celdas vacías, `NA`, `null` o `nulo` se leen como `nulo`.

### Tuberías

Cada operación recibe la tabla anterior y produce una nueva:

| Operación | Ejemplo |
|---|---|
| Seleccionar columnas | `\|> seleccionar [pais, anio, co2]` |
| Filtrar filas | `\|> filtrar donde anio >= 2018 && co2 > 50` |
| Crear columna calculada | `\|> crear total = co2 * 2` |
| Renombrar | `\|> renombrar energia_renovable como renovable` |
| Ordenar | `\|> ordenar por co2 desc` |
| Eliminar duplicados | `\|> eliminar duplicados` |
| Eliminar nulos | `\|> eliminar nulos` o `\|> eliminar nulos en [co2]` |
| Rellenar nulos | `\|> rellenar nulos en energia_renovable con 0.0` |
| Convertir tipo | `\|> convertir anio como texto` |
| Limitar filas | `\|> limitar 10` |
| Agrupar y resumir | `\|> agrupar por [pais] \|> resumir total = suma(co2)` |

### Agrupar y resumir

```
fijo por_pais = datos
    |> agrupar por [pais]
    |> resumir co2_acumulado = suma(co2),
               promedio      = media(co2),
               registros     = contar()
    |> ordenar por co2_acumulado desc
```

### Describir

```
mostrar(describir(datos))
```

Devuelve una tabla con una fila por columna: conteo, nulos, media, desviación, mínimo, mediana y máximo para columnas numéricas; valores distintos y el más frecuente para columnas de texto.

### Guardar

```
guardar resumen como "salidas/resumen.csv"
```

---

## Regresión lineal

Terra incluye un módulo de regresión lineal implementado desde cero con el método de mínimos cuadrados. No usa numpy ni sklearn.

### Regresión simple — `regresion_lineal(xs, ys)`

Ajusta el modelo **y = a + b·x** sobre dos listas de igual longitud.

```
fijo anios = [2015, 2018, 2021, 2023]
fijo co2   = [86.2, 92.5, 79.8, 88.1]

fijo modelo = regresion_lineal(anios, co2)
mostrar(modelo)
```

Salida:

```
Modelo lineal simple (n=4)
  y = -1697.405 + 0.8825 * x
  R^2 = 0.1026
```

### Regresión múltiple — `regresion_multiple(X, ys)`

Ajusta el modelo **y = b0 + b1·x1 + b2·x2 + ...** donde `X` es una lista de listas (cada sublista es una observación).

```
fijo X = [
    [2015, 46700000],
    [2018, 48300000],
    [2021, 51500000],
    [2023, 52300000]
]

fijo modelo2 = regresion_multiple(X, co2)
mostrar(modelo2)
```

### Funciones de regresión

| Función | Descripción |
|---|---|
| `regresion_lineal(xs, ys)` | Entrena modelo simple; devuelve el modelo |
| `regresion_multiple(X, ys)` | Entrena modelo múltiple; devuelve el modelo |
| `predecir(modelo, x)` | Aplica el modelo simple a un valor de x |
| `predecir_multiple(modelo, [x1, x2, ...])` | Aplica el modelo múltiple a un vector |
| `coeficientes(modelo)` | Devuelve `[intercepto, pendiente]` o `[b0, b1, ...]` |
| `r_cuadrado(modelo)` | Devuelve el coeficiente R² (0 a 1) |

```
fijo modelo = regresion_lineal([1, 2, 3, 4], [2, 4, 5, 4])

mostrar("Coeficientes:", coeficientes(modelo))
mostrar("R^2:",          r_cuadrado(modelo))
mostrar("Pred. x=5:",    predecir(modelo, 5))
```

**Errores detectados:**

```
# Menos de 2 observaciones
regresion_lineal([1], [2])                 # Error

# Listas de distinta longitud
regresion_lineal([1, 2, 3], [4, 5])       # Error

# Todos los valores de X son iguales (matriz singular)
regresion_lineal([5, 5, 5], [1, 2, 3])    # Error
```

---

## Árbol de decisión

Terra incluye un árbol de decisión CART para clasificación, implementado desde cero con el índice de Gini como criterio de división. No usa sklearn.

### Entrenar — `arbol_decision(X, ys)`

```
fijo X = [
    [1.68, 26.4],
    [1.91, 27.1],
    [4.41, 31.2],
    [4.72, 34.5],
    [2.38, 22.1],
    [2.42, 23.4]
]
fijo y = ["moderada", "moderada", "alta", "alta", "media", "media"]

fijo modelo = arbol_decision(X, y, 3)
mostrar(modelo)
```

Salida:

```
Arbol de decision  (n=6, p=2, clases=['alta', 'media', 'moderada'])
  Profundidad: 2  |  Nodos internos: 3  |  Hojas: 4
  Usa mostrar_arbol(modelo) para ver la estructura completa
```

### Visualizar la estructura — `mostrar_arbol(modelo)`

```
mostrar_arbol(modelo, ["co2_per_capita", "energia_renovable"])
```

Salida:

```
Arbol de decision (n=6, p=2, clases=['alta', 'media', 'moderada'])
Profundidad real: 2  |  Nodos internos: 3  |  Hojas: 4
------------------------------------------------------------
├── [co2_per_capita <= 2.42] (n=6)
│   ├── [co2_per_capita <= 1.91] (n=4)
│   │   ├── HOJA → moderada [moderada:2] (n=2)
│   │   └── HOJA → media [media:2] (n=2)
│   └── HOJA → alta [alta:2] (n=2)
```

### Clasificar observaciones nuevas

```
# Una observacion
fijo clase = clasificar(modelo, [1.75, 27.0])
mostrar(clase)                              # moderada

# Varias observaciones a la vez
fijo clases = clasificar_lote(modelo, [
    [1.75, 27.0],
    [4.50, 35.0],
    [2.30, 24.0]
])
mostrar(clases)                             # [moderada, alta, media]
```

### Evaluar el modelo

```
fijo resultado = evaluar_arbol(modelo, X, y)
mostrar(resultado)
mostrar("Exactitud:", exactitud(resultado))
```

Salida:

```
Evaluacion: exactitud=100.0%  (6/6)
Exactitud: 1.0
```

### Funciones del árbol de decisión

| Función | Descripción |
|---|---|
| `arbol_decision(X, ys)` | Entrena con parámetros por defecto (prof. 5, min. 2 muestras) |
| `arbol_decision(X, ys, max_prof)` | Entrena con profundidad máxima indicada |
| `arbol_decision(X, ys, max_prof, min_muestras)` | Entrena con ambos hiperparámetros |
| `clasificar(modelo, [x1, x2, ...])` | Clasifica una observación |
| `clasificar_lote(modelo, X)` | Clasifica una lista de observaciones |
| `evaluar_arbol(modelo, X, ys)` | Calcula exactitud y matriz de confusión |
| `exactitud(resultado)` | Extrae la exactitud del resultado de `evaluar_arbol` |
| `mostrar_arbol(modelo)` | Imprime la estructura del árbol |
| `mostrar_arbol(modelo, nombres)` | Imprime el árbol con nombres de columnas |

**Errores detectados:**

```
# X vacío
arbol_decision([], [])                        # Error

# X e y de distinto tamaño
arbol_decision([[1.0]], ["a", "b"])           # Error

# X no es lista de listas
arbol_decision([1.0, 2.0], ["a", "b"])        # Error

# Hiperparámetro inválido
arbol_decision(X, y, 0)                       # Error: max_profundidad >= 1

# Número incorrecto de predictores al clasificar
clasificar(modelo, [1.0])                     # Error si el modelo espera 2
```

---

## Funciones incorporadas — referencia completa

### Matemáticas

| Función | Resultado |
|---|---|
| `raiz(x)` | Raíz cuadrada |
| `absoluto(x)` | Valor absoluto |
| `redondear(x)` | Redondeo al entero más cercano |
| `redondear(x, decimales)` | Redondeo con precisión |

### Estadística sobre listas

| Función | Resultado |
|---|---|
| `suma(lista)` | Suma de los valores |
| `media(lista)` | Promedio |
| `mediana(lista)` | Valor central |
| `minimo(lista)` / `maximo(lista)` | Extremos |
| `desviacion(lista)` | Desviación estándar muestral |
| `longitud(lista)` | Número de elementos |

### Tablas

| Función | Resultado |
|---|---|
| `contar()` | Filas del grupo (solo en `resumir`) |
| `contar(lista)` | Número de elementos no nulos |
| `describir(tabla)` | Resumen estadístico por columna |

### Listas

| Función | Resultado |
|---|---|
| `agregar(lista, elemento)` | Lista nueva con el elemento al final |
| `insertar(lista, indice, elemento)` | Lista nueva con el elemento en la posición indicada |
| `quitar(lista, indice)` | Lista nueva sin el elemento en esa posición |

### Conversión de tipos

| Función | Resultado |
|---|---|
| `a_numero(valor)` | Convierte a número entero |
| `a_decimal(valor)` | Convierte a decimal |
| `a_texto(valor)` | Convierte a texto |

### Regresión lineal

| Función | Resultado |
|---|---|
| `regresion_lineal(xs, ys)` | Modelo simple entrenado |
| `regresion_multiple(X, ys)` | Modelo múltiple entrenado |
| `predecir(modelo, x)` | Valor predicho (simple) |
| `predecir_multiple(modelo, xs)` | Valor predicho (múltiple) |
| `coeficientes(modelo)` | Lista de coeficientes del modelo |
| `r_cuadrado(modelo)` | Coeficiente de determinación R² |

### Árbol de decisión

| Función | Resultado |
|---|---|
| `arbol_decision(X, ys, ...)` | Modelo CART entrenado |
| `clasificar(modelo, x)` | Clase predicha para una observación |
| `clasificar_lote(modelo, X)` | Lista de clases predichas |
| `evaluar_arbol(modelo, X, ys)` | Diccionario con exactitud y matriz de confusión |
| `exactitud(resultado)` | Decimal entre 0 y 1 |
| `mostrar_arbol(modelo, ...)` | Imprime la estructura del árbol |

---

## Decisiones de diseño

- **Las tuberías no modifican la tabla original.** Cada operación devuelve una tabla nueva.
- **Las funciones de lista devuelven copias.** `agregar`, `insertar` y `quitar` siguen el mismo principio funcional que el resto del lenguaje: sin mutación implícita.
- **Nulos estrictos en operaciones, ignorados en agregaciones.** `co2 + 1` con un `co2` nulo es un error de tipos; `media(co2)` ignora los nulos.
- **El tipo se infiere por columna, no por celda.**
- **Regresión implementada con álgebra matricial propia.** La regresión múltiple resuelve β = (XᵀX)⁻¹Xᵀy con eliminación de Gauss-Jordan.
- **El árbol usa CART con índice de Gini.** Soporta columnas numéricas y de texto. Los nulos en predictores se envían a la rama derecha por convención.
- **Modelos como diccionarios.** Tanto el modelo de regresión como el árbol de decisión son diccionarios Terra (tipo `desconocido` internamente, con clave `"tipo"`), lo que los hace inspeccionables y pasables como argumentos a cualquier función.

---

## Errores

Terra informa cada error con su línea y un mensaje en español:

```
Error en evaluacion: [linea 2] la columna 'temperatura' no existe
Error en evaluacion: [linea 3] la condicion de 'filtrar' debe ser booleana
Error en evaluacion: [linea 5] indice 5 fuera de rango (longitud 3)
Error en evaluacion: [linea 7] regresion_lineal(): todos los valores de X son iguales
Error en evaluacion: [linea 9] arbol_decision(): X debe ser una lista de listas
Error en evaluacion: [linea 11] clasificar() espera 2 predictores, se recibieron 1
```

---

## Ejemplos

| Archivo | Contenido |
|---|---|
| `01_variables.terra` | Variables y tipos |
| `02_operadores.terra` | Operadores y precedencia |
| `03_control.terra` | Condicionales y ciclos |
| `04_carga_seleccion.terra` | Carga y selección |
| `05_indicadores_ambientales.terra` | Indicadores con listas |
| `06_errores.terra` | Programa con errores intencionales |
| `07_carga_guardado.terra` | Cargar y guardar CSV |
| `08_tuberias.terra` | Preparación de datos con tuberías |
| `09_funciones.terra` | Funciones definidas por el usuario |
| `10_agrupacion.terra` | Agrupamiento y estadísticas por grupo |
| `11_describir.terra` | Resumen descriptivo |
| `12_opciones_carga.terra` | Separador y encabezado configurables |
| `13_caso_owid.terra` | Caso de estudio con datos reales (50 000+ filas) |
| `14_regresion.terra` | Regresión lineal simple y múltiple |
| `15_clasificacion.terra` | Árbol de decisión para clasificación de emisiones |

```bash
python src/terra.py ejemplos/14_regresion.terra
python src/terra.py ejemplos/15_clasificacion.terra
```

---

## Caso de estudio: emisiones de CO₂

`13_caso_owid.terra` procesa el conjunto público de emisiones de CO₂ de **Our World in Data** (50 411 filas, 79 columnas, 254 países y regiones).

`14_regresion.terra` y `15_clasificacion.terra` amplían ese análisis con modelos predictivos:

- `14_regresion.terra` ajusta una recta sobre la evolución temporal de emisiones y predice valores futuros.
- `15_clasificacion.terra` entrena un árbol para clasificar países por nivel de intensidad de carbono a partir de sus indicadores.

Para descargar el dataset actualizado:

```bash
curl -L -o datos/owid-co2-data.csv \
  https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv
```

Fuente: Our World in Data, *CO₂ and Greenhouse Gas Emissions* (https://github.com/owid/co2-data).

---

## Pruebas

```bash
pytest tests/ -q                         # todas
pytest tests/test_terra.py -v            # primer corte
pytest tests/test_corte2.py -v           # segundo corte (datos y tuberías)
pytest tests/test_regresion.py -v        # regresión lineal
pytest tests/test_arbol_decision.py -v   # árbol de decisión
```

| Suite | Pruebas | Qué cubre |
|---|---|---|
| `test_terra.py` | ~100 | Léxico, sintaxis, tipos, operadores, control de flujo, funciones |
| `test_corte2.py` | 53 | CSV, cargar/guardar, todas las operaciones de tubería, agrupar/resumir, describir |
| `test_regresion.py` | 10 | Coeficientes, R², predicción, errores de entrada |
| `test_arbol_decision.py` | 24 | Entrenamiento, clasificación simple y en lote, evaluación, errores |

---

## Estado por corte

| Corte | Contenido | Estado |
|---|---|---|
| Primero | Gramática, lexer y parser, variables, expresiones, control de flujo, funciones, errores léxicos y sintácticos | Completo |
| Segundo | Tabla y CSV desde cero, tuberías, agrupamiento, agregaciones, `describir`, errores semánticos, caso con datos reales | Completo |
| Tercero | Manipulación de listas (`agregar`, `insertar`, `quitar`), regresión lineal simple y múltiple, árbol de decisión CART | Completo |

`graficar` se reconoce sintácticamente; su ejecución (exportación a PNG) queda pendiente.