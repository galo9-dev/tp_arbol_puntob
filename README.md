# TP 5 — Árbol de decisión, Punto B: Predicción de abandono en estudiantes terciarios

### Materia: Procesamiento de Aprendizaje Automático

---

## Descripción

El Instituto Tecnológico Beltrán busca identificar a los estudiantes con riesgo de abandonar la carrera para implementar estrategias de acompañamiento. En este trabajo se construye un **modelo de árbol de decisión** que, a partir de los datos de cursada del primer cuatrimestre, predice si un estudiante **abandonará** o **continuará** la carrera.

El programa entrena el modelo con datos históricos (donde se conoce el resultado), lo evalúa con una parte de esos datos que el modelo no vio, interpreta qué variables usó para decidir y, por último, predice el resultado de un grupo de estudiantes nuevos.

## Archivos del proyecto

| Archivo | Descripción |
|---|---|
| `modelo_abandono.py` | Script principal: preparación de datos, entrenamiento, evaluación, interpretación y predicción. |
| `TablaPrediccionAbandono-Entrenamiento_OG.xlsx` | Datos históricos de 100 estudiantes **con** la columna `EstadoFinal` (Abandonó / Continúa). Se usa para entrenar y evaluar el modelo. |
| `TablaPrediccionAbandono-DatosFinal.xlsx` | Datos de 100 estudiantes **sin** `EstadoFinal`. Son los casos sobre los que el modelo hace su predicción. |

**Archivos que genera el programa al ejecutarse:**

| Archivo | Descripción |
|---|---|
| `predicciones_DatosFinal.xlsx` | Los estudiantes de DatosFinal con dos columnas nuevas: `Prediccion` y `ProbAbandono`, ordenados de mayor a menor riesgo. |
| `arbol_decision.png` | Gráfico del árbol de decisión entrenado. |
| `matriz_confusion.png` | Matriz de confusión sobre el conjunto de prueba. |
| `importancia_variables.png` | Gráfico de barras con la importancia de cada variable usada por el árbol. |

## Variables del dataset

| Variable original | Nombre en el código | Tipo |
|---|---|---|
| `edad` | `edad` | Numérica |
| `genero` | `genero` | Categórica (m / f) |
| `carrera` | `carrera` | Categórica (9 tecnicaturas) |
| `PromedioPrimerCuatrimestre` | `promedio` | Numérica |
| `CantMateriasAprobadasPrimerCuatrimestre` | `aprobadas` | Numérica |
| `CantMateriasDesaprobadasPrimerCuatrimestre` | `desaprobadas` | Numérica |
| `AsistenciaPromedio(%)` | `asistencia` | Numérica |
| `trabaja/NoTrabaja` | `trabaja` | Categórica (Si / No) |
| `DistanciaDomicilioAlInstituto(Kms)` | `distancia_km` | Numérica |
| `ActividadesExtracurriculares(Estudio)` | `extracurriculares` | Categórica (Si / No) |
| `EstadoFinal` | `estado_final` | **Variable objetivo** (Abandonó / Continúa) |

## Qué hace el código

El script está dividido en seis secciones:

### 1. Carga y preparación de los datos

- Lee los dos archivos Excel.
- **Renombra las columnas** a nombres cortos, sin paréntesis, barras ni símbolos.
- **Limpia el texto**: elimina espacios de más y corrige un error de tipeo en la columna `carrera` ("TÉCNICATURA" en lugar de "TECNICATURA").
- **Codifica las variables categóricas**, porque scikit-learn solo trabaja con números:
  - Las binarias (`genero`, `trabaja`, `extracurriculares`) se pasan a 0 / 1.
  - `carrera` se transforma con **One-Hot Encoding**: una columna 0/1 por cada carrera, ya que las carreras no tienen un orden entre sí.
  - La variable objetivo se codifica como **1 = Abandonó** (el caso que interesa detectar) y **0 = Continúa**.
- Aplica **exactamente la misma preparación a los dos archivos**, para que el modelo pueda leer DatosFinal con las mismas columnas y en el mismo orden que usó para entrenar.
- Verifica que no queden valores sin codificar.

### 2. División en entrenamiento y prueba

El archivo de entrenamiento se divide en **70% para entrenar** y **30% para evaluar** con `train_test_split`. Se usa `stratify` para que ambas partes tengan la misma proporción de abandonos, y una semilla fija (`SEMILLA = 42`) para que los resultados sean reproducibles.

DatosFinal **no** se usa para evaluar, porque no tiene la respuesta real con la cual comparar.

### 3. Entrenamiento del árbol

Se entrena un `DecisionTreeClassifier` con criterio **Gini** y dos parámetros que limitan su crecimiento para evitar el **sobreajuste** (que el árbol memorice los datos en lugar de aprender patrones):

- `max_depth = 4`: profundidad máxima del árbol.
- `min_samples_leaf = 5`: mínimo de estudiantes en cada hoja.

### 4. Evaluación

Sobre el conjunto de prueba se calculan:

- **Accuracy** en entrenamiento y en prueba (si la primera es mucho mayor que la segunda, hay sobreajuste).
- **Matriz de confusión**, para ver qué tipo de errores comete el modelo.
- **Precisión, recall y F1** para cada clase.

### 5. Interpretación (Punto 4 del TP)

Esta sección responde qué variables parecen ser más importantes para predecir el abandono, y si esa respuesta es confiable:

- **Importancia de variables**: cuánto contribuyó cada variable a reducir la impureza (Gini) en los cortes del árbol. Los valores suman 1; las variables con 0 no se usaron. Se imprimen en consola y se grafican en `importancia_variables.png`.
- **Reglas del árbol** en formato texto, para leer los cortes que hizo. La variable de la **raíz** (primer corte) es la que mejor separa a los alumnos al principio. También se guarda el gráfico del árbol completo (`arbol_decision.png`).
- **Comparación con un modelo trivial** que siempre predice la clase más frecuente. Si el árbol no lo supera, sus importancias no tienen valor predictivo.
- **Prueba de estabilidad**: se reentrena el árbol con 5 divisiones train/test distintas y se muestran las 3 variables principales de cada una. Si una variable fuera realmente importante, debería aparecer siempre.
- Al final se imprime una **interpretación automática** que resume todo lo anterior y saca una conclusión según los resultados.

### 6. Predicción sobre DatosFinal

El modelo entrenado predice el estado de los 100 estudiantes nuevos y calcula la **probabilidad de abandono** de cada uno. El resultado se guarda en `predicciones_DatosFinal.xlsx`, ordenado de mayor a menor riesgo.

## Cómo ejecutarlo

**1. Instalar las librerías**

```
pip install pandas scikit-learn matplotlib openpyxl
```

> Si hay más de una versión de Python instalada, conviene instalar con el mismo Python que ejecuta el script, por ejemplo: `py -3.14 -m pip install pandas scikit-learn matplotlib openpyxl`

**2. Ubicar los archivos**

Los dos Excel tienen que estar en la **misma carpeta** que `modelo_abandono.py`, con los nombres indicados en las variables `ARCHIVO_ENTRENAMIENTO` y `ARCHIVO_NUEVOS` al principio del script.

```
tp_arbol_puntob/
├── modelo_abandono.py
├── TablaPrediccionAbandono-Entrenamiento_OG.xlsx
└── TablaPrediccionAbandono-DatosFinal.xlsx
```

**3. Ejecutar**

```
python modelo_abandono.py
```

Los resultados se muestran en la consola y los archivos generados se guardan en la misma carpeta.

## Librerías utilizadas

| Librería | Uso |
|---|---|
| **pandas** | Carga de los Excel, limpieza, codificación y exportación de las predicciones. |
| **scikit-learn** | División train/test, árbol de decisión, reglas del árbol y métricas de evaluación. |
| **matplotlib** | Gráficos del árbol y de la matriz de confusión. |
| **openpyxl** | Lectura y escritura de archivos `.xlsx` (la usa pandas internamente). |

## Resultados obtenidos

| Métrica | Valor |
|---|---|
| Accuracy en entrenamiento | 0,80 |
| Accuracy en prueba | 0,50 |
| Modelo trivial (predecir siempre "Abandonó") | 0,53 |
| Validación cruzada (5 particiones, distintas profundidades) | entre 0,39 y 0,46 |

**Predicción sobre DatosFinal:** 73 estudiantes clasificados como "Abandonó" y 27 como "Continúa".

### Variables más importantes (Punto 4)

| Variable | Importancia |
|---|---|
| asistencia | 0,325 |
| edad | 0,274 |
| carrera: Administración de Pymes | 0,151 |
| desaprobadas | 0,130 |
| promedio | 0,120 |

El primer corte del árbol (raíz) se hace por **asistencia**. El resto de las variables (género, trabaja, distancia, extracurriculares, aprobadas y las demás carreras) no fueron usadas.

Sin embargo, la **prueba de estabilidad** muestra que este ranking no se sostiene:

| División | Accuracy en prueba | 3 variables principales |
|---|---|---|
| 0 | 0,57 | aprobadas, extracurriculares, edad |
| 1 | 0,43 | carrera Adm. de Pymes, edad, carrera Radiología |
| 2 | 0,33 | promedio, edad, género |
| 3 | 0,67 | aprobadas, carrera Ciencia de Datos e IA, extracurriculares |
| 4 | 0,37 | edad, promedio, asistencia |

En cada división la variable principal es distinta, y la accuracy varía entre 0,33 y 0,67 solo por cambiar qué alumnos quedan en entrenamiento y cuáles en prueba. Ninguna variable aparece de forma consistente como la más importante. Edad es la que más se repite entre las tres primeras, pero el árbol que la usa tampoco supera al modelo trivial.


### Interpretación general

El modelo **no logra predecir mejor que el azar**: su accuracy en prueba (50%) es menor que la de un modelo trivial que siempre predice "Abandonó" (53%). La gran diferencia entre entrenamiento (80%) y prueba (50%) indica que el árbol memorizó ruido de los datos de entrenamiento.

Para descartar que fuera un problema del modelo o de la división elegida, se probaron otras profundidades del árbol, un Random Forest y una regresión logística con validación cruzada. Todos obtuvieron resultados similares o peores, lo que indica que **las variables del dataset no muestran relación con el abandono**. Esto también se observa al comparar los promedios de cada variable entre quienes abandonaron y quienes continuaron: las diferencias son mínimas.

Además, las variables "importantes" cambian notablemente al modificar una sola fila del dataset o al cambiar la división train/test, lo que confirma que el árbol no encontró patrones estables. Por lo tanto, **las predicciones sobre DatosFinal no deben considerarse confiables**.
