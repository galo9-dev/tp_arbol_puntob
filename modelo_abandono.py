"""
Predicción de abandono en estudiantes terciarios - Árbol de decisión
Instituto Tecnológico Beltrán

Pasos:
  1. Cargar y preparar los datos (Entrenamiento y DatosFinal)
  2. Dividir Entrenamiento en train / test
  3. Entrenar el árbol de decisión
  4. Evaluar con el conjunto de prueba
  5. Interpretar: importancia de variables, árbol y su confiabilidad (Punto 4)
  6. Predecir sobre DatosFinal (alumnos sin resultado conocido)
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

# Trabaja siempre en la carpeta del script (lee y guarda los archivos ahí)
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def buscar_excel(palabra):
    """Busca en la carpeta un .xlsx cuyo nombre contenga la palabra."""
    encontrados = [f for f in glob.glob("*.xlsx")
                   if palabra.lower() in f.lower()
                   and not f.startswith("predicciones")
                   and not f.startswith("~$")]  # archivos temporales de Excel abierto
    if not encontrados:
        raise FileNotFoundError(f"No hay ningún .xlsx con '{palabra}' en {os.getcwd()}")
    print(f"Usando: {encontrados[0]}")
    return encontrados[0]


ARCHIVO_ENTRENAMIENTO = buscar_excel("Entrenamiento")
ARCHIVO_NUEVOS = buscar_excel("DatosFinal")
SEMILLA = 42  # fija el azar para que todos en el equipo obtengan el mismo resultado


# =============================================================
# 1. CARGA Y PREPARACIÓN
# =============================================================

# Nombres cortos y sin símbolos raros (paréntesis, barras, %)
NOMBRES = {
    "edad": "edad",
    "genero": "genero",
    "carrera": "carrera",
    "PromedioPrimerCuatrimestre": "promedio",
    "CantMateriasAprobadasPrimerCuatrimestre": "aprobadas",
    "CantMateriasDesaprobadasPrimerCuatrimestre": "desaprobadas",
    "AsistenciaPromedio(%)": "asistencia",
    "trabaja/NoTrabaja": "trabaja",
    "DistanciaDomicilioAlInstituto(Kms)": "distancia_km",
    "ActividadesExtracurriculares(Estudio)": "extracurriculares",
    "EstadoFinal": "estado_final",
}


def preparar(df):
    """Limpia y codifica un DataFrame. Se aplica IGUAL a los dos archivos."""
    df = df.rename(columns=NOMBRES).copy()

    # Limpieza de texto: espacios de más y el error de tipeo "TÉCNICATURA"
    for col in ["genero", "carrera", "trabaja", "extracurriculares"]:
        df[col] = df[col].astype(str).str.strip()
    df["carrera"] = (df["carrera"]
                     .str.replace("TÉCNICATURA", "TECNICATURA")
                     .str.replace("TECNICATURA SUPERIOR EN ", ""))

    # Categóricas binarias -> 0/1
    df["genero"] = df["genero"].map({"m": 0, "f": 1})
    df["trabaja"] = df["trabaja"].map({"No": 0, "Si": 1})
    df["extracurriculares"] = df["extracurriculares"].map({"No": 0, "Si": 1})

    # Variable objetivo (solo existe en el archivo de entrenamiento)
    # 1 = Abandonó (es el caso que queremos detectar), 0 = Continúa
    if "estado_final" in df.columns:
        df["estado_final"] = df["estado_final"].map({"Abandonó": 1, "Continúa": 0})

    return df


datos = preparar(pd.read_excel(ARCHIVO_ENTRENAMIENTO, sheet_name=0))
nuevos = preparar(pd.read_excel(ARCHIVO_NUEVOS, sheet_name=0))

# Carrera tiene varias categorías sin orden -> One-Hot Encoding
# (una columna 0/1 por carrera). Se calcula sobre los dos archivos
# juntos para que ambos terminen con exactamente las mismas columnas.
carreras = sorted(set(datos["carrera"]) | set(nuevos["carrera"]))
datos["carrera"] = pd.Categorical(datos["carrera"], categories=carreras)
nuevos["carrera"] = pd.Categorical(nuevos["carrera"], categories=carreras)
datos = pd.get_dummies(datos, columns=["carrera"], prefix="carr", dtype=int)
nuevos = pd.get_dummies(nuevos, columns=["carrera"], prefix="carr", dtype=int)

# Control: no debería quedar ningún nulo (si aparece, el map no reconoció algún valor)
assert datos.isna().sum().sum() == 0, "Hay valores sin codificar en Entrenamiento"
assert nuevos.isna().sum().sum() == 0, "Hay valores sin codificar en DatosFinal"

X = datos.drop(columns="estado_final")  # variables predictoras
y = datos["estado_final"]               # lo que queremos predecir
X_nuevos = nuevos[X.columns]            # mismas columnas, mismo orden

print("=== DATOS ===")
print(f"Filas de entrenamiento: {len(datos)} | Alumnos a predecir: {len(nuevos)}")
print("Distribución de la clase (1=Abandonó, 0=Continúa):")
print(y.value_counts().to_string(), "\n")


# =============================================================
# 2. DIVISIÓN TRAIN / TEST
# =============================================================
# 70% para entrenar, 30% para evaluar.
# stratify=y mantiene la misma proporción de abandonos en ambas partes.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEMILLA, stratify=y)


# =============================================================
# 3. ENTRENAMIENTO
# =============================================================
# max_depth limita la profundidad para evitar sobreajuste
# (con solo 70 filas, un árbol sin límite se "memoriza" los datos).
# min_samples_leaf exige un mínimo de alumnos en cada hoja.
# -> Prueben cambiar estos valores y comparen resultados.
modelo = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=5,
    random_state=SEMILLA,
)
modelo.fit(X_train, y_train)


# =============================================================
# 4. EVALUACIÓN
# =============================================================
pred_train = modelo.predict(X_train)
pred_test = modelo.predict(X_test)

print("=== EVALUACIÓN ===")
print(f"Accuracy en entrenamiento: {accuracy_score(y_train, pred_train):.2f}")
print(f"Accuracy en prueba:        {accuracy_score(y_test, pred_test):.2f}")
print("(Si entrenamiento >> prueba, hay sobreajuste)\n")

print("Matriz de confusión (filas = real, columnas = predicho):")
print(pd.DataFrame(confusion_matrix(y_test, pred_test, labels=[0, 1]),
                   index=["Real: Continúa", "Real: Abandonó"],
                   columns=["Pred: Continúa", "Pred: Abandonó"]), "\n")

print(classification_report(y_test, pred_test,
                            target_names=["Continúa", "Abandonó"], zero_division=0))

ConfusionMatrixDisplay.from_predictions(
    y_test, pred_test, display_labels=["Continúa", "Abandonó"], cmap="Blues")
plt.title("Matriz de confusión - conjunto de prueba")
plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=150)
plt.close()


# =============================================================
# 5. INTERPRETACIÓN
# =============================================================
# Punto 4 del TP: ¿qué variables parecen ser más importantes para predecir el abandono?

# 5.1 Importancia de variables: cuánto reduce cada variable la impureza (Gini)
#     del árbol. Suman 1 en total; las que valen 0 no fueron usadas en ningún corte.
importancias = (pd.Series(modelo.feature_importances_, index=X.columns)
                .sort_values(ascending=False))
usadas = importancias[importancias > 0]
print("=== IMPORTANCIA DE VARIABLES ===")
print(usadas.round(3).to_string(), "\n")

# Gráfico de barras para el informe
plt.figure(figsize=(8, 0.5 * len(usadas) + 1.5))
usadas.sort_values().plot.barh(color="#4C72B0")
plt.xlabel("Importancia (reducción de Gini)")
plt.title("Importancia de variables en el árbol")
plt.tight_layout()
plt.savefig("importancia_variables.png", dpi=150)
plt.close()

# 5.2 Reglas del árbol: la variable del primer corte (raíz) es la que mejor
#     separa a los alumnos al principio.
print("=== REGLAS DEL ÁRBOL ===")
print(export_text(modelo, feature_names=list(X.columns)))

plt.figure(figsize=(22, 10))
plot_tree(modelo, feature_names=X.columns, class_names=["Continúa", "Abandonó"],
          filled=True, rounded=True, fontsize=9)
plt.title("Árbol de decisión - Predicción de abandono")
plt.tight_layout()
plt.savefig("arbol_decision.png", dpi=150)
plt.close()

# 5.3 ¿Son confiables estas importancias?
#     a) Se compara el modelo contra uno trivial que siempre predice la clase
#        más frecuente. Si no lo supera, el árbol no aprendió nada útil.
#     b) Se repite el entrenamiento con 5 divisiones train/test distintas.
#        Si las variables importantes fueran reales, deberían repetirse.
acc_test = accuracy_score(y_test, pred_test)
acc_trivial = max(y_test.mean(), 1 - y_test.mean())
raiz = X.columns[modelo.tree_.feature[0]]

print("=== ESTABILIDAD DE LAS VARIABLES IMPORTANTES ===")
top_por_division = []
for semilla in range(5):
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=semilla, stratify=y)
    arbol = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=SEMILLA).fit(Xa, ya)
    top = (pd.Series(arbol.feature_importances_, index=X.columns)
           .sort_values(ascending=False).index[:3].tolist())
    top_por_division.append(top[0])
    print(f"División {semilla}: accuracy prueba = {accuracy_score(yb, arbol.predict(Xb)):.2f} | "
          f"top 3 = {', '.join(top)}")
estable = len(set(top_por_division)) == 1

print("\n=== INTERPRETACIÓN (Punto 4) ===")
print(f"- Variables más importantes: {', '.join(usadas.index[:3])}.")
print(f"- Primer corte del árbol (raíz): '{raiz}'.")
print(f"- Accuracy en prueba: {acc_test:.2f} | Modelo trivial: {acc_trivial:.2f}")
if acc_test <= acc_trivial:
    print("- El árbol NO supera al modelo trivial: no aprendió patrones útiles.")
else:
    print("- El árbol supera al modelo trivial.")
if estable:
    print(f"- La variable principal es la misma en todas las divisiones ('{top_por_division[0]}').")
else:
    print("- La variable principal CAMBIA según cómo se dividen los datos: "
          "las importancias no son estables.")
if acc_test <= acc_trivial or not estable:
    print("=> Conclusión: no se puede afirmar que alguna variable sea realmente "
          "predictiva del abandono con estos datos.\n")
else:
    print("=> Conclusión: las variables anteriores parecen relevantes para predecir el abandono.\n")


# =============================================================
# 6. PREDICCIÓN SOBRE DatosFinal
# =============================================================
resultado = pd.read_excel(ARCHIVO_NUEVOS, sheet_name=0)  # datos originales, legibles
resultado["Prediccion"] = ["Abandonó" if p == 1 else "Continúa"
                           for p in modelo.predict(X_nuevos)]
resultado["ProbAbandono"] = modelo.predict_proba(X_nuevos)[:, 1].round(2)
resultado = resultado.sort_values("ProbAbandono", ascending=False)
resultado.to_excel("predicciones_DatosFinal.xlsx", index=False)

print("=== PREDICCIÓN SOBRE DatosFinal ===")
print(resultado["Prediccion"].value_counts().to_string())
print("\nArchivos generados: predicciones_DatosFinal.xlsx, arbol_decision.png, "
      "matriz_confusion.png, importancia_variables.png")
