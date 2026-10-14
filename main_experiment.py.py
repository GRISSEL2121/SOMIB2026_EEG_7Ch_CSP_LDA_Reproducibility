import os
import sys
import platform
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import sklearn

from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


# ==============================================================================
# 0. CONFIGURACIÓN GENERAL
# ==============================================================================

DATA_PATH = "."
RAW_FILENAME = "eeglab_data.set"
RANDOM_STATE = 42
N_SPLITS = 5

FULL_CONFIGURATION_NAME = "Full Channels (32 ch)"
REDUCED_CONFIGURATION_NAME = "Reduced Channels (7 ch)"

REDUCED_CHANNELS = ["C3", "Cz", "C4", "O1", "O2", "P3", "P4"]

OUTPUT_FOLD_RESULTS = "CrossValidation_Fold_Metrics.csv"
OUTPUT_SUMMARY_RESULTS = "Table1_Performance_Metrics.csv"
OUTPUT_LATEX_TABLE = "Table1_Performance_Metrics.tex"
OUTPUT_FIGURE = "Figure2_Metric_Distributions.tif"
OUTPUT_VERSIONS = "Software_Versions.txt"

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==============================================================================
# 1. FUNCIONES AUXILIARES
# ==============================================================================

def print_section(title: str) -> None:
    """Imprime un encabezado legible en consola."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_software_versions(filename: str) -> None:
    """Guarda versiones para documentar la reproducibilidad del experimento."""
    version_text = (
        f"Python: {sys.version}\n"
        f"Platform: {platform.platform()}\n"
        f"NumPy: {np.__version__}\n"
        f"Pandas: {pd.__version__}\n"
        f"MNE-Python: {mne.__version__}\n"
        f"scikit-learn: {sklearn.__version__}\n"
        f"Matplotlib: {plt.matplotlib.__version__}\n"
        f"Random state: {RANDOM_STATE}\n"
        f"Cross-validation folds: {N_SPLITS}\n"
    )

    with open(filename, "w", encoding="utf-8") as file:
        file.write(version_text)


def validate_required_channels(
    available_channels: list[str],
    required_channels: list[str]
) -> None:
    """Detiene el programa si falta algún canal de la topología propuesta."""
    missing_channels = [
        channel
        for channel in required_channels
        if channel not in available_channels
    ]

    if missing_channels:
        raise ValueError(
            "No se encontraron todos los canales requeridos para la "
            f"configuración reducida. Faltan: {missing_channels}"
        )


def percentage_mean_sd(values: np.ndarray) -> str:
    """Devuelve media ± SD en porcentaje, lista para tabla."""
    mean_value = np.mean(values) * 100
    sd_value = np.std(values, ddof=1) * 100
    return f"{mean_value:.2f} ± {sd_value:.2f}"


# ==============================================================================
# 2. CARGA DEL DATASET
# ==============================================================================

print_section("[PASO 1] Cargando el dataset")

raw_file = os.path.join(DATA_PATH, RAW_FILENAME)

if not os.path.exists(raw_file):
    raise FileNotFoundError(
        f"No se encontró el archivo '{raw_file}'. "
        "Verifica que eeglab_data.set esté en la carpeta de ejecución."
    )

raw = mne.io.read_raw_eeglab(
    raw_file,
    preload=True,
    verbose=False
)

print(f"Archivo cargado: {raw_file}")
print(f"Frecuencia de muestreo: {raw.info['sfreq']:.2f} Hz")
print(f"Número de canales originales: {len(raw.ch_names)}")
print(f"Duración: {raw.times[-1]:.2f} s")


# ==============================================================================
# 3. FILTRADO TEMPORAL
# ==============================================================================

print_section("[PASO 2] Aplicando filtro pasa-bandas de 1–35 Hz")

raw.filter(
    l_freq=1.0,
    h_freq=35.0,
    method="fir",
    fir_design="firwin",
    skip_by_annotation="edge",
    verbose=False
)

print("Filtrado terminado.")


# ==============================================================================
# 4. EXTRACCIÓN DE ÉPOCAS
# ==============================================================================

print_section("[PASO 3] Extrayendo épocas de rt y square")

events, event_id = mne.events_from_annotations(
    raw,
    verbose=False
)

required_events = ["rt", "square"]
missing_events = [
    event_name
    for event_name in required_events
    if event_name not in event_id
]

if missing_events:
    raise KeyError(
        "No se encontraron las anotaciones requeridas: "
        f"{missing_events}. Eventos disponibles: {event_id}"
    )

target_events = {
    "rt": event_id["rt"],
    "square": event_id["square"]
}

epochs = mne.Epochs(
    raw,
    events,
    event_id=target_events,
    tmin=-0.2,
    tmax=0.7,
    baseline=(-0.2, 0.0),
    proj=True,
    preload=True,
    reject_by_annotation=True,
    verbose=False
)

original_labels = epochs.events[:, -1]

# Convierte los códigos internos de eventos a etiquetas binarias 0 y 1.
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(original_labels)

class_counts = np.bincount(labels)

print(f"Total de épocas: {len(labels)}")
print(f"Clase 0: {class_counts[0]} épocas")
print(f"Clase 1: {class_counts[1]} épocas")
print(f"Correspondencia de clases: {label_encoder.classes_}")


# ==============================================================================
# 5. CONFIGURACIONES DE CANALES
# ==============================================================================

print_section("[PASO 4] Preparando configuraciones espaciales")

available_channels = epochs.ch_names
validate_required_channels(available_channels, REDUCED_CHANNELS)

configurations = {
    FULL_CONFIGURATION_NAME: available_channels,
    REDUCED_CONFIGURATION_NAME: REDUCED_CHANNELS
}

print(f"Canales disponibles: {len(available_channels)}")
print(f"Canales reducidos: {REDUCED_CHANNELS}")
print(
    "Reducción de canales: "
    f"{(1 - len(REDUCED_CHANNELS) / len(available_channels)) * 100:.1f}%"
)


# ==============================================================================
# 6. VALIDACIÓN CRUZADA Y MÉTRICAS
# ==============================================================================

print_section(
    "[PASO 5] Ejecutando CSP + LDA con validación cruzada sin leakage"
)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Al codificar las etiquetas como 0 y 1, se evalúa la clase 1 como positiva.
scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score,
        pos_label=1,
        zero_division=0
    ),
    "sensitivity": make_scorer(
        recall_score,
        pos_label=1,
        zero_division=0
    ),
    "f1_score": make_scorer(
        f1_score,
        pos_label=1,
        zero_division=0
    )
}

metric_names = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "f1_score": "F1-score"
}

# Guarda los vectores fold-by-fold.
results = {}

# Cada fila representa una configuración y un fold.
fold_rows = []

for configuration_name, selected_channels in configurations.items():

    print(f"\nProcesando: {configuration_name}")

    selected_epochs = epochs.copy().pick(selected_channels)
    X = selected_epochs.get_data(copy=True)

    # Se conservan cuatro componentes para ambas configuraciones,
    # equivalentes a dos pares de filtros espaciales.
    n_components = min(4, len(selected_channels))

    csp = CSP(
        n_components=n_components,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False,
        cov_est="concat",
        transform_into="average_power"
    )

    lda = LinearDiscriminantAnalysis(
        solver="svd"
    )

    # Es esencial colocar CSP dentro del pipeline.
    # De esta manera, CSP se ajusta exclusivamente con los datos
    # de entrenamiento de cada fold.
    pipeline = Pipeline([
        ("csp", csp),
        ("lda", lda)
    ])

    cv_results = cross_validate(
        estimator=pipeline,
        X=X,
        y=labels,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        error_score="raise",
        n_jobs=None
    )

    results[configuration_name] = {}

    for metric_key, metric_label in metric_names.items():
        fold_values = cv_results[f"test_{metric_key}"]
        results[configuration_name][metric_key] = fold_values

        mean_value = np.mean(fold_values) * 100
        sd_value = np.std(fold_values, ddof=1) * 100

        print(
            f"  {metric_label}: "
            f"{mean_value:.2f} ± {sd_value:.2f}%"
        )

    for fold_index in range(N_SPLITS):
        fold_rows.append({
            "Configuration": configuration_name,
            "Fold": fold_index + 1,
            "Accuracy": (
                results[configuration_name]["accuracy"][fold_index] * 100
            ),
            "Precision": (
                results[configuration_name]["precision"][fold_index] * 100
            ),
            "Sensitivity": (
                results[configuration_name]["sensitivity"][fold_index] * 100
            ),
            "F1-score": (
                results[configuration_name]["f1_score"][fold_index] * 100
            )
        })


# ==============================================================================
# 7. GUARDAR RESULTADOS POR FOLD
# ==============================================================================

print_section("[PASO 6] Guardando resultados por fold")

fold_dataframe = pd.DataFrame(fold_rows)

fold_dataframe.to_csv(
    OUTPUT_FOLD_RESULTS,
    index=False,
    float_format="%.4f"
)

print(f"Archivo creado: {OUTPUT_FOLD_RESULTS}")
print(fold_dataframe.to_string(index=False))


# ==============================================================================
# 8. CREAR TABLA RESUMEN CON MEDIA ± SD
# ==============================================================================

print_section("[PASO 7] Generando Tabla 1 con todas las métricas")

summary_rows = []

for configuration_name in configurations:

    summary_rows.append({
        "Spatial Configuration": configuration_name,
        "Accuracy (%)": percentage_mean_sd(
            results[configuration_name]["accuracy"]
        ),
        "Precision (%)": percentage_mean_sd(
            results[configuration_name]["precision"]
        ),
        "Sensitivity (%)": percentage_mean_sd(
            results[configuration_name]["sensitivity"]
        ),
        "F1-score (%)": percentage_mean_sd(
            results[configuration_name]["f1_score"]
        )
    })

summary_dataframe = pd.DataFrame(summary_rows)

summary_dataframe.to_csv(
    OUTPUT_SUMMARY_RESULTS,
    index=False
)

print(summary_dataframe.to_string(index=False))
print(f"\nArchivo creado: {OUTPUT_SUMMARY_RESULTS}")


# ==============================================================================
# 9. GENERAR TABLA LATEX
# ==============================================================================

latex_table = summary_dataframe.to_latex(
    index=False,
    escape=False,
    column_format="lcccc",
    caption=(
        "Cross-validation performance for the full and reduced "
        "channel configurations. Values are reported as mean "
        "$\\pm$ standard deviation across the five folds."
    ),
    label="tab:performance_metrics"
)

# Evita que el símbolo ± quede escrito como texto Unicode dentro de LaTeX.
latex_table = latex_table.replace("±", "$\\pm$")

with open(OUTPUT_LATEX_TABLE, "w", encoding="utf-8") as file:
    file.write(latex_table)

print(f"Tabla LaTeX creada: {OUTPUT_LATEX_TABLE}")


# ==============================================================================
# 10. GENERAR FIGURA CON LAS CUATRO MÉTRICAS
# ==============================================================================

print_section(
    "[PASO 8] Generando figura de distribución de las cuatro métricas"
)

metrics_order = [
    "accuracy",
    "precision",
    "sensitivity",
    "f1_score"
]

metrics_labels = [
    "Accuracy",
    "Precision",
    "Sensitivity",
    "F1-score"
]

full_data = [
    results[FULL_CONFIGURATION_NAME][metric]
    for metric in metrics_order
]

reduced_data = [
    results[REDUCED_CONFIGURATION_NAME][metric]
    for metric in metrics_order
]

fig, ax = plt.subplots(figsize=(7.5, 4.8))

metric_centers = np.arange(1, len(metrics_order) + 1)
offset = 0.17

full_positions = metric_centers - offset
reduced_positions = metric_centers + offset

full_boxplot = ax.boxplot(
    full_data,
    positions=full_positions,
    widths=0.28,
    patch_artist=True,
    manage_ticks=False,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": 4
    },
    medianprops={
        "linewidth": 1.6
    }
)

reduced_boxplot = ax.boxplot(
    reduced_data,
    positions=reduced_positions,
    widths=0.28,
    patch_artist=True,
    manage_ticks=False,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": 4
    },
    medianprops={
        "linewidth": 1.6
    }
)

# Colores diferenciados para las configuraciones.
for patch in full_boxplot["boxes"]:
    patch.set_facecolor("#DCEBFA")
    patch.set_edgecolor("#245A8D")

for element_name in ["whiskers", "caps", "medians"]:
    for element in full_boxplot[element_name]:
        element.set_color("#245A8D")

for patch in reduced_boxplot["boxes"]:
    patch.set_facecolor("#F3DFCA")
    patch.set_edgecolor("#A75D24")

for element_name in ["whiskers", "caps", "medians"]:
    for element in reduced_boxplot[element_name]:
        element.set_color("#A75D24")

ax.set_xticks(metric_centers)
ax.set_xticklabels(metrics_labels, fontsize=10)

ax.set_ylabel("Cross-validation score", fontsize=11)
ax.set_ylim(0.40, 1.02)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.45
)

# Leyenda usando rectángulos invisibles equivalentes a las cajas.
legend_handles = [
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#DCEBFA",
        edgecolor="#245A8D"
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#F3DFCA",
        edgecolor="#A75D24"
    )
]

ax.legend(
    legend_handles,
    [
        "Full Channels (32 ch)",
        "Reduced Channels (7 ch)"
    ],
    loc="lower left",
    frameon=True,
    fontsize=9
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight",
    format="tiff",
    pil_kwargs={"compression": "tiff_lzw"}
)

plt.close(fig)

print(f"Figura creada: {OUTPUT_FIGURE}")


# ==============================================================================
# 11. GUARDAR VERSIONES Y FINALIZAR
# ==============================================================================

save_software_versions(OUTPUT_VERSIONS)

print_section("[ÉXITO] Pipeline terminado")

print("Archivos generados:")
print(f"1. {OUTPUT_FOLD_RESULTS}")
print(f"2. {OUTPUT_SUMMARY_RESULTS}")
print(f"3. {OUTPUT_LATEX_TABLE}")
print(f"4. {OUTPUT_FIGURE}")
print(f"5. {OUTPUT_VERSIONS}")