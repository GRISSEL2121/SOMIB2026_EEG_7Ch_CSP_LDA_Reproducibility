# Reproducible Evaluation of a Fixed 7-Channel EEG Topology Using CSP–LDA

[![Python](https://img.shields.io/badge/Python-3.13.2-blue)](https://www.python.org/)
[![MNE](https://img.shields.io/badge/MNE--Python-1.12.1-blue)](https://mne.tools/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)

This repository contains the Python implementation accompanying the manuscript **“Evaluation of a Fixed 7-Channel EEG Topology Using Regularized Common Spatial Patterns for Visual Stimulus and Manual-Response Classification”**, prepared for SOMIB 2026.

The experiment compares a **full 32-channel EEG configuration** against a **fixed 7-channel topology** using regularized Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA). The focus is the effect of reducing the number of EEG input channels while maintaining a consistent classification and evaluation procedure.

> **Research scope:** This is a single-subject, within-recording study using an EEGLAB tutorial dataset. It does not demonstrate cross-subject generalization, clinical applicability, or statistically significant superiority of either montage.

## 1. Repository contents

```text
SOMIB2026_EEG_7Ch_CSP_LDA_Reproducibility/
├── main_experiment.py
├── README.md
├── requirements.txt
└── .gitignore
```

The EEG dataset is **not redistributed** in this repository. After downloading it from EEGLAB, place `eeglab_data.set` alongside `main_experiment.py` (and `eeglab_data.fdt`, if provided).

## 2. Dataset availability

This study uses the publicly available **EEGLAB tutorial/reference EEG dataset** (`eeglab_data.set`), distributed through the EEGLAB project associated with the Swartz Center for Computational Neuroscience (SCCN), University of California San Diego.

The recording is described in the manuscript as a single healthy subject, **32 EEG channels**, sampled at **128 Hz**. The analysis uses annotations corresponding to visual stimulus presentation (`square`) and manual response (`rt`).

**Official source:** https://eeglab.org/tutorials/tutorial_data.html

The dataset is **not included in this repository**. To reproduce the experiment, obtain the original file from the official source and place it in the repository's root directory. Keep any accompanying `.fdt` file beside the `.set` file.

Dataset/software background reference:

> Delorme, A., & Makeig, S. (2004). EEGLAB: An open source toolbox for analysis of single-trial EEG dynamics including independent component analysis. *Journal of Neuroscience Methods*, 134, 9–21. https://doi.org/10.1016/j.jneumeth.2003.10.009

The EEGLAB software citation does not itself establish the dataset's redistribution license; users should consult the source's applicable terms.

## 3. Experimental design

| Parameter | Specification |
|---|---|
| Data source | EEGLAB tutorial/reference dataset |
| Subjects | 1 |
| Sampling frequency | 128 Hz |
| Classification task | `square` versus `rt` |
| Epochs reported in the manuscript | 154 (77 per class) |
| Temporal filter | 1–35 Hz FIR band-pass, `firwin` |
| Epoch interval | −0.2 to +0.7 s relative to the event |
| Baseline correction | −0.2 to 0.0 s |
| Full montage | 32 channels |
| Reduced montage | 7 fixed channels |
| Feature extraction | CSP with Ledoit–Wolf regularization |
| CSP components | 4 |
| Classifier | Linear Discriminant Analysis |
| Cross-validation | Stratified 5-fold, shuffled |
| Random seed | 42 |
| Reported metrics | Accuracy, precision, sensitivity, F1-score |
| Summary statistics | Mean ± sample SD across folds (`ddof=1`) |

### Fixed electrode topology

The reduced montage consists of:

- **Central:** C3, Cz, C4
- **Occipital:** O1, O2
- **Parietal:** P3, P4

The topology is specified *a priori*, not selected by optimizing performance on the dataset.

Relative to the 32-channel configuration, it uses **78.1% fewer channels**:

\[
100\times\left(1-\frac{7}{32}\right)=78.125\%.
\]

This is a reduction in the number of input channels, **not** a directly measured reduction in computational cost, hardware power consumption, or acquisition time.

## 4. Processing and classification pipeline

1. Load `eeglab_data.set` with MNE-Python.
2. Apply a 1–35 Hz FIR band-pass filter.
3. Extract epochs around `rt` and `square` events with baseline correction.
4. Create the full (32-channel) and reduced (7-channel) datasets.
5. Use identical stratified cross-validation partitions for both configurations.
6. Within each training fold, fit regularized CSP and LDA.
7. Evaluate the held-out fold using accuracy, precision, sensitivity, and F1-score.
8. Aggregate fold-level metrics and export tables and plots.

### Model configuration

The principal implementation is:

```python
csp = CSP(
    n_components=4,
    reg="ledoit_wolf",
    log=True,
    norm_trace=False,
    cov_est="concat",
    transform_into="average_power",
)

lda = LinearDiscriminantAnalysis(solver="svd")

pipeline = Pipeline([
    ("csp", csp),
    ("lda", lda),
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

The actual script determines the CSP component count with `min(4, len(selected_channels))`, which gives four components for both evaluated montages.

**Avoiding model-fitting leakage:** Unlike an earlier experimental script that fitted CSP on all epochs before cross-validation, `main_experiment.py` places CSP **inside** the scikit-learn pipeline. CSP and LDA are fitted only on the training subset of each fold. Temporal filtering and epoching are performed before the split; the leakage-prevention statement therefore specifically concerns fitted CSP/LDA model stages.

The code uses `LabelEncoder` to encode classes as 0 and 1 and evaluates precision, sensitivity, and F1-score for the positive class (`pos_label=1`). The mapping from original event codes to these labels is printed during execution.

## 5. Requirements and installation

The manuscript records the following environment:

| Component | Version |
|---|---|
| Python | 3.13.2 |
| MNE-Python | 1.12.1 |
| scikit-learn | 1.8.0 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| Matplotlib | Required; exact version recorded at runtime |
| Operating system | Windows 11 |

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on **macOS/Linux**:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## 6. Run the experiment

After downloading the dataset, your local folder should look like:

```text
SOMIB2026_EEG_7Ch_CSP_LDA_Reproducibility/
├── main_experiment.py
├── eeglab_data.set
├── eeglab_data.fdt        # Only if supplied/required
├── README.md
├── requirements.txt
└── .gitignore
```

From the repository root, run:

```bash
python main_experiment.py
```

The program checks for the required dataset file and electrode names. It prints the class counts, model performance, and output filenames.

## 7. Generated outputs

The experiment automatically generates:

| File | Description |
|---|---|
| `CrossValidation_Fold_Metrics.csv` | Four evaluation metrics for each of the five folds and both montages |
| `Table1_Performance_Metrics.csv` | Mean ± SD for both configurations |
| `Table1_Performance_Metrics.tex` | LaTeX-formatted performance table |
| `Figure2_Metric_Distributions.tif` | Comparative boxplots for all four metrics |
| `Software_Versions.txt` | Python, library, and platform versions |

Generated files are ignored by the supplied `.gitignore` by default. Remove the corresponding ignore rules if you deliberately choose to version verified results.

## 8. Results reported in the manuscript

The following values are taken from the final manuscript and are expressed as **mean ± sample standard deviation** over five cross-validation folds.

| Configuration | Accuracy (%) | Precision (%) | Sensitivity (%) | F1-score (%) |
|---|---:|---:|---:|---:|
| Full montage (32 ch) | 61.72 ± 6.39 | 63.10 ± 8.02 | 66.25 ± 18.01 | 63.55 ± 8.89 |
| Fixed topology (7 ch) | 62.97 ± 5.05 | 65.54 ± 7.79 | 65.00 ± 9.48 | 64.54 ± 3.10 |

The seven-channel configuration achieved a slightly higher mean accuracy in this experiment while requiring fewer input channels. **This observation must not be interpreted as a demonstrated statistically significant advantage.**

**Reproducibility status:** These are manuscript-reported values. Exact numerical agreement with a fresh execution of the packaged script and original dataset has **not yet been independently verified** as part of repository preparation. Users should compare the newly generated `Table1_Performance_Metrics.csv` with the table above.

## 9. Limitations

- The analysis uses one subject and one EEG recording.
- The reduced electrode topology is predefined; no exhaustive electrode selection study was conducted.
- Cross-validation is performed within the recording; no held-out participant or independent recording session was tested.
- The experiment does not establish clinical utility or broad deployment readiness.
- Reducing input channel count does not directly establish lower energy consumption or shorter processing latency.
- Fold-level differences should not be interpreted as independent-subject evidence.
- Dataset access and exact dependency builds may affect numerical reproducibility.

## 10. Citation

If this code contributes to your work, please cite the associated manuscript once its final publication details are available:

> *Evaluation of a Fixed 7-Channel EEG Topology Using Regularized Common Spatial Patterns for Visual Stimulus and Manual-Response Classification*. SOMIB 2026. **Publication metadata and DOI to be added after publication.**

Please also acknowledge the EEGLAB project and the original dataset source.

## 11. License

**No source-code license has been assigned yet.** Before publishing under an open-source license, the repository owners should select a license appropriate to their institution and coauthor agreements. The EEG dataset is governed by its own applicable terms and is not covered by a future license for this repository's code.

## 12. Contact

For questions about the implementation or reproducibility, please open a GitHub Issue in this repository after publication.
