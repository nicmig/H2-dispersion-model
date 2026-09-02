# AGENTS.md — H2 Dispersion Model

> This file is intended for AI coding agents. It assumes you know nothing about the project.

---

## 1. Project Overview

This is a **Python-based machine learning research project** that models hydrogen (H₂) dispersion using Gaussian Processes (GPs). The goal is to predict H₂ concentration fields in space and time, and to infer the underlying mass flow rate from sparse sensor observations.

**What the code does:**
- **Training**: Learns a GP mapping from `(time, mass_flow, y, z)` → `h2_volume_fraction` using a combination of CFD simulation data and real experimental sensor data.
- **Inference**: Given sensor readings at a point in time, infers the most likely mass flow rate (MAP or MCMC) and predicts the full concentration field with uncertainty.
- **Data fusion**: Merges high-fidelity CFD outputs with noisy experimental measurements into unified datasets.

**Primary language**: All comments, docstrings, and documentation are in English.

---

## 2. Technology Stack

| Layer | Packages |
|-------|----------|
| **Python** | 3.10 (isolated venv at `./venv/`, `include-system-site-packages = false`) |
| **Deep Learning** | PyTorch 2.10.0+cu128, Triton 3.6.0 |
| **Gaussian Processes** | GPyTorch 1.15.2, linear_operator 0.6.1, pykeops 2.3, keopscore 2.3 |
| **Scientific** | NumPy 1.26.4, SciPy 1.15.3, pandas 2.3.3, scikit-learn 1.7.2 |
| **GPU / CUDA** | CUDA 12.8 bindings (nvidia-cublas, nvidia-cuda-runtime, etc.) |
| **Visualization** | matplotlib 3.10.8, seaborn 0.13.2 |
| **Other** | tqdm 4.67.3, pysindy 2.1.0 (SINDy PDE discovery), faiss-gpu 1.7.3 |
| **OS** | Linux (local development, no Docker/containerization) |

**Important**: There is **no `requirements.txt`, `pyproject.toml`, `setup.py`, or `setup.cfg`** in the repo root. Dependencies are managed manually inside the venv. To see installed packages, run:
```bash
source venv/bin/activate
pip list
```

---

## 3. Directory Structure

```
H2-dispersion-model/
├── data/                       # All data (gitignored)
│   ├── CFD/                    # CFD simulation outputs (scenarios A-H, O1, O2)
│   ├── raw/                    # Experimental sensor data
│   ├── CFD_vs_exp/             # Comparison visualizations
│   └── unified_raw*.csv        # Preprocessed merged datasets
├── models/                     # Trained model checkpoints (.pth) (gitignored)
├── experiments/                # JSON logs and summary .txt files (gitignored)
├── logs/                       # Full training logs (gitignored)
├── validation_viz/             # Validation plots (gitignored)
├── venv/                       # Python virtual environment (gitignored)
├── __pycache__/                # (gitignored)
│
├── h2_dispersion_gp.py         # Main training module (~1141 lines)
├── additive_kernels.py         # Custom GPyTorch additive kernels
├── h2_dispersion_inference.py  # Operational inference module
├── create_unified_raw.py       # Data preprocessing & unification
├── h2_dispersion_forecast_sindy.py  # SINDy / PDE discovery (experimental)
├── run_training.sh             # Bash launcher for background training
├── data_analysis.ipynb         # EDA notebook
├── kernel_analysis.ipynb       # GP kernel analysis notebook
├── README.md                   # Minimal project title
└── BACKGROUND_TRAINING_GUIDE.md  # Training logistics guide
```

---

## 4. Code Organization & Module Responsibilities

### `h2_dispersion_gp.py` — Training Core
This is the largest and most important file. It defines:
- **`SparseH2DispersionGP`** — Sparse Variational GP with inducing points (`CholeskyVariationalDistribution`).
- **`ExactGPModel`** — Full exact GP with pairwise RBF kernel products.
- **`VNNGP`** — **Primary model** used in recent training. Variational Nearest Neighbor GP (`NNVariationalStrategy`) with `ScaleAdditiveKernel`.
- **`ExperimentLogger`** — JSON-based experiment tracker.
- **Training functions**:
  - `train_h2_dispersion_gp()` — Exact GP training.
  - `train_h2_dispersion_gp_approximate_additive()` — Sparse / VNNGP training with validation. Supports `model_type="SVGP"` or `"VNNGP"`, and `likelihood_type="gaussian"` or `"beta"`.
  - `train_h2_dispersion_gp_exact_additive()` — Exact GP with `FullAdditiveKernel`.
- **Utilities**: `save_checkpoint()`, `evaluate_gp_model()`, `plot_gp_predictive_distributions()`, `save_model()`, `load_model()`.

**Key constant**: `LOG_EPSILON = 1e-4` — added before log-transforming the target to avoid `log(0)`.

### `additive_kernels.py` — Custom Kernels
Implements additive GP kernels from Duvenaud et al. (2011):
- **`FullAdditiveKernel`** — All interaction orders (1st, 2nd, 3rd, 4th). For 4D input this creates 15 terms. Uses `keops` RBF/Matern kernels.
- **`ScaleAdditiveKernel`** — A more interpretable variant with fewer outputscale parameters. This is the one currently used in `VNNGP`.

### `h2_dispersion_inference.py` — Operational Inference
Defines **`H2DispersionInference`**:
- `infer_mass_flow()` — Infer mass flow from sensor readings using MAP, MCMC (Metropolis-Hastings), or Laplace approximation.
- `predict_field()` — Predict concentration field on a `(y, z)` grid.
- `full_inference()` — End-to-end pipeline: mass-flow inference + field prediction + uncertainty quantification.
- `load_inference_model()` — Helper to load a checkpoint into an inference wrapper.

Sensor positions are hardcoded in a `SENSOR_POSITIONS` dict (also present in `create_unified_raw.py`).

### `create_unified_raw.py` — Data Preprocessing
Merges CFD and experimental data into unified CSVs:
- Resamples CFD from 100 Hz → 1.6 Hz.
- Maps CFD scenarios (A-H, O1, O2) to experimental runs.
- Applies time cut-offs per experiment (leakage stop times).
- Defines train/test splits (`HELD_OUT_TEST_EXPERIMENTS`).
- Output columns: `time, mass_flow, sensor_id, x, y, z, h2_volume_fraction, source, scenario, experiment_id, split`.

**Active dataset**: `data/unified_raw_two_modes.csv` (used by `run_training.sh`).

---

## 5. Data Pipeline & Formats

### Raw Data Sources
1. **CFD** (`data/CFD/{scenario}/h2SensorXX/0.1/data.csv`):
   - Columns: `time, mass_flow, h2_mass_fraction, h2_volume_fraction`
   - 100 Hz time series.
   - Scenarios: A (0.086 kg/s), B (0.15), C (0.20), D (0.30), E (0.48), F (0.74), G (1.00), H (1.27), O1 (0.091), O2 (0.45).
   - 10 scenarios but 6 of them represent experimental tests.
   - 4 scenarios add actual new data to the overall data.

2. **Experiments** (`data/raw/23_FFI_P101_T000xx/`):
   - 22 tests between 60 and 240 seconds.
   - 1.6 Hz time series.
   - CSV files with sensor readings and JPEG plots.
   - H₂ concentrations are percentages (divided by 100 during unification).

### Unified Data
- Produced by running `create_unified_raw.py`.
- The standard output is `data/unified_raw_two_modes.csv`.
- `split` column values: `train` / `test`.
- Training data is further split into train/validation by scenario (random 20% of training scenarios held out).

---

## 6. Model Variants

| Model | Class | Use Case | Status |
|-------|-------|----------|--------|
| **Exact GP** | `ExactGPModel` | Small datasets only | Legacy |
| **Sparse Variational GP** | `SparseH2DispersionGP` | Medium datasets with inducing points | Supported |
| **VNNGP** | `VNNGP` | Large datasets, nearest-neighbor variational approximation | **Primary** |
| **Additive Kernel (Full)** | `FullAdditiveKernel` | Interpretable interactions of all orders | Supported |
| **Additive Kernel (Scale)** | `ScaleAdditiveKernel` | Interpretable with fewer parameters | **Used in VNNGP** |

**Target preprocessing**: `y_log = log(y + 1e-4)` because H₂ concentrations range from ~`1e-8` to ~`0.3`. Predictions are inverse-transformed with `exp(pred) - 1e-4`.

**Input scaling**: For approximate models, `StandardScaler` is applied to inputs; exact models use raw inputs.

---

## 7. Build & Run Commands

### Environment Setup
```bash
# Activate the virtual environment
source venv/bin/activate

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Training
```bash
# Recommended: background training with logging and summary
./run_training.sh my_experiment_name

# Monitor
 tail -f logs/experiment_scaleAdditiveVNNGP_K64_Gaussian_*.log

# Check status
ps aux | grep python
nvidia-smi
```

The `run_training.sh` script:
- Sets `CUDA_VISIBLE_DEVICES=0`
- Hardcodes config: `model_type='VNNGP'`, `likelihood_type='gaussian'`, `k=64`, `n_inducing=500`, `n_epochs=300`, `lr=0.005`, `batch_size=1024`
- Saves per-epoch checkpoints to `models/approximate_scaleAdditive_vnngp_k64_gaussian_cutoff_2d.pth{epoch}`
- Saves final model to `models/approximate_scaleAdditive_vnngp_k64_gaussian_cutoff_2d.pth`
- Writes JSON experiment logs to `experiments/`
- Writes text summaries to `experiments/*_summary.txt`

### Interactive Training (Python)
```python
import pandas as pd
from h2_dispersion_gp import train_h2_dispersion_gp_approximate_additive

df = pd.read_csv('data/unified_raw_cut_off_2d.csv')

model, likelihood, history = train_h2_dispersion_gp_approximate_additive(
    df=df,
    split_ratio=0.2,
    n_inducing=500,
    k=64,
    training_batch_size=1024,
    model_type="VNNGP",
    likelihood_type="gaussian",
    n_epochs=300,
    learning_rate=0.005,
    device='cuda:0',
    model_path='models/my_model.pth',
    trained_model=None  # or path to resume
)
```

### Inference
```python
from h2_dispersion_inference import load_inference_model

# Sensor positions must match training
SENSOR_POSITIONS = { ... }

inference = load_inference_model(
    'models/approximate_scaleAdditive_vnngp_k64_gaussian_cutoff_2d.pth',
    SENSOR_POSITIONS,
    device='cuda:0'
)

result = inference.full_inference(
    time=10.0,
    sensor_readings={1: 0.01, 5: 0.05, 10: 0.02},
    grid_points=grid,  # N x 2 array of (y, z)
    prediction_method='marginalize'
)
```

### Notebooks
```bash
# Start Jupyter (ensure venv is active)
jupyter notebook
# Open data_analysis.ipynb or kernel_analysis.ipynb
```

### Data Regeneration
```bash
# Re-create unified datasets from raw CFD + experimental data
python create_unified_raw.py
```

---

## 8. Development Conventions

### Code Style
- **Mixed type hints**: Newer modules (`h2_dispersion_inference.py`) use type hints extensively; training scripts use them sparingly.
- **Docstrings**: Present but inconsistent (Google-style vs free-form).
- **Constants**: `LOG_EPSILON = 1e-4` is the canonical epsilon for log transforms; do not change without retraining.
- **Paths**: Some scripts contain hardcoded absolute paths (`/home/niclasflehmig/VisualCodeProjects/H2-dispersion-model`). Be careful when moving the project.
- **Commented code**: Significant blocks of commented-out experimental code remain in `h2_dispersion_gp.py` (e.g., alternative kernels, old training calls). Treat these as historical artifacts, not active code.

### Git
- `.gitignore` ignores: `/data`, `/venv`, `/validation_viz`, `/models`, `/__pycache__`, `/experiments`, `BACKGROUND_TRAINING_GUIDE.md`.
- **Do not commit** model weights, logs, or large CSVs.

### Experiment Tracking
- `ExperimentLogger` writes JSON to `experiments/experiment_{timestamp}.json`.
- Each entry includes: `event` (training_start, epoch, training_end, evaluation), `timestamp`, `config`, metrics.

---

## 9. Testing Strategy

**There is currently no formal test suite.** No `pytest`, `unittest`, or CI/CD pipelines exist.

**Manual validation practices:**
- Train/validation split by scenario (not random points) to test generalization across experimental conditions.
- Held-out test experiments: `23_FFI_P101_T00005`, `T00014`, `T00031`, `T00045`.
- Visual validation via `kernel_analysis.ipynb` and `validation_viz/` plots.
- `evaluate_gp_model()` in `h2_dispersion_gp.py` provides MAE, RMSE, and NLL on test data.

**If adding tests**, focus on:
1. Data pipeline correctness (`create_unified_raw.py`): verify sensor positions, mass flow mappings, and split logic.
2. Model forward pass: ensure kernels produce finite outputs for representative inputs.
3. Checkpoint round-trip: save and reload a model, verify predictions match.

---

## 10. Deployment / Operational Inference

For operational use (not training):
1. Train a model and save checkpoint (`.pth`).
2. Load via `load_inference_model()` or manually:
   ```python
   checkpoint = torch.load(path, map_location='cuda:0')
   model = VNNGP(...)
   model.load_state_dict(checkpoint['model_state_dict'])
   ```
3. Wrap in `H2DispersionInference` with sensor positions.
4. Call `full_inference()` with real-time sensor readings.

**Performance**: VNNGP inference is batched and GPU-accelerated. Field predictions on a dense grid should be batched to avoid OOM.

---

## 11. Known Issues & Agent Tips

### Bug: `run_training.sh` summary formatting crash
**Location**: `run_training.sh` line 164.
**Problem**:
```python
  - MAE: {history.get('MAE', 'N/A'):.4f}
```
`history['MAE']` is initialized as `[]` and only appended once after training. When the success summary is built, `history.get('MAE', 'N/A')` returns a list, causing:
```
TypeError: unsupported format string passed to list.__format__
```
**Fix**: Access the scalar element: `history['MAE'][-1] if history['MAE'] else 'N/A'`.

### Unfinished code
- `h2_dispersion_forecast_sindy.py` has undefined variables (`C_field` vs `c_field`, `y_unique`, `z_unique`). Do not treat this as production-ready.

### Path assumptions
- `run_training.sh` assumes the CWD is the project root because it reads `data/unified_raw_cut_off_2d.csv` with a relative path.
- The temporary Python script hardcodes the project root in `sys.path.insert(0, '/home/niclasflehmig/VisualCodeProjects/H2-dispersion-model')`.

### GPU memory
- VNNGP with `k=64`, `n_inducing=500`, and `batch_size=1024` trains fine on an RTX 3090 (24 GB).
- If you increase `k`, `n_inducing`, or batch size, monitor with `nvidia-smi` or `watch -n 1 nvidia-smi`.

### Reproducibility
- Random seed `42` is used for validation scenario selection.
- No global PyTorch/NumPy random seed is set in the training scripts. Add `torch.manual_seed(42)` and `np.random.seed(42)` if strict reproducibility is needed.

---

## 12. Security Considerations

- No API keys, passwords, or secrets are stored in code.
- No network services are exposed.
- `run_training.sh` runs background jobs with `&` and `source venv/bin/activate` — no privilege escalation.
- Temporary scripts are written to `/tmp/run_training_${TIMESTAMP}.py` and deleted after execution.
