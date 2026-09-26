# SecureNet Corp. — AI Intrusion Detection Panel

This is a full-stack web application designed for a Security Operations Center (SOC) analyst to detect network intrusions using a trained LightGBM machine learning model.

## Project Objective (CLO 4 Assignment)
Design, build and evaluate a proof-of-concept machine-learning model that classifies network traffic as Normal or one of nine attack families, to augment SecureNet Corp.'s NIDS. Full write-up: `CLO4_IDS_Report.docx`.

## Dataset Setup
- Dataset: **UNSW-NB15** (official partitioned CSVs `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`), downloadable from the UNSW Canberra Cyber site or Kaggle.
- Place both CSVs where the notebook can read them (the notebook uses `/content/` for Google Colab; change the path when running locally).
- Note: in the notebook, the file named "testing-set" (175,341 rows) is used for training and "training-set" (82,332 rows) as the final unseen test.

## How to Run the Notebook
1. `pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm tensorflow pytorch-tabnet`
2. Open `Assignmetn_one_file.ipynb` in Jupyter/Colab, fix the CSV paths, and run all cells top to bottom (SMOTENC and the deep models take several minutes).

## Results Summary (unseen test set, 82,332 flows)
| Model | 10-class accuracy | Macro F1 | Attack recall | False alarms |
|---|---|---|---|---|
| **Two-Stage IDS (deployed)** | **0.774** | **0.516** | 96.5% | 15.4% |
| LightGBM (SMOTENC, notebook) | 0.71 | 0.48 | 98.9% | 32.8% |
| XGBoost (notebook) | 0.71 | 0.48 | - | - |
| MLP (notebook) | 0.67 | 0.42 | - | - |
| Deep NN (notebook) | 0.51 | 0.29 | - | - |

The two-stage model (a Normal-vs-attack LightGBM plus an attack-family LightGBM) has a Normal-vs-attack ROC-AUC of 0.985. Its alert threshold is tunable: at 0.9 it flags 89% of attacks with only 2.6% false alarms. Analysis and Backdoor stay hard to identify. Why the first-round notebook results were weak (SMOTENC applied before splitting, partition shift, a serving preprocessing bug) is explained in `CLO4_IDS_Report.docx`.

### Retraining the two-stage model
```bash
pip install matplotlib
python scripts/train_two_stage.py --data <folder containing UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv>
```
This rewrites `model/pipeline_twostage.pkl`, `model/twostage_metrics.json` and the `*_twostage.png` figures in `reports/figures/`. The API loads them on startup.

## Features
- **FastAPI Backend**: Serves single-record predictions and batch CSV predictions.
- **Dynamic Preprocessing**: Leverages a saved preprocessing recipe to appropriately drop columns, one-hot encode categorical fields, and align the output before it hits the model pipeline.
- **Frontend Panel**: A modern dark-mode dashboard allowing single-record analysis (via dynamic form or raw JSON) and bulk CSV processing with visual distribution charts.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js (or any simple local HTTP server for the frontend)

### Backend
Requires Python 3.11 or 3.12 — the committed model files were saved with scikit-learn 1.6.1, which has no prebuilt wheel for newer Python releases (3.13+) and will fail to build from source on most machines.

1. From the **repository root** (not `backend/` — `main.py` uses package-relative imports and must be run as `backend.main`), create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
   pip install -r backend/requirements.txt
   ```
2. (Optional) Set a stable API key and allowed frontend origins. If `API_KEY` isn't set, a random one is generated and printed to the console each time the server starts:
   ```bash
   set API_KEY=your-own-secret-key           # PowerShell: $env:API_KEY="your-own-secret-key"
   set ALLOWED_ORIGINS=http://127.0.0.1:5500 # comma-separated if serving the frontend from elsewhere
   ```
3. Run the FastAPI development server from the repository root:
   ```bash
   uvicorn backend.main:app --reload
   ```
   The backend will start at `http://127.0.0.1:8000`.

### Frontend
Two UIs are provided: `frontend/` (dashboard) and `frontend_3d/` (3D Three.js globe design; use this for the demo). Both work the same way; the steps below apply to either. For the 3D one, add its origin to `ALLOWED_ORIGINS`. Batch upload limits can be raised with the `MAX_UPLOAD_MB` and `MAX_ROWS` env vars.

1. The frontend consists of static files in the `frontend` folder.
2. Serve the `frontend` directory using any local web server. For example:
   - Using VS Code Live Server extension: Right click `index.html` -> Open with Live Server.
   - Using Python: `cd frontend && python -m http.server 5500`
3. Navigate to the local URL (e.g., `http://127.0.0.1:5500/index.html`) in your browser.
4. Go to the **Settings** page and confirm the API Base URL points to the running backend (default is `http://127.0.0.1:8000/api`), and paste in the API key printed in the backend's console (or the one you set via `API_KEY`). Click **Test Connection** to save both.
