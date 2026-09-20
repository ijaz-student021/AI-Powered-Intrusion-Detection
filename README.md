# SecureNet Corp. — AI Intrusion Detection Panel

This is a full-stack web application designed for a Security Operations Center (SOC) analyst to detect network intrusions using a trained LightGBM machine learning model.

## Features
- **FastAPI Backend**: Serves single-record predictions and batch CSV predictions.
- **Dynamic Preprocessing**: Leverages a saved preprocessing recipe to appropriately drop columns, one-hot encode categorical fields, and align the output before it hits the model pipeline.
- **Frontend Panel**: A modern dark-mode dashboard allowing single-record analysis (via dynamic form or raw JSON) and bulk CSV processing with visual distribution charts.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js (or any simple local HTTP server for the frontend)

### Backend
1. Open a terminal and navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI development server:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will start at `http://127.0.0.1:8000`.

### Frontend
1. The frontend consists of static files in the `frontend` folder.
2. Serve the `frontend` directory using any local web server. For example:
   - Using VS Code Live Server extension: Right click `index.html` -> Open with Live Server.
   - Using Python: `cd frontend && python -m http.server 5500`
3. Navigate to the local URL (e.g., `http://127.0.0.1:5500/index.html`) in your browser.
4. Ensure the `API_BASE` constant in `frontend/app.js` correctly points to the running backend (default is `http://127.0.0.1:8000/api`).
