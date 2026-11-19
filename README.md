# ClientPulse

ClientPulse is a client churn predictor for B2B teams. A support or account team uploads monthly usage data for their clients. The app finds the clients who are likely to leave, explains why, and suggests what to do.

> Note: The model is trained on synthetic data, so the scores show that the pipeline works, not real-world accuracy. The risk score is a ranking score, not a probability.

## Features

- Upload client data as CSV or Excel (at least 3 months per client)
- Churn risk score for every client using XGBoost
- Risk levels: High, Medium and Low (thresholds can be changed)
- SHAP chart that shows why a client is at risk
- Suggested actions for each client
- AI-written note for each client using the Gemini API (falls back to a rule-based note)
- Revenue at risk and a savings estimator
- List of clients whose risk went up since the last run
- Feedback buttons (good call, false alarm, client left, client stayed)
- Model metrics page

## Tech stack

- Python
- XGBoost, scikit-learn, SHAP
- FastAPI (backend)
- Streamlit (frontend)
- Gemini API (AI notes)
- pytest (tests)

## How it works

1. The user uploads monthly client data.
2. `ml/features.py` turns the history into 10 features (login drop, feature adoption, tickets, days to renewal, plan type, and so on).
3. The XGBoost model gives each client a risk score.
4. SHAP explains which features raised or lowered the score.
5. `ml/actions.py` suggests actions and `ml/llm_note.py` writes a short note.
6. The Streamlit dashboard shows everything. It talks to the FastAPI backend over HTTP.

## Project structure

```
ClientPulse/
├── backend/        FastAPI app (main.py, routers, schemas, pipeline, db)
├── ml/             features, training, SHAP, actions, LLM notes
├── data/           data generator and sample CSVs
├── frontend/       Streamlit app (app.py)
├── models/         trained model and metrics
├── db/             database file
├── tests/          pytest tests
├── requirements.txt
└── .env.example
```

## Setup

1. Clone the repo

```
git clone https://github.com/<your-username>/ClientPulse.git
cd ClientPulse
```

2. Create and activate a virtual environment (Windows PowerShell)

```
python -m venv venv
venv\Scripts\activate
```

3. Install the requirements

```
pip install -r requirements.txt
```

4. Create a `.env` file (copy `.env.example`) and add your Gemini API key

5. Train the model

```
python -m ml.train
```

## Run the app

Open two terminals. Activate the virtual environment in both.

Terminal 1 (backend):

```
uvicorn backend.main:app --reload
```

The API docs are at http://127.0.0.1:8000/docs

Terminal 2 (frontend):

```
streamlit run frontend/app.py
```

The dashboard opens at http://localhost:8501

## How to use

1. Click **Load sample data** in the sidebar, or upload your own CSV.
2. Click **Run analysis**.
3. Check the Overview, Client list and Client details tabs.

### CSV format

| Column | Example |
|---|---|
| client_id | C001 |
| month | 2026-01 |
| logins | 55 |
| features_used | 12 (out of 20) |
| tickets | 2 |
| last_login_days_ago | 4 |
| contract_end_date | 2027-03-31 |
| plan_type | Standard, Premium or Enterprise |
| monthly_revenue | 45000 |

Each client needs at least 3 months of data. A template can be downloaded from the app sidebar.

## Model results (synthetic test set)

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| AUC | 0.97 | 0.96 |
| Precision | 0.61 | 0.89 |
| Recall | 0.89 | 0.89 |
| F1 | 0.72 | 0.89 |

The test set is small (125 clients), so these numbers can change a lot between runs.

## Run the tests

```
pytest
```

## Limitations

- Trained on synthetic data, not real client data
- Risk score is a ranking score, not a calibrated probability
- Savings estimate depends on the assumptions chosen by the user
