# backend/main.py
# This is the FastAPI application entry point.
# It creates the database tables on startup and wires in all the route groups (routers).
# Think of it as the main() of the API server.

from fastapi import FastAPI
from backend.db import create_tables
from backend.routers import upload, analysis, clients, feedback

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield

app = FastAPI(
    title="ClientPulse API",
    description="B2B client churn and health predictor",
    version="2.0.0",
    lifespan=lifespan,
)


# Include routers – each file handles its own group of endpoints
app.include_router(upload.router)
app.include_router(analysis.router)
app.include_router(clients.router)
app.include_router(feedback.router)


@app.get("/health", tags=["health"])
def health():
    from backend.db import count_history
    from ml.explain import load_model
    import os
    model_loaded = os.path.exists("models/model.pkl")
    clients_in_db = 0
    try:
        clients_in_db = count_history()
    except Exception:
        pass
    return {"status": "ok", "model_loaded": model_loaded, "clients_in_db": clients_in_db}
