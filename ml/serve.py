"""NWIS ML Microservice: FastAPI Server for Real-Time Inference, RAG, and Event Extraction.

Exposes REST APIs conforming strictly to NWIS Roadmap Contracts:
- POST /api/v1/predict/mud-loss (Roadmap Section 4.3)
- POST /api/v1/rag/query (Roadmap Section 4.4)
- POST /api/v1/documents/process (Roadmap Section 23)
- GET  /health
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import requests

# Add ML root and project root to path
ML_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT.parent))

from src.risk.mud_loss_model import MudLossRiskModel
from src.risk.schemas import RiskPredictionRequest, RiskPredictionResult
from src.rag.assistant import RAGKnowledgeAssistant
from src.rag.schemas import RAGQueryFilter, RAGResponse
from src.features.stratigraphy import get_formation_for_depth
from src.similarity.well_similarity import WellSimilarityEngine
from src.features.spatial_features import SpatialOffsetFeatureExtractor
from pipelines.extract_events import process_processed_document

from contextlib import asynccontextmanager

# Global model and RAG instances
MODEL_PATH = ML_ROOT / "models/trained/mud_loss_random_forest.joblib"
risk_model: MudLossRiskModel | None = None
rag_assistant: RAGKnowledgeAssistant | None = None
spatial_extractor: SpatialOffsetFeatureExtractor | None = None


def init_models():
    global risk_model, rag_assistant, spatial_extractor
    if risk_model is None and MODEL_PATH.exists():
        risk_model = MudLossRiskModel.load(MODEL_PATH)
        print(f"[ML Server] Loaded Risk Model from {MODEL_PATH}")

    if rag_assistant is None:
        processed_dir = ML_ROOT / "data/processed/good_data_processed"
        rag_assistant = RAGKnowledgeAssistant(processed_dir=processed_dir)
        print(f"[ML Server] Initialized RAG Assistant with {len(rag_assistant.chunks)} chunks")

    if spatial_extractor is None:
        processed_dir = ML_ROOT / "data/processed/good_data_processed"
        wells_csv = processed_dir / "historical_dataset/wells.csv"
        events_csv = processed_dir / "historical_dataset/events.csv"
        similarity_engine = WellSimilarityEngine(
            wells_csv_path=wells_csv if wells_csv.exists() else None,
            events_csv_path=events_csv if events_csv.exists() else None,
        )
        spatial_extractor = SpatialOffsetFeatureExtractor(similarity_engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()
    yield


app = FastAPI(
    title="NWIS ML Intelligence Microservice",
    description="Real-time drilling risk prediction, RAG knowledge assistant, and event extraction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eagerly initialize so models are immediately available upon import
init_models()


class PredictRiskRequestDto(BaseModel):
    well_id: str
    depth_md: float
    formation: str | None = None
    lithology: str = "Porous Sandstone / Siltstone"
    rop_mhr: float = 12.0
    wob_klbs: float = 24.0
    rpm: float = 110.0
    torque_kftlb: float = 8.0
    spp_psi: float = 2200.0
    flow_rate_gpm: float = 550.0
    mud_weight_ppg: float = 9.8
    ecd_ppg: float = 10.2
    offset_wells_count_in_radius: int | None = None
    max_offset_similarity: float | None = None
    offset_mud_loss_count: int | None = None
    offset_stuck_pipe_count: int | None = None
    offset_kick_count: int | None = None
    offset_torque_spike_count: int | None = None
    nearest_hazard_distance_km: float | None = None
    offset_hazard_density: float | None = None


class RAGQueryRequestDto(BaseModel):
    query: str | None = None
    question: str | None = None
    well_id: str | None = None
    formation: str | None = None
    event_type: str | None = None
    min_depth: float | None = None
    max_depth: float | None = None
    top_k: int = 4

    def get_query(self) -> str:
        return (self.query or self.question or "").strip()


class ProcessDocumentRequestDto(BaseModel):
    document_dir: str
    backend_ingest_url: str | None = "http://localhost:3000/api/v1/ml/events/ingest"


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "risk_model_loaded": risk_model is not None,
        "rag_assistant_loaded": rag_assistant is not None,
        "rag_chunks_indexed": len(rag_assistant.chunks) if rag_assistant else 0,
    }


@app.post("/api/v1/predict/mud-loss")
def predict_mud_loss(req: PredictRiskRequestDto) -> dict[str, Any]:
    """Real-time mud loss risk prediction conforming to Roadmap Section 4.3."""
    if not risk_model:
        raise HTTPException(status_code=503, detail="Risk model not loaded")

    formation_str = req.formation
    lithology_str = req.lithology
    if not formation_str or formation_str == "Unknown":
        fmt_info = get_formation_for_depth(req.depth_md)
        formation_str = fmt_info.name
        if not lithology_str:
            lithology_str = fmt_info.primary_lithology

    # Auto-infer spatial features if not supplied
    spatial = spatial_extractor.extract_offset_features(
        well_id=req.well_id,
        depth_md=req.depth_md,
        formation=formation_str,
    ) if spatial_extractor else {}

    model_req = RiskPredictionRequest(
        well_id=req.well_id,
        depth_md=req.depth_md,
        formation=formation_str,
        lithology=lithology_str or "Unknown",
        rop_mhr=req.rop_mhr,
        wob_klbs=req.wob_klbs,
        rpm=req.rpm,
        torque_kftlb=req.torque_kftlb,
        spp_psi=req.spp_psi,
        flow_rate_gpm=req.flow_rate_gpm,
        mud_weight_ppg=req.mud_weight_ppg,
        ecd_ppg=req.ecd_ppg,
        offset_wells_count_in_radius=req.offset_wells_count_in_radius or spatial.get("offset_wells_count_in_radius", 0),
        max_offset_similarity=req.max_offset_similarity or spatial.get("max_offset_similarity", 0.0),
        offset_mud_loss_count=req.offset_mud_loss_count or spatial.get("offset_mud_loss_count", 0),
        offset_stuck_pipe_count=req.offset_stuck_pipe_count or spatial.get("offset_stuck_pipe_count", 0),
        offset_kick_count=req.offset_kick_count or spatial.get("offset_kick_count", 0),
        offset_torque_spike_count=req.offset_torque_spike_count or spatial.get("offset_torque_spike_count", 0),
        nearest_hazard_distance_km=req.nearest_hazard_distance_km or spatial.get("nearest_hazard_distance_km", 999.0),
        offset_hazard_density=req.offset_hazard_density or spatial.get("offset_hazard_density", 0.0),
    )

    result = risk_model.predict_one(model_req)
    return result.to_dict()


@app.post("/api/v1/rag/query")
def query_rag(req: RAGQueryRequestDto) -> dict[str, Any]:
    """Evidence-backed Q&A query conforming to Roadmap Section 4.4."""
    if not rag_assistant:
        raise HTTPException(status_code=503, detail="RAG Knowledge Assistant not initialized")

    q = req.get_query()
    if not q:
        raise HTTPException(status_code=400, detail="Query or question is required")

    filters = RAGQueryFilter(
        well_id=req.well_id,
        formation=req.formation,
        event_type=req.event_type,
        min_depth=req.min_depth,
        max_depth=req.max_depth,
    )

    resp = rag_assistant.answer_query(q, filters=filters, top_k=req.top_k)
    return resp.to_dict()


@app.post("/api/v1/documents/process")
def process_document(req: ProcessDocumentRequestDto, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Processes Stage 4 document output and posts extracted events to backend."""
    doc_path = Path(req.document_dir)
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document folder not found: {req.document_dir}")

    events_json_path = process_processed_document(doc_path)
    events_payload = Path(events_json_path).read_text(encoding="utf-8")
    import json
    data = json.loads(events_payload)

    # Optional background callback to backend
    if req.backend_ingest_url and data.get("events"):
        def post_to_backend(url: str, events: list):
            try:
                requests.post(url, json={"events": events}, timeout=10)
            except Exception as e:
                print(f"[ML Server] Backend callback failed: {e}")

        background_tasks.add_task(post_to_backend, req.backend_ingest_url, data["events"])

    return {
        "status": "success",
        "events_count": len(data.get("events", [])),
        "events_path": str(events_json_path),
        "events": data.get("events", []),
    }


def main():
    port = int(os.environ.get("ML_PORT", "8000"))
    uvicorn.run("serve:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
