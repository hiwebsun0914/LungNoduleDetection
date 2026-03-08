from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inference import (
    ModelRunner,
    detect_nodules,
    load_dicom_series,
    save_detection_frames,
    voxel_center_to_world_xyz,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "Website" / "frontend"
RESULTS_DIR = PROJECT_ROOT / "Website" / "backend" / "static" / "results"

DEFAULT_DATASET = "H:\\ScienceAndExplore\\MedicalCreation_dataset\\LIDC\\LIDC-IDRI-0001"
DEFAULT_CHECKPOINT = "checkpoint/checkpoint-200000"

app = FastAPI(title="Lung CT AI Nodule Early Detection")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = ModelRunner()


class PredictRequest(BaseModel):
    folder_path: str = Field(default=DEFAULT_DATASET)
    checkpoint_path: str = Field(default=DEFAULT_CHECKPOINT)
    probability_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


def resolve_project_path(path_text: str) -> Path:
    path_text = path_text.strip().strip("\"'")
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@app.get("/")
def index() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="前端页面不存在")
    return FileResponse(index_file)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    folder_path = resolve_project_path(req.folder_path)
    checkpoint_path = resolve_project_path(req.checkpoint_path)

    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(status_code=400, detail=f"数据目录不存在: {folder_path}")
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        raise HTTPException(status_code=400, detail=f"模型目录不存在: {checkpoint_path}")

    try:
        model = runner.load(checkpoint_path)
        volume, metadata = load_dicom_series(folder_path)
        detections = detect_nodules(model, volume, probability_threshold=req.probability_threshold)

        result_id = uuid4().hex
        frames_dir = RESULTS_DIR / result_id
        saved_frames = save_detection_frames(volume, detections, frames_dir)

        response_detections = []
        for det in detections:
            world_xyz = voxel_center_to_world_xyz(det["center_zyx"], metadata)
            response_detections.append(
                {
                    "probability": det["probability"],
                    "bbox_zyx": det["bbox_zyx"],
                    "center_zyx": det["center_zyx"],
                    "center_xyz_mm": world_xyz,
                }
            )

        return {
            "folder_path": str(folder_path),
            "checkpoint_path": str(checkpoint_path),
            "shape_zyx": list(volume.shape),
            "dicom_files": metadata.source_files,
            "spacing_xyz": metadata.spacing_xyz.tolist(),
            "origin_xyz": metadata.origin_xyz.tolist(),
            "detections": response_detections,
            "frame_urls": [
                {
                    "url": f"/static/results/{result_id}/{frame['file_name']}",
                    "z_index": frame["z_index"],
                }
                for frame in saved_frames
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "Website" / "backend" / "static")), name="static")
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
