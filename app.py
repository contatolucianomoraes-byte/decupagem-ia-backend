import shutil
import uuid
from pathlib import Path
from zipfile import ZipFile
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from processar_video import run_pipeline

app = FastAPI(title="Decupagem IA API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/process")
async def process(
    request: Request,
    video_file: UploadFile = File(...),
    mogrt_file: UploadFile | None = File(default=None),
    model: str = Form("small"),
    max_final_seconds: float = Form(180.0),
    letterings_per_minute: int = Form(5),
    mogrt_text_param: str = Form("TEXT"),
):
    job_id = uuid.uuid4().hex[:10]
    job_dir = JOBS_DIR / job_id
    inputs_dir = job_dir / "inputs"
    outputs_dir = job_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    video_path = inputs_dir / (video_file.filename or "video.mp4")
    with video_path.open("wb") as f:
        shutil.copyfileobj(video_file.file, f)

    effective_mogrt_path = ""
    if mogrt_file and mogrt_file.filename:
        mogrt_path = inputs_dir / mogrt_file.filename
        with mogrt_path.open("wb") as f:
            shutil.copyfileobj(mogrt_file.file, f)
        effective_mogrt_path = str(mogrt_path)

    try:
        run_pipeline(
            video_path=video_path,
            out_dir=outputs_dir,
            model=model,
            max_final_seconds=max_final_seconds,
            letterings_per_minute=letterings_per_minute,
            mogrt_path=effective_mogrt_path,
            mogrt_text_param=mogrt_text_param,
        )
    except Exception as exc:
        return JSONResponse({"error": f"Falha no processamento: {exc}"}, status_code=500)

    zip_path = job_dir / "pacote_premiere.zip"
    with ZipFile(zip_path, "w") as zf:
        for item in outputs_dir.iterdir():
            zf.write(item, arcname=item.name)

    base = str(request.base_url).rstrip("/")
    return {
        "job_id": job_id,
        "download_url": f"{base}/download/{job_id}",
    }

@app.get("/download/{job_id}")
def download(job_id: str):
    zip_path = JOBS_DIR / job_id / "pacote_premiere.zip"
    if not zip_path.exists():
        return JSONResponse({"error": "arquivo nao encontrado"}, status_code=404)
    return FileResponse(
        path=zip_path,
        filename="pacote_premiere.zip",
        media_type="application/zip",
    )
