import shutil
import uuid
import threading
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

BASE_DIR = Path("/tmp/decupagem_jobs")
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job status tracker
job_statuses: dict[str, dict] = {}


def process_job(job_id: str, video_path: Path, mogrt_path: Path | None,
                model: str, letterings_per_minute: int, mogrt_text_param: str):
    """Run processing in a background thread."""
    try:
        job_statuses[job_id]["status"] = "processing"
        job_dir = JOBS_DIR / job_id
        out_dir = job_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        run_pipeline(
            video_path=str(video_path),
            output_dir=str(out_dir),
            model_size=model,
            letterings_per_minute=letterings_per_minute,
            mogrt_path=str(mogrt_path) if mogrt_path else None,
            mogrt_text_param=mogrt_text_param,
        )

        zip_path = job_dir / "pacote_premiere.zip"
        with ZipFile(zip_path, "w") as zf:
            for f in out_dir.rglob("*"):
                zf.write(f, f.relative_to(out_dir))

        job_statuses[job_id]["status"] = "completed"
    except Exception as e:
        job_statuses[job_id]["status"] = "failed"
        job_statuses[job_id]["error"] = str(e)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process_video(
    request: Request,
    video_file: UploadFile = File(...),
    mogrt_file: UploadFile | None = File(None),
    model: str = Form("small"),
    letterings_per_minute: int = Form(5),
    mogrt_text_param: str = Form("TEXT"),
):
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save video
    video_path = job_dir / video_file.filename
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video_file.file, f)

    # Save MOGRT if provided
    mogrt_path = None
    if mogrt_file and mogrt_file.filename:
        mogrt_path = job_dir / mogrt_file.filename
        with open(mogrt_path, "wb") as f:
            shutil.copyfileobj(mogrt_file.file, f)

    # Register job and start background processing
    job_statuses[job_id] = {"status": "queued"}
    thread = threading.Thread(
        target=process_job,
        args=(job_id, video_path, mogrt_path, model, letterings_per_minute, mogrt_text_param),
        daemon=True,
    )
    thread.start()

    # Return immediately
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str, request: Request):
    if job_id not in job_statuses:
        return JSONResponse({"error": "job não encontrado"}, status_code=404)

    info = job_statuses[job_id]
    base = str(request.base_url).rstrip("/")
    result = {"status": info["status"]}

    if info["status"] == "completed":
        result["download_url"] = f"{base}/download/{job_id}"
    elif info["status"] == "failed":
        result["error"] = info.get("error", "Erro desconhecido")

    return result


@app.get("/download/{job_id}")
def download(job_id: str):
    zip_path = JOBS_DIR / job_id / "pacote_premiere.zip"
    if not zip_path.exists():
        return JSONResponse({"error": "arquivo não encontrado"}, status_code=404)
    return FileResponse(
        path=zip_path,
        filename="pacote_premiere.zip",
        media_type="application/zip",
    )
