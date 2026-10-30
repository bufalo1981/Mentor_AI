"""FastAPI app for Mentor AI: REST API, the live-interview WebSocket, and the static frontend."""

import logging
import os
import shutil
import sys
from contextlib import asynccontextmanager
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, File, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import agents
import config
import db
import orchestrator
from gemini import friendly_error
from models import (
    JobDescriptionIn,
    JobRatingIn,
    MessageIn,
    RatingsIn,
    SaveJobsIn,
    ToggleSaveIn,
    TopicStatusIn,
)
from voice import turn_based
from voice.live_proxy import LiveInterview

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mentor")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    recovered = orchestrator.recover_interrupted_work()
    if recovered:
        log.warning("Marked %d interrupted background job(s) as failed; the UI offers Retry.", recovered)
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_CLOUD_PROJECT"):
        log.warning("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    yield


app = FastAPI(
    title="Mentor AI",
    description="Multimodal AI agent for job search and voice mock interviews",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------- errors → HTTP status codes ----------

@app.exception_handler(agents.AgentLockedError)
async def locked_handler(_: Request, error: agents.AgentLockedError):
    return JSONResponse(status_code=423, content={"detail": str(error), "agent": error.name})


@app.exception_handler(orchestrator.StateError)
async def state_handler(_: Request, error: orchestrator.StateError):
    return JSONResponse(status_code=409, content={"detail": str(error)})


@app.exception_handler(LookupError)
async def not_found_handler(_: Request, error: LookupError):
    return JSONResponse(status_code=404, content={"detail": str(error)})


@app.exception_handler(ValueError)
async def bad_request_handler(_: Request, error: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.exception_handler(Exception)
async def gemini_handler(_: Request, error: Exception):
    log.exception("Request failed")
    return JSONResponse(status_code=502, content={"detail": friendly_error(error)})


# ---------- payload builders ----------

def search_payload(request: dict) -> dict:
    return {
        "id": request["id"],
        "status": request["status"],
        "round": request["round"],
        "error": request["error"],
        "understanding": db.loads(request["understanding_json"]),
        "messages": db.list_messages(f"search:{request['id']}"),
        "results": db.list_search_results(request["id"], request["round"]) if request["round"] else [],
    }


def interview_payload(session: dict) -> dict:
    reviews = {r["question_id"]: r for r in db.list_reviews(session["id"])}
    questions = []
    for q in db.list_questions(session["id"]):
        review = reviews.get(q["id"])
        if review:
            review = {**review, "rubric": db.loads(review.pop("rubric_json"), {})}
        questions.append({**q, "review": review})
    return {
        "id": session["id"],
        "job_id": session["job_id"],
        "status": session["status"],
        "voice_mode": session["voice_mode"],
        "error": session["error"],
        "research": db.loads(session["research_json"]),
        "review": db.loads(session["review_json"]),
        "summary_message": session["summary_message"],
        "questions": questions,
        "created_at": session["created_at"],
        "completed_at": session["completed_at"],
    }


def job_payload(job: dict, detail: bool = False) -> dict:
    payload = dict(job)
    payload["has_description"] = len((job.get("job_description") or "").strip()) >= orchestrator.MIN_JOB_DESCRIPTION
    if detail:
        session = db.latest_interview_for_job(job["id"])
        payload["topics"] = db.list_topics(job["id"])
        payload["messages"] = db.list_messages(f"job:{job['id']}")
        payload["interview"] = interview_payload(session) if session else None
    return payload


# ---------- health & profile ----------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "agents": agents.status(),
        "voice_mode": config.VOICE_MODE,
        "question_count": config.QUESTION_COUNT,
        "models": {
            "text": config.TEXT_MODEL,
            "live": config.LIVE_MODEL,
            "tts": config.TTS_MODEL,
            "transcribe": config.TRANSCRIBE_MODEL,
        },
    }


def _profile_payload() -> dict:
    profile = db.get_profile()
    return {"has_resume": bool(profile and profile.get("resume_path")),
            "resume_filename": profile and profile.get("resume_filename")}


@app.get("/api/profile")
async def get_profile():
    return _profile_payload()


@app.post("/api/profile/resume")
async def upload_resume(file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename or "")[1].lower()
    mime = config.RESUME_MIME_TYPES.get(extension)
    if mime is None:
        raise ValueError("Upload a PDF, PNG, JPG, or TXT resume.")
    data = await file.read()
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > config.MAX_RESUME_BYTES:
        raise ValueError("Resume must be smaller than 10 MB.")
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / f"resume{extension}"
    path.write_bytes(data)
    db.set_resume(str(path), mime, file.filename)
    return _profile_payload()


@app.post("/api/profile/sample-resume")
async def use_sample_resume():
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / "resume.pdf"
    shutil.copyfile(config.SAMPLE_RESUME, path)
    db.set_resume(str(path), "application/pdf", "sample_resume.pdf (Jordan Rivera)")
    return _profile_payload()


@app.get("/api/profile/resume")
async def download_resume():
    profile = db.get_profile()
    if not profile or not profile.get("resume_path") or not os.path.exists(profile["resume_path"]):
        raise LookupError("No resume uploaded.")
    return FileResponse(profile["resume_path"], media_type=profile["resume_mime"])


# ---------- Flow 1: job search ----------

@app.get("/api/search/latest")
async def latest_search():
    request = db.latest_search_request()
    return search_payload(request) if request else None


@app.post("/api/search")
async def start_search(body: MessageIn):
    request_id = await orchestrator.start_search(body.message)
    return search_payload(db.get_search_request(request_id))


@app.get("/api/search/{request_id}")
async def get_search(request_id: int):
    request = db.get_search_request(request_id)
    if request is None:
        raise LookupError(f"Search {request_id} not found.")
    return search_payload(request)


@app.post("/api/search/{request_id}/messages")
async def chat_search(request_id: int, body: MessageIn):
    await orchestrator.chat_search(request_id, body.message)
    return search_payload(db.get_search_request(request_id))


@app.post("/api/search/{request_id}/confirm")
async def confirm_search(request_id: int):
    orchestrator.confirm_search(request_id)
    return search_payload(db.get_search_request(request_id))


@app.post("/api/search/{request_id}/feedback")
async def search_feedback(request_id: int, body: MessageIn):
    orchestrator.give_feedback(request_id, body.message)
    return search_payload(db.get_search_request(request_id))


@app.post("/api/search/{request_id}/save")
async def save_jobs(request_id: int, body: SaveJobsIn):
    job_ids = orchestrator.save_jobs(request_id, body.result_ids)
    return {"saved_job_ids": job_ids, "search": search_payload(db.get_search_request(request_id))}


@app.post("/api/search/results/{result_id}/toggle-save")
async def toggle_save_result(result_id: int, body: Optional[ToggleSaveIn] = None):
    info = db.toggle_save_search_result(result_id, force_save=bool(body and body.force_save))
    request_id = info["result"]["request_id"]
    return {
        **info,
        "search": search_payload(db.get_search_request(request_id)),
        "jobs": [job_payload(j) for j in db.list_jobs()],
    }


@app.put("/api/search/results/{result_id}/rating")
async def rate_search_result(result_id: int, body: JobRatingIn):
    result = db.set_search_result_rating(result_id, body.rating)
    return {
        "result": result,
        "search": search_payload(db.get_search_request(result["request_id"])),
        "jobs": [job_payload(j) for j in db.list_jobs()],
    }


# ---------- Flow 2: saved jobs, study plan, interview ----------

@app.get("/api/jobs")
async def list_jobs():
    return [job_payload(job) for job in db.list_jobs()]


def _job(job_id: int) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise LookupError(f"Saved job {job_id} not found.")
    return job


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: int):
    return job_payload(_job(job_id), detail=True)


@app.patch("/api/jobs/{job_id}")
async def update_job_description(job_id: int, body: JobDescriptionIn):
    _job(job_id)
    db.update_job(job_id, job_description=body.job_description.strip())
    return job_payload(_job(job_id), detail=True)


@app.put("/api/jobs/{job_id}/rating")
async def rate_saved_job(job_id: int, body: JobRatingIn):
    _job(job_id)
    db.set_job_rating(job_id, body.rating)
    return job_payload(_job(job_id), detail=True)


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: int):
    _job(job_id)
    db.delete_job(job_id)
    return {"deleted": job_id}


@app.post("/api/jobs/{job_id}/topics")
async def load_topics(job_id: int):
    await orchestrator.load_topics(job_id)
    return job_payload(_job(job_id), detail=True)


@app.post("/api/jobs/{job_id}/ratings")
async def rate_topics(job_id: int, body: RatingsIn):
    await orchestrator.rate_topics(job_id, body.ratings)
    return job_payload(_job(job_id), detail=True)


@app.patch("/api/topics/{topic_id}")
async def set_topic_status(topic_id: int, body: TopicStatusIn):
    orchestrator.set_topic_status(topic_id, body.status)
    return db.get_topic(topic_id)


@app.post("/api/jobs/{job_id}/study-done")
async def finish_studying(job_id: int):
    orchestrator.finish_studying(job_id)
    return job_payload(_job(job_id), detail=True)


@app.post("/api/jobs/{job_id}/interviews")
async def new_interview(job_id: int):
    orchestrator.new_interview(job_id)
    return job_payload(_job(job_id), detail=True)


def _interview(session_id: int) -> dict:
    session = db.get_interview(session_id)
    if session is None:
        raise LookupError(f"Interview {session_id} not found.")
    return session


@app.get("/api/interviews/{session_id}")
async def get_interview(session_id: int):
    return interview_payload(_interview(session_id))


@app.post("/api/interviews/{session_id}/finish")
async def finish_interview(session_id: int):
    orchestrator.finish_interview(session_id)
    return interview_payload(_interview(session_id))


@app.post("/api/interviews/{session_id}/retry")
async def retry_interview(session_id: int):
    orchestrator.retry_interview(session_id)
    return interview_payload(_interview(session_id))


@app.get("/api/interviews/{session_id}/turn/{position}")
async def question_turn(session_id: int, position: int):
    return await turn_based.question_turn(session_id, position)


@app.post("/api/interviews/{session_id}/turn/{position}/answer")
async def answer_turn(session_id: int, position: int, audio: UploadFile = File(...)):
    return await turn_based.answer_turn(session_id, position, await audio.read(), audio.content_type)


@app.websocket("/ws/interviews/{session_id}")
async def live_interview(websocket: WebSocket, session_id: int):
    await LiveInterview(websocket, session_id).run()


# ---------- frontend ----------

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
