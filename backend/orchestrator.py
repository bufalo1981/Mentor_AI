"""Moves each flow from one status to the next and starts the right agent.

Code — not the model — decides what happens next. The Summary Agent runs in the foreground
(the user waits for its reply); the Web Search and Review agents run as background tasks and
the frontend polls for their results. Status diagrams are in db.md.
"""

import asyncio
import json
import logging
from typing import Coroutine

import agents
import config
import db
from agents import review, summary, web_search
from gemini import friendly_error

log = logging.getLogger("mentor.orchestrator")

MIN_JOB_DESCRIPTION = 200

_background: set[asyncio.Task] = set()


class StateError(Exception):
    """The request does not fit the current status (e.g. confirming a search twice)."""


RESTART_MESSAGE = "Interrupted by a server restart — press Retry."


def recover_interrupted_work() -> int:
    """On startup: background work that was running when the server stopped can never finish.

    Marks it failed (with a Retry hint) instead of leaving a spinner that never ends.
    Returns how many rows were recovered.
    """
    recovered = 0
    for request in db.query("SELECT id FROM search_requests WHERE status = 'searching'"):
        db.update_search_request(request["id"], status="failed", error=RESTART_MESSAGE)
        recovered += 1
    for session in db.query(
        "SELECT id FROM interview_sessions WHERE status IN ('researching', 'composing', 'reviewing')"
    ):
        db.update_interview(session["id"], status="failed", error=RESTART_MESSAGE)
        recovered += 1
    return recovered


def run_in_background(coro: Coroutine) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)
    return task


def _profile() -> dict:
    profile = db.get_profile()
    if not profile or not profile.get("resume_path"):
        raise ValueError("Upload your resume first (or click 'Use sample resume').")
    return profile


def _search_thread(request_id: int) -> str:
    return f"search:{request_id}"


def _job_thread(job_id: int) -> str:
    return f"job:{job_id}"


def _get_search(request_id: int) -> dict:
    request = db.get_search_request(request_id)
    if request is None:
        raise LookupError(f"Search {request_id} not found.")
    return request


def _get_job(job_id: int) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise LookupError(f"Saved job {job_id} not found.")
    return job


def _get_interview(session_id: int) -> dict:
    session = db.get_interview(session_id)
    if session is None:
        raise LookupError(f"Interview {session_id} not found.")
    return session


# ============================================================
# Flow 1 — Job search
#   collecting ⟲ (chat) → searching → results_ready ⟲ (feedback → searching) → saved
#   any background failure → failed (confirm again to retry)
# ============================================================

async def start_search(message: str) -> int:
    agents.require("summary")
    _profile()
    request_id = db.create_search_request()
    await chat_search(request_id, message)
    return request_id


async def chat_search(request_id: int, message: str) -> None:
    """User talks to the Summary Agent until it understands the request."""
    agents.require("summary")
    request = _get_search(request_id)
    if request["status"] not in ("collecting", "failed"):
        raise StateError("This search is already running. Give feedback on the results instead.")
    thread = _search_thread(request_id)
    db.add_message(thread, "user", message)
    reply = await summary.understand_request(_profile(), db.list_messages(thread))
    db.add_message(thread, "summary", reply.reply)
    db.update_search_request(
        request_id,
        status="collecting",
        understanding_json=reply.understanding.model_dump_json(),
        error=None,
    )


def confirm_search(request_id: int) -> None:
    """User pressed "Yes, search": hand the confirmed understanding to the Web Search Agent."""
    agents.require("summary", "web_search")
    request = _get_search(request_id)
    if request["status"] not in ("collecting", "failed") or not request["understanding_json"]:
        raise StateError("Nothing to confirm yet — tell the Summary Agent what you are looking for.")
    _start_search_round(request_id, request["round"] + 1)


def give_feedback(request_id: int, message: str) -> None:
    """User is not satisfied: pass the feedback to the Web Search Agent and search again."""
    agents.require("summary", "web_search")
    request = _get_search(request_id)
    if request["status"] not in ("results_ready", "saved"):
        raise StateError("Feedback can only be given on results.")
    db.add_message(_search_thread(request_id), "feedback", message)
    _start_search_round(request_id, request["round"] + 1)


def _start_search_round(request_id: int, round_no: int) -> None:
    db.update_search_request(request_id, status="searching", round=round_no, error=None)
    run_in_background(run_job_search(request_id, round_no))


async def run_job_search(request_id: int, round_no: int) -> None:
    """Background: Web Search Agent finds posts → Summary Agent explains them."""
    thread = _search_thread(request_id)
    try:
        request = _get_search(request_id)
        preferences = db.loads(request["understanding_json"], {})
        feedback = [m["content"] for m in db.list_messages(thread) if m["role"] == "feedback"]
        already_shown = db.list_search_results(request_id)

        jobs, note = await web_search.find_jobs(preferences, feedback, already_shown)
        if not jobs:
            raise ValueError(
                "The Web Search Agent found no new matching posts, even after widening the search. "
                "Try a different role title or location."
            )

        presentation = await summary.present_jobs(_profile(), preferences, jobs, note)
        for pick in presentation.jobs:
            if 1 <= pick.index <= len(jobs):
                jobs[pick.index - 1]["summary"] = pick.summary
                jobs[pick.index - 1]["why_relevant"] = pick.why_relevant

        db.add_search_results(request_id, round_no, jobs)
        db.add_message(thread, "summary", presentation.message)
        db.update_search_request(request_id, status="results_ready")
    except Exception as error:  # background task: record the failure for the UI
        log.exception("Job search failed")
        db.update_search_request(request_id, status="failed", error=friendly_error(error))


def save_jobs(request_id: int, result_ids: list[int]) -> list[int]:
    """User is satisfied: store the chosen posts as saved jobs."""
    request = _get_search(request_id)
    if request["status"] not in ("results_ready", "saved"):
        raise StateError("There are no results to save yet.")
    job_ids = db.save_jobs(request_id, result_ids)
    if not job_ids:
        raise ValueError("None of the selected posts belong to this search.")
    db.update_search_request(request_id, status="saved")
    return job_ids


# ============================================================
# Flow 2 — Interview prep for one saved job
#   study_status:     not_started → studying → done
#   interview_status: not_started → started → done
#   interview session (while "started"):
#     researching → composing → ready → in_progress → reviewing → done   (any step → failed)
# ============================================================

def _require_description(job: dict) -> None:
    if len((job.get("job_description") or "").strip()) < MIN_JOB_DESCRIPTION:
        raise ValueError(
            "This job has no full job description yet. Paste it on the job card first — "
            "every interview step is built from it."
        )


async def load_topics(job_id: int) -> None:
    """Summary Agent lists what the job expects the user to know."""
    agents.require("summary")
    job = _get_job(job_id)
    _require_description(job)
    result = await summary.list_topics(_profile(), job)
    db.replace_topics(job_id, [t.model_dump() for t in result.topics])
    db.add_message(_job_thread(job_id), "summary", result.message)


async def rate_topics(job_id: int, ratings: dict[int, int]) -> None:
    """User rated each topic; Summary Agent turns that into study-first / review-later."""
    agents.require("summary")
    job = _get_job(job_id)
    topics = db.list_topics(job_id)
    if not topics:
        raise StateError("Load the topics first.")
    for topic in topics:
        if topic["id"] in ratings:
            db.update_topic(topic["id"], self_rating=ratings[topic["id"]])
    topics = db.list_topics(job_id)

    plan = await summary.make_study_plan(_profile(), job, topics)
    for item in plan.items:
        if 1 <= item.topic_number <= len(topics):
            topic = topics[item.topic_number - 1]
            db.update_topic(topic["id"], priority=item.priority, plan=item.plan, status="studying")
    db.update_job(job_id, study_status="studying", study_message=plan.message)
    db.add_message(_job_thread(job_id), "summary", plan.message)


def set_topic_status(topic_id: int, status: str) -> None:
    if db.get_topic(topic_id) is None:
        raise LookupError(f"Topic {topic_id} not found.")
    db.update_topic(topic_id, status=status)


def finish_studying(job_id: int) -> int:
    """User is done studying: study → done, interview → started, start interview research."""
    agents.require("summary", "web_search")
    job = _get_job(job_id)
    _require_description(job)
    if job["study_status"] != "studying":
        raise StateError("Make a study plan first.")
    db.update_job(job_id, study_status="done", interview_status="started")
    session_id = db.create_interview(job_id, config.VOICE_MODE)
    run_in_background(prepare_interview(session_id))
    return session_id


async def prepare_interview(session_id: int) -> None:
    """Background: Web Search Agent researches real questions → Summary Agent writes the interview."""
    try:
        session = _get_interview(session_id)
        job = _get_job(session["job_id"])

        db.update_interview(session_id, status="researching", error=None)
        research = await web_search.find_interview_reports(job)
        db.update_interview(session_id, status="composing", research_json=_dumps(research))

        plan = await summary.compose_interview(_profile(), job, research)
        questions = [q.model_dump() for q in plan.questions][: config.QUESTION_COUNT]
        if not questions:
            raise ValueError("The Summary Agent returned no questions. Press Retry.")
        db.add_questions(session_id, questions)
        db.add_message(_job_thread(job["id"]), "summary", plan.message)
        db.update_interview(session_id, status="ready")
    except Exception as error:
        log.exception("Interview preparation failed")
        db.update_interview(session_id, status="failed", error=friendly_error(error))


def begin_interview(session_id: int) -> dict:
    """Mock Interview Agent is about to speak. Returns the session."""
    agents.require("mock_interview")
    session = _get_interview(session_id)
    if session["status"] not in ("ready", "in_progress"):
        raise StateError("This interview is not ready to start.")
    if session["status"] == "ready":
        db.update_interview(session_id, status="in_progress", started_at=db.now())
    return _get_interview(session_id)


def finish_interview(session_id: int) -> None:
    """All questions asked (or the user ended early): hand the transcripts to the Review Agent."""
    agents.require("summary", "review")
    session = _get_interview(session_id)
    if session["status"] != "in_progress":
        raise StateError("This interview is not in progress.")
    questions = db.list_questions(session_id)
    if not any((q.get("transcript") or "").strip() for q in questions):
        raise ValueError("No answers were recorded yet — answer at least one question first.")
    db.update_interview(session_id, status="reviewing", error=None)
    run_in_background(run_review(session_id))


async def run_review(session_id: int) -> None:
    """Background: Review Agent grades every answer → Summary Agent presents the result."""
    try:
        session = _get_interview(session_id)
        job = _get_job(session["job_id"])
        questions = db.list_questions(session_id)
        db.update_interview(session_id, status="reviewing", error=None)

        result = await review.review_interview(_profile(), job, questions)
        by_position = {q["position"]: q for q in questions}
        scores = []
        for item in result.reviews:
            question = by_position.get(item.question_number)
            if question is None:
                continue
            score = item.rubric.overall()
            scores.append(score)
            db.save_review(question["id"], score, item.rubric.model_dump(), item.strengths,
                           item.improvements, item.senior_lens)

        presentation = await summary.present_review(job, result, scores)
        db.update_interview(
            session_id,
            status="done",
            completed_at=db.now(),
            review_json=_dumps({**result.model_dump(exclude={"reviews"}), "prepare_next": presentation.prepare_next}),
            summary_message=presentation.message,
        )
        db.update_job(job["id"], interview_status="done")
        db.add_message(_job_thread(job["id"]), "summary", presentation.message)
    except Exception as error:
        log.exception("Interview review failed")
        db.update_interview(session_id, status="failed", error=friendly_error(error))


def retry_interview(session_id: int) -> None:
    """Re-run whichever background step failed."""
    session = _get_interview(session_id)
    if session["status"] != "failed":
        raise StateError("Only a failed step can be retried.")
    has_answers = any((q.get("transcript") or "").strip() for q in db.list_questions(session_id))
    if has_answers:
        agents.require("summary", "review")
        db.update_interview(session_id, status="reviewing", error=None)
        run_in_background(run_review(session_id))
    else:
        agents.require("summary", "web_search")
        db.update_interview(session_id, status="researching", error=None)
        run_in_background(prepare_interview(session_id))


def new_interview(job_id: int) -> int:
    """Take the mock interview again for the same job (a fresh session with new questions)."""
    agents.require("summary", "web_search")
    job = _get_job(job_id)
    if job["study_status"] != "done":
        raise StateError("Finish studying first.")
    db.update_job(job_id, interview_status="started")
    session_id = db.create_interview(job_id, config.VOICE_MODE)
    run_in_background(prepare_interview(session_id))
    return session_id


def _dumps(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False)
