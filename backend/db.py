"""SQLite storage: one local file (data/mentor.db), one user, no ORM. Schema is documented in db.md."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Optional

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    resume_path      TEXT,
    resume_mime      TEXT,
    resume_filename  TEXT,
    updated_at       TEXT
);

CREATE TABLE IF NOT EXISTS search_requests (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    status              TEXT NOT NULL DEFAULT 'collecting',
    understanding_json  TEXT,
    round               INTEGER NOT NULL DEFAULT 0,
    error               TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread      TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id       INTEGER NOT NULL REFERENCES search_requests(id) ON DELETE CASCADE,
    round            INTEGER NOT NULL,
    title            TEXT NOT NULL,
    company          TEXT NOT NULL,
    location         TEXT,
    level            TEXT,
    url              TEXT,
    source           TEXT,
    posted_date      TEXT,
    job_description  TEXT,
    summary          TEXT,
    why_relevant     TEXT,
    fit_score        INTEGER,
    user_rating      INTEGER,
    status           TEXT NOT NULL DEFAULT 'shown',
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS saved_jobs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    search_result_id  INTEGER,
    title             TEXT NOT NULL,
    company           TEXT NOT NULL,
    location          TEXT,
    level             TEXT,
    url               TEXT,
    source            TEXT,
    posted_date       TEXT,
    job_description   TEXT,
    why_relevant      TEXT,
    fit_score         INTEGER,
    user_rating       INTEGER,
    study_status      TEXT NOT NULL DEFAULT 'not_started',
    interview_status  TEXT NOT NULL DEFAULT 'not_started',
    study_message     TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS study_topics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           INTEGER NOT NULL REFERENCES saved_jobs(id) ON DELETE CASCADE,
    position         INTEGER NOT NULL,
    name             TEXT NOT NULL,
    why_needed       TEXT,
    resume_evidence  TEXT,
    self_rating      INTEGER,
    priority         TEXT,
    plan             TEXT,
    status           TEXT NOT NULL DEFAULT 'not_started'
);

CREATE TABLE IF NOT EXISTS interview_sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           INTEGER NOT NULL REFERENCES saved_jobs(id) ON DELETE CASCADE,
    status           TEXT NOT NULL DEFAULT 'researching',
    voice_mode       TEXT NOT NULL,
    research_json    TEXT,
    review_json      TEXT,
    summary_message  TEXT,
    error            TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    completed_at     TEXT
);

CREATE TABLE IF NOT EXISTS interview_questions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER NOT NULL REFERENCES interview_sessions(id) ON DELETE CASCADE,
    position       INTEGER NOT NULL,
    question       TEXT NOT NULL,
    source         TEXT NOT NULL,
    source_url     TEXT,
    what_it_tests  TEXT,
    asked_text     TEXT,
    transcript     TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id    INTEGER NOT NULL UNIQUE REFERENCES interview_questions(id) ON DELETE CASCADE,
    score          INTEGER NOT NULL,
    rubric_json    TEXT NOT NULL,
    strengths      TEXT,
    improvements   TEXT,
    senior_lens    TEXT,
    created_at     TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Columns added after v1.0: (table, column, type). Added to existing databases on startup.
MIGRATIONS = [
    ("search_results", "fit_score", "INTEGER"),
    ("saved_jobs", "fit_score", "INTEGER"),
    ("search_results", "user_rating", "INTEGER"),
    ("saved_jobs", "user_rating", "INTEGER"),
]


def init_db() -> None:
    with closing(connect()) as conn:
        conn.executescript(SCHEMA)
        for table, column, kind in MIGRATIONS:
            columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        conn.commit()


def query(sql: str, params: tuple = ()) -> list[dict]:
    with closing(connect()) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def query_one(sql: str, params: tuple = ()) -> Optional[dict]:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> int:
    """Run one write statement and return the new row id."""
    with closing(connect()) as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.lastrowid


def _update(table: str, row_id: int, fields: dict[str, Any]) -> None:
    if not fields:
        return
    for key in fields:
        if not key.isidentifier():
            raise ValueError(f"Invalid column name: {key}")
    assignments = ", ".join(f"{key} = ?" for key in fields)
    execute(f"UPDATE {table} SET {assignments} WHERE id = ?", (*fields.values(), row_id))


def loads(value: Optional[str], default: Any = None) -> Any:
    return json.loads(value) if value else default


# ---------- profile ----------

def get_profile() -> Optional[dict]:
    return query_one("SELECT * FROM profile WHERE id = 1")


def set_resume(path: str, mime: str, filename: str) -> None:
    execute(
        """INSERT INTO profile (id, resume_path, resume_mime, resume_filename, updated_at)
           VALUES (1, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET resume_path = excluded.resume_path,
               resume_mime = excluded.resume_mime, resume_filename = excluded.resume_filename,
               updated_at = excluded.updated_at""",
        (path, mime, filename, now()),
    )


# ---------- messages (summary agent chat threads) ----------

def add_message(thread: str, role: str, content: str) -> int:
    return execute(
        "INSERT INTO messages (thread, role, content, created_at) VALUES (?, ?, ?, ?)",
        (thread, role, content, now()),
    )


def list_messages(thread: str) -> list[dict]:
    return query("SELECT * FROM messages WHERE thread = ? ORDER BY id", (thread,))


# ---------- job search ----------

def create_search_request() -> int:
    ts = now()
    return execute(
        "INSERT INTO search_requests (status, created_at, updated_at) VALUES ('collecting', ?, ?)",
        (ts, ts),
    )


def get_search_request(request_id: int) -> Optional[dict]:
    return query_one("SELECT * FROM search_requests WHERE id = ?", (request_id,))


def latest_search_request() -> Optional[dict]:
    return query_one("SELECT * FROM search_requests ORDER BY id DESC LIMIT 1")


def update_search_request(request_id: int, **fields: Any) -> None:
    _update("search_requests", request_id, {**fields, "updated_at": now()})


def add_search_results(request_id: int, round_no: int, jobs: list[dict]) -> list[int]:
    ids = []
    for job in jobs:
        ids.append(execute(
            """INSERT INTO search_results (request_id, round, title, company, location, level, url, source,
                   posted_date, job_description, summary, why_relevant, fit_score, user_rating, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (request_id, round_no, job["title"], job["company"], job.get("location"), job.get("level"),
             job.get("url"), job.get("source"), job.get("posted_date"), job.get("job_description"),
             job.get("summary"), job.get("why_relevant"), job.get("fit_score"), job.get("user_rating"), now()),
        ))
    return ids


def _enrich_search_results(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    saved_map = {
        r["search_result_id"]: r["id"]
        for r in query("SELECT id, search_result_id FROM saved_jobs WHERE search_result_id IS NOT NULL")
    }
    for row in rows:
        row["saved_job_id"] = saved_map.get(row["id"])
    return rows


def get_search_result(result_id: int) -> Optional[dict]:
    row = query_one("SELECT * FROM search_results WHERE id = ?", (result_id,))
    if row is None:
        return None
    return _enrich_search_results([row])[0]


def list_search_results(request_id: int, round_no: Optional[int] = None) -> list[dict]:
    if round_no is None:
        rows = query("SELECT * FROM search_results WHERE request_id = ? ORDER BY id", (request_id,))
    else:
        rows = query(
            "SELECT * FROM search_results WHERE request_id = ? AND round = ? ORDER BY id",
            (request_id, round_no),
        )
    return _enrich_search_results(rows)


def update_search_result(result_id: int, **fields: Any) -> None:
    _update("search_results", result_id, fields)


def save_jobs(request_id: int, result_ids: list[int]) -> list[int]:
    """Copy chosen search results into saved_jobs. Returns the saved job ids."""
    job_ids = []
    for result_id in result_ids:
        result = query_one(
            "SELECT * FROM search_results WHERE id = ? AND request_id = ?", (result_id, request_id)
        )
        if result is None:
            continue
        existing = query_one("SELECT id FROM saved_jobs WHERE search_result_id = ?", (result_id,))
        if existing:
            update_search_result(result_id, status="saved")
            job_ids.append(existing["id"])
            continue
        ts = now()
        job_ids.append(execute(
            """INSERT INTO saved_jobs (search_result_id, title, company, location, level, url, source,
                   posted_date, job_description, why_relevant, fit_score, user_rating, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (result_id, result["title"], result["company"], result["location"], result["level"],
             result["url"], result["source"], result["posted_date"], result["job_description"],
             result["why_relevant"], result["fit_score"], result.get("user_rating"), ts, ts),
        ))
        update_search_result(result_id, status="saved")
    return job_ids


def toggle_save_search_result(result_id: int, force_save: bool = False) -> dict:
    """Save or unsave a single search result and keep saved_jobs + search_results synchronized."""
    result = query_one("SELECT * FROM search_results WHERE id = ?", (result_id,))
    if result is None:
        raise LookupError(f"Search result {result_id} not found.")
    existing = query_one("SELECT id FROM saved_jobs WHERE search_result_id = ?", (result_id,))
    if existing and not force_save:
        delete_job(existing["id"])
        return {"saved": False, "saved_job_id": None, "result": get_search_result(result_id)}
    if existing and force_save:
        update_search_result(result_id, status="saved")
        return {"saved": True, "saved_job_id": existing["id"], "result": get_search_result(result_id)}
    job_ids = save_jobs(result["request_id"], [result_id])
    saved_id = job_ids[0] if job_ids else None
    return {"saved": True, "saved_job_id": saved_id, "result": get_search_result(result_id)}


def set_search_result_rating(result_id: int, rating: Optional[int]) -> dict:
    result = query_one("SELECT * FROM search_results WHERE id = ?", (result_id,))
    if result is None:
        raise LookupError(f"Search result {result_id} not found.")
    update_search_result(result_id, user_rating=rating)
    existing = query_one("SELECT id FROM saved_jobs WHERE search_result_id = ?", (result_id,))
    if existing:
        update_job(existing["id"], user_rating=rating)
    return get_search_result(result_id)


def set_job_rating(job_id: int, rating: Optional[int]) -> dict:
    job = get_job(job_id)
    if job is None:
        raise LookupError(f"Saved job {job_id} not found.")
    update_job(job_id, user_rating=rating)
    if job.get("search_result_id"):
        res = query_one("SELECT id FROM search_results WHERE id = ?", (job["search_result_id"],))
        if res:
            update_search_result(res["id"], user_rating=rating)
    return get_job(job_id)


# ---------- saved jobs & study plan ----------

def list_jobs() -> list[dict]:
    return query("SELECT * FROM saved_jobs ORDER BY id DESC")


def get_job(job_id: int) -> Optional[dict]:
    return query_one("SELECT * FROM saved_jobs WHERE id = ?", (job_id,))


def update_job(job_id: int, **fields: Any) -> None:
    _update("saved_jobs", job_id, {**fields, "updated_at": now()})


def delete_job(job_id: int) -> None:
    job = get_job(job_id)
    if job and job.get("search_result_id"):
        res = query_one("SELECT id FROM search_results WHERE id = ?", (job["search_result_id"],))
        if res:
            update_search_result(res["id"], status="shown")
    execute("DELETE FROM saved_jobs WHERE id = ?", (job_id,))


def replace_topics(job_id: int, topics: list[dict]) -> None:
    execute("DELETE FROM study_topics WHERE job_id = ?", (job_id,))
    for position, topic in enumerate(topics, start=1):
        execute(
            """INSERT INTO study_topics (job_id, position, name, why_needed, resume_evidence)
               VALUES (?, ?, ?, ?, ?)""",
            (job_id, position, topic["name"], topic.get("why_needed"), topic.get("resume_evidence")),
        )


def list_topics(job_id: int) -> list[dict]:
    return query("SELECT * FROM study_topics WHERE job_id = ? ORDER BY position", (job_id,))


def get_topic(topic_id: int) -> Optional[dict]:
    return query_one("SELECT * FROM study_topics WHERE id = ?", (topic_id,))


def update_topic(topic_id: int, **fields: Any) -> None:
    _update("study_topics", topic_id, fields)


# ---------- interviews ----------

def create_interview(job_id: int, voice_mode: str) -> int:
    return execute(
        "INSERT INTO interview_sessions (job_id, status, voice_mode, created_at) VALUES (?, 'researching', ?, ?)",
        (job_id, voice_mode, now()),
    )


def get_interview(session_id: int) -> Optional[dict]:
    return query_one("SELECT * FROM interview_sessions WHERE id = ?", (session_id,))


def latest_interview_for_job(job_id: int) -> Optional[dict]:
    return query_one(
        "SELECT * FROM interview_sessions WHERE job_id = ? ORDER BY id DESC LIMIT 1", (job_id,)
    )


def update_interview(session_id: int, **fields: Any) -> None:
    _update("interview_sessions", session_id, fields)


def add_questions(session_id: int, questions: list[dict]) -> None:
    execute("DELETE FROM interview_questions WHERE session_id = ?", (session_id,))
    for position, q in enumerate(questions, start=1):
        execute(
            """INSERT INTO interview_questions (session_id, position, question, source, source_url, what_it_tests)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, position, q["question"], q["source"], q.get("source_url"), q.get("what_it_tests")),
        )


def list_questions(session_id: int) -> list[dict]:
    return query(
        "SELECT * FROM interview_questions WHERE session_id = ? ORDER BY position", (session_id,)
    )


def get_question(session_id: int, position: int) -> Optional[dict]:
    return query_one(
        "SELECT * FROM interview_questions WHERE session_id = ? AND position = ?", (session_id, position)
    )


def update_question(question_id: int, **fields: Any) -> None:
    _update("interview_questions", question_id, fields)


def save_review(question_id: int, score: int, rubric: dict, strengths: str, improvements: str,
                senior_lens: str) -> None:
    execute(
        """INSERT INTO reviews (question_id, score, rubric_json, strengths, improvements, senior_lens, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(question_id) DO UPDATE SET score = excluded.score, rubric_json = excluded.rubric_json,
               strengths = excluded.strengths, improvements = excluded.improvements,
               senior_lens = excluded.senior_lens, created_at = excluded.created_at""",
        (question_id, score, json.dumps(rubric), strengths, improvements, senior_lens, now()),
    )


def list_reviews(session_id: int) -> list[dict]:
    return query(
        """SELECT r.*, q.position FROM reviews r
           JOIN interview_questions q ON q.id = r.question_id
           WHERE q.session_id = ? ORDER BY q.position""",
        (session_id,),
    )
