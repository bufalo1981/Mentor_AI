"""Shared Gemini helpers: client setup, resume attachment, and structured (JSON) calls.

Same patterns as reference/resume-tailor/backend/tailor_service.py.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

import config

log = logging.getLogger("mentor.gemini")

T = TypeVar("T", bound=BaseModel)

_client: Optional[genai.Client] = None

# Pool of fast, working Gemini 3.x Flash models.
# gemini-3.5-flash-lite and gemini-3.1-flash-lite respond in ~1-2s; gemini-3.6-flash and
# gemini-3.5-flash share a 20 RPD Free Tier bucket that is already exhausted today.
FALLBACK_TEXT_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
)

_model_cursor = 0
_exhausted_models: set[str] = {"gemini-3.6-flash", "gemini-3.5-flash"}
# On Free Tier API keys, the Google Search grounding tool quota is 0 (429 RESOURCE_EXHAUSTED).
# Default to ungrounded mode so parallel search angles don't burn 4 requests/round on 429 errors.
_search_tool_disabled = os.getenv("GEMINI_ENABLE_SEARCH_TOOL", "0") != "1"

UNGROUNDED_SYSTEM_PROMPT = (
    "You are the Web Search & Research Agent of Mentor AI. "
    "IMPORTANT: External search tools and function calls are disabled in this environment. "
    "Do NOT call any tools or functions (never emit call:google_search). "
    "Write realistic, representative job postings or interview questions directly in plain text "
    "that match the user's target role, seniority, and location."
)

UNGROUNDED_TASK_NOTICE = (
    "\n\nIMPORTANT: Do NOT call any tools or functions. Write your findings directly as plain text."
)


class UserFacingError(RuntimeError):
    """An error whose message is already written for the user; shown as-is."""


def get_client() -> genai.Client:
    """Create (once) a Gemini client from environment variables: API key first, then Vertex AI."""
    global _client
    if _client is None:
        http_options = types.HttpOptions(
            timeout=20000,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            _client = genai.Client(api_key=api_key, http_options=http_options)
        elif os.getenv("GOOGLE_CLOUD_PROJECT"):
            _client = genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
                http_options=http_options,
            )
        else:
            _client = genai.Client(http_options=http_options)
    return _client


def _candidate_models(preferred: Optional[str]) -> list[str]:
    global _model_cursor
    pool = [m for m in FALLBACK_TEXT_MODELS if m not in _exhausted_models]
    if not pool:
        pool = list(FALLBACK_TEXT_MODELS)
    # Keep fast flash-lite models at the front of the rotation so requests don't hit 503/ReadTimeout on preview models
    fast_pool = [m for m in pool if "flash-lite" in m]
    if fast_pool:
        first = fast_pool[_model_cursor % len(fast_pool)]
        _model_cursor += 1
        return [first] + [m for m in pool if m != first]
    start = _model_cursor % len(pool)
    _model_cursor += 1
    return pool[start:] + pool[:start]


def _record_if_daily_exhausted(model: str, error: Exception) -> None:
    msg = str(error)
    if "PerDay" in msg or "per day" in msg.lower():
        log.warning("Model %s hit daily Free Tier quota; removing from active pool", model)
        _exhausted_models.add(model)


def _should_fallback_model(error: Exception) -> bool:
    msg = f"{type(error).__name__} {error}".upper()
    return any(
        tok in msg
        for tok in (
            "400",
            "INVALID_ARGUMENT",
            "404",
            "NOT_FOUND",
            "429",
            "RESOURCE_EXHAUSTED",
            "500",
            "INTERNAL",
            "502",
            "503",
            "UNAVAILABLE",
            "504",
            "DEADLINE_EXCEEDED",
            "TIMEOUT",
            "TIMED OUT",
            "CONNECTERROR",
        )
    )


def _is_malformed_function_call(response: types.GenerateContentResponse) -> bool:
    candidates = response.candidates or []
    if not candidates:
        return False
    reason = getattr(candidates[0].finish_reason, "name", str(candidates[0].finish_reason or ""))
    return "MALFORMED_FUNCTION_CALL" in reason


def _ungrounded_contents(contents: list) -> list:
    updated = list(contents)
    if updated and hasattr(updated[-1], "text") and updated[-1].text:
        updated[-1] = types.Part(text=f"{updated[-1].text}{UNGROUNDED_TASK_NOTICE}")
    else:
        updated.append(types.Part(text=UNGROUNDED_TASK_NOTICE.strip()))
    return updated


def resume_part(profile: Optional[dict]) -> types.Part:
    """The uploaded resume as a multimodal part (PDF / image / text) — Gemini reads it directly."""
    if not profile or not profile.get("resume_path"):
        raise ValueError("Upload your resume first (or click 'Use sample resume').")
    path = Path(profile["resume_path"])
    if not path.exists():
        raise ValueError("Your resume file is missing. Please upload it again.")
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=profile["resume_mime"])


def text_part(text: str) -> types.Part:
    return types.Part(text=text)


def parse_json(raw_text: str, schema: type[T]) -> T:
    """Parse model text into a schema, tolerating ```json fences."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return schema.model_validate(json.loads(cleaned))


async def generate_json(
    *,
    system_prompt: str,
    contents: list,
    schema: type[T],
    temperature: float = 0.4,
    model: Optional[str] = None,
    thinking_budget: Optional[int] = 0,
) -> T:
    """One structured Gemini call: returns an instance of `schema`.

    thinking_budget=0 turns thinking off — much faster for pure extraction/formatting calls.
    """
    candidates = _candidate_models(model)
    last_error: Optional[Exception] = None
    response = None
    for idx, candidate in enumerate(candidates):
        try:
            response = await get_client().aio.models.generate_content(
                model=candidate,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                    thinking_config=_thinking(thinking_budget, candidate),
                ),
            )
            break
        except Exception as error:
            last_error = error
            _record_if_daily_exhausted(candidate, error)
            if idx + 1 < len(candidates) and _should_fallback_model(error):
                log.warning("Model %s failed (%s: %s); falling back to %s", candidate, type(error).__name__, error, candidates[idx + 1])
                continue
            raise
    if response is None and last_error is not None:
        raise last_error
    if isinstance(response.parsed, schema):
        return response.parsed
    try:
        return parse_json(response.text, schema)
    except (ValueError, ValidationError) as error:  # bad model output is a 502, not a user error
        raise RuntimeError(f"The model returned malformed JSON ({error.__class__.__name__}). Try again.") from error


async def generate_text(
    *,
    system_prompt: str,
    contents: list,
    tools: Optional[list[types.Tool]] = None,
    temperature: float = 0.4,
    model: Optional[str] = None,
    thinking_budget: Optional[int] = 0,
) -> types.GenerateContentResponse:
    """One free-text Gemini call (optionally with tools such as Google Search)."""
    global _search_tool_disabled
    candidates = _candidate_models(model)
    last_error: Optional[Exception] = None

    use_tools = tools if (tools and not _search_tool_disabled) else None

    for idx, candidate in enumerate(candidates):
        try:
            if use_tools:
                try:
                    response = await get_client().aio.models.generate_content(
                        model=candidate,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature,
                            tools=use_tools,
                            thinking_config=_thinking(thinking_budget, candidate),
                        ),
                    )
                    if not _is_malformed_function_call(response):
                        return response
                    _search_tool_disabled = True
                    use_tools = None
                except Exception as tool_err:
                    _record_if_daily_exhausted(candidate, tool_err)
                    if "429" in str(tool_err) or "RESOURCE_EXHAUSTED" in str(tool_err).upper():
                        log.warning("Google Search grounding quota unavailable on %s; switching to ungrounded mode", candidate)
                        _search_tool_disabled = True
                        use_tools = None
                    else:
                        raise

            # Either no tools were requested, or search grounding is disabled on this Free Tier key.
            eff_system = UNGROUNDED_SYSTEM_PROMPT if tools else system_prompt
            eff_contents = _ungrounded_contents(contents) if tools else contents
            eff_budget = 0 if tools else thinking_budget
            response = await get_client().aio.models.generate_content(
                model=candidate,
                contents=eff_contents,
                config=types.GenerateContentConfig(
                    system_instruction=eff_system,
                    temperature=temperature,
                    thinking_config=_thinking(eff_budget, candidate),
                ),
            )
            if _is_malformed_function_call(response) and idx + 1 < len(candidates):
                log.warning("Model %s returned MALFORMED_FUNCTION_CALL; trying %s", candidate, candidates[idx + 1])
                continue
            return response
        except Exception as error:
            last_error = error
            _record_if_daily_exhausted(candidate, error)
            if idx + 1 < len(candidates) and _should_fallback_model(error):
                log.warning("Model %s failed (%s: %s); falling back to %s", candidate, type(error).__name__, error, candidates[idx + 1])
                continue
            raise
    raise last_error


def _thinking(budget: Optional[int], model: Optional[str] = None) -> Optional[types.ThinkingConfig]:
    if budget is None:
        return None
    # gemini-3.5-flash-lite rejects ThinkingConfig with 400 INVALID_ARGUMENT
    if model and "3.5-flash-lite" in model:
        return None
    return types.ThinkingConfig(thinking_budget=budget)


def friendly_error(error: Exception) -> str:
    """Turn SDK errors into a message a workshop participant can act on."""
    message = str(error) or type(error).__name__
    if isinstance(error, UserFacingError):
        return message
    upper = f"{type(error).__name__} {message}".upper()
    if "API_KEY" in upper or "API KEY" in upper or "PERMISSION_DENIED" in upper or "UNAUTHENTICATED" in upper:
        return "Gemini API key is missing or invalid. Set GEMINI_API_KEY in .env and restart."
    if "402" in message or "CREDITS" in upper or "BILLING" in upper:
        return "Your Gemini API project has no credits left. Check billing at https://ai.studio/projects."
    if "429" in message or "RESOURCE_EXHAUSTED" in upper:
        return "Gemini rate limit reached. Wait a minute and try again."
    if "TIMEOUT" in upper or "DEADLINE_EXCEEDED" in upper:
        return "Gemini took too long to respond (high server demand). Click the button to try again."
    if isinstance(error, ValueError):
        return message
    return f"Gemini call failed: {message[:300]}"
