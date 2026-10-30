"""Pydantic schemas: what each agent returns (structured JSON output) and what the API accepts."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ============================================================
# Summary Agent outputs
# ============================================================


class JobPreferences(BaseModel):
    """The Summary Agent's understanding of what the user is looking for."""
    target_roles: list[str] = Field(description="Job titles / role types to search for")
    levels: list[str] = Field(description="Seniority levels, e.g. Internship, Entry-level, New Grad, Mid-level")
    locations: list[str] = Field(description="Cities / regions, or 'Remote'")
    work_mode: str = Field(description="onsite, hybrid, remote, or any")
    must_haves: list[str] = Field(description="Hard requirements the user stated (visa, salary, industry, ...)")
    deal_breakers: list[str] = Field(description="Things the user does not want")
    resume_highlights: list[str] = Field(description="Strongest resume skills/experiences to match jobs against")
    search_keywords: list[str] = Field(description="Keywords the Web Search Agent should use")


class UnderstandingReply(BaseModel):
    reply: str = Field(description="Message to the user: the understanding in plain words + a confirm question")
    understanding: JobPreferences
    ready_to_confirm: bool = Field(description="True when role, level and location are all known")


class JobPick(BaseModel):
    index: int = Field(description="1-based number of the job post in the list you were given")
    summary: str = Field(description="One or two sentences on what the job is")
    why_relevant: str = Field(description="Why it fits this user, citing their resume")


class JobPresentation(BaseModel):
    message: str = Field(description="Short intro message to the user about these results")
    jobs: list[JobPick]


class TopicItem(BaseModel):
    name: str = Field(description="Short topic name, e.g. 'SQL joins and indexing'")
    why_needed: str = Field(description="Which part of the job description requires it")
    resume_evidence: str = Field(description="What in the resume shows experience with it, or 'None found'")


class TopicList(BaseModel):
    message: str = Field(description="Message to the user introducing the topics and asking for self-ratings")
    topics: list[TopicItem]


class PlanItem(BaseModel):
    topic_number: int = Field(description="1-based number of the topic in the list you were given")
    priority: Literal["study_first", "review_later"]
    plan: str = Field(description="Concrete study actions for this topic")


class StudyPlan(BaseModel):
    message: str = Field(description="Message to the user explaining the plan and the order")
    items: list[PlanItem]


class PlannedQuestion(BaseModel):
    question: str = Field(description="The interview question exactly as the interviewer should ask it")
    source: Literal["reported", "jd_concept", "resume_deep_dive"]
    source_url: str = Field(description="URL of the report for 'reported' questions, otherwise empty")
    what_it_tests: str = Field(description="What a strong answer must demonstrate")


class InterviewPlan(BaseModel):
    message: str = Field(description="Message to the user describing the interview they are about to take")
    questions: list[PlannedQuestion]


class ReviewPresentation(BaseModel):
    message: str = Field(description="Message to the user summarizing how the interview went")
    prepare_next: list[str] = Field(description="Concrete things to prepare next, most important first")


# ============================================================
# Web Search Agent outputs
# ============================================================


class FoundJob(BaseModel):
    title: str
    company: str
    location: str
    level: str
    url: str = Field(description="Link to the posting, copied from the sources list")
    source: Literal["linkedin", "handshake", "other"]
    posted_date: str = Field(description="YYYY-MM-DD, or 'unknown'")
    job_description: str = Field(
        description="Responsibilities and requirements summarized in your own words (6-10 sentences)"
    )
    key_requirements: list[str]


class FoundJobs(BaseModel):
    jobs: list[FoundJob]
    search_notes: str = Field(description="Anything the user should know about the search")


class JobScore(BaseModel):
    index: int = Field(description="1-based number of the candidate post in the list you were given")
    fit_score: int = Field(description="0-100: how well the post fits the request and the resume strengths")
    reason: str = Field(description="One short sentence explaining the score")


class JobRanking(BaseModel):
    scores: list[JobScore]


class ReportedQuestion(BaseModel):
    question: str
    source_url: str
    source_title: str
    context: str = Field(description="Interview round / type / date if known")


class InterviewResearch(BaseModel):
    questions: list[ReportedQuestion]
    process_notes: str = Field(description="What candidates report about the interview process")
    search_notes: str


# ============================================================
# Review Agent outputs
# ============================================================


class RubricScores(BaseModel):
    accuracy: int = Field(description="1-10: is the content correct?")
    depth: int = Field(description="1-10: does it go beyond the surface, trade-offs, why?")
    structure: int = Field(description="1-10: clear structure (STAR for behavioral questions)?")
    communication: int = Field(description="1-10: concise, confident, easy to follow?")

    def overall(self) -> int:
        """Overall score is computed in code, not by the model, so it is consistent."""
        values = [self.accuracy, self.depth, self.structure, self.communication]
        return max(1, min(10, round(sum(values) / len(values))))


class QuestionReview(BaseModel):
    question_number: int
    rubric: RubricScores
    strengths: str
    improvements: str
    senior_lens: str = Field(description="What a senior interviewer listens for on this question")


class InterviewReview(BaseModel):
    reviews: list[QuestionReview]
    overall_summary: str
    top_strengths: list[str]
    focus_areas: list[str]


# ============================================================
# API request bodies
# ============================================================


class MessageIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class SaveJobsIn(BaseModel):
    result_ids: list[int] = Field(..., min_length=1)


class ToggleSaveIn(BaseModel):
    force_save: bool = False


class JobRatingIn(BaseModel):
    rating: Optional[int] = Field(default=None, ge=1, le=5)


class JobDescriptionIn(BaseModel):
    job_description: str = Field(..., min_length=50, max_length=30000)


class RatingsIn(BaseModel):
    ratings: dict[int, int] = Field(..., description="topic_id -> self rating 1..5")

    @field_validator("ratings")
    @classmethod
    def check_range(cls, ratings: dict[int, int]) -> dict[int, int]:
        if not ratings:
            raise ValueError("Rate at least one topic.")
        for value in ratings.values():
            if not 1 <= value <= 5:
                raise ValueError("Ratings must be between 1 and 5.")
        return ratings


class TopicStatusIn(BaseModel):
    status: Literal["not_started", "studying", "done"]
