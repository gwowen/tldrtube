"""Schemas for transcripts, model responses, and length presets.

Field descriptions are part of the JSON schema sent to Ollama. They tell the
model what each value means, so they are instructions as well as documentation.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL = "lfm3:latest"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
# Token budget for a chunk extraction. The final summary uses the preset budget.
EXTRACT_NUM_PREDICT = 500
# A claim cites the few lines that support it. A long id list spends the token
# budget before the rest of the summary is written.
MAX_EVIDENCE_IDS = 8


class Segment(BaseModel):
    # Frozen so later stages cannot rewrite the evidence they cite.
    model_config = ConfigDict(frozen=True)

    id: str
    start: float
    duration: float
    text: str
    # True when the whole caption is a cue such as "[Music]", not speech.
    non_speech: bool = False


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    language: str
    language_code: str
    is_generated: bool
    segments: tuple[Segment, ...]


class Chunk(BaseModel):
    """A consecutive span of segments. Chunking never reorders the transcript.

    context repeats the end of the previous span so a thought can finish.
    It is not part of this span's own coverage.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    segments: tuple[Segment, ...]
    context: tuple[Segment, ...] = ()


# The pattern goes into the JSON schema sent to Ollama so grammar-constrained
# decoding can only write ids such as s633. It is not a validator: the verifier
# still repairs a stray form such as 633 instead of rejecting the response.
_SegmentId = Annotated[str, Field(json_schema_extra={"pattern": "^s[0-9]+$"})]


class Claim(BaseModel):
    text: str = Field(description="A concise statement supported only by the transcript.")
    evidence_ids: list[_SegmentId] = Field(
        min_length=1,
        max_length=MAX_EVIDENCE_IDS,
        description=(
            f"From 1 to {MAX_EVIDENCE_IDS} segment ids. "
            "Copy only the id in brackets, such as s127. "
            "An id is the letter s followed by a number. Never drop the s. "
            "One id per line. A span is not an id."
        ),
    )


class SummaryResponse(BaseModel):
    """The only shape the final model call is allowed to return."""

    main_idea: Claim = Field(description="The video's central message.")
    key_points: list[Claim] = Field(
        description="Supporting points in spoken order."
    )
    conclusion: Claim | None = Field(
        default=None,
        description="A conclusion or recommendation the speaker actually states. Null if none is stated.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Reasons a complete summary could not be established. Empty when the transcript is sufficient.",
    )


class ExtractResponse(BaseModel):
    """Pass A result. It has no main idea, so a chunk cannot invent the video's thesis."""

    points: list[Claim] = Field(
        default_factory=list,
        description="Complete ideas that are new in this span, in spoken order.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limits of this span. Empty if supported points were found.",
    )


class LengthPreset(BaseModel):
    """Word and point bounds for one summary length.

    num_predict is the generation-token ceiling, which has to be larger than
    the word target because the model also emits the JSON structure.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    min_words: int
    max_words: int
    min_points: int
    max_points: int
    num_predict: int


PRESETS: dict[str, LengthPreset] = {
    "short": LengthPreset(
        name="short",
        min_words=50,
        max_words=80,
        min_points=3,
        max_points=3,
        num_predict=400,
    ),
    "medium": LengthPreset(
        name="medium",
        min_words=100,
        max_words=150,
        min_points=3,
        max_points=5,
        num_predict=700,
    ),
    "detailed": LengthPreset(
        name="detailed",
        min_words=200,
        max_words=300,
        min_points=5,
        max_points=7,
        num_predict=1100,
    ),
}
