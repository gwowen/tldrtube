"""Fetch, normalise, and chunk YouTube transcripts.

This module does the work that does not need language understanding. It keeps
caption order, text, and timestamps, and it never asks the model to invent a
summary from a title or description.
"""

import logging
import re
from html import unescape
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    FetchedTranscriptSnippet,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from tldrtube.models import Chunk, Segment, Transcript

logger = logging.getLogger("tldrtube")

# A 2.6B model stays more faithful on short spans than on one huge prompt.
# Budgets count caption characters, and a segment is never split in half.
# Cuts prefer the end of a sentence, a pause, or a non-speech cue.
SINGLE_PASS_CHAR_LIMIT = 12_000
CHUNK_CHAR_LIMIT = 6_000
# A sentence may run past the target so the thought stays in one extraction.
CHUNK_HARD_CHAR_LIMIT = 8_000
# A pause or cue opens a new chunk only after this much speech.
MIN_CHUNK_CHARS = 2_000
# A shorter final chunk is merged back when the hard cap still holds.
MIN_TAIL_CHARS = 1_500
PAUSE_SPLIT_SECONDS = 3.0
# The previous thought copied into the next chunk. A longer caption stays behind.
LOOKBACK_CHAR_LIMIT = 800
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "jr",
        "sr",
        "st",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "u.s",
        "u.k",
    }
)
_CLOSING_QUOTE = "\"'”’"

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_NON_SPEECH = re.compile(
    r"^[\s\W_]*("
    r"music|musical interlude|applause|laughter|laughs|laughing|laugh|"
    r"silence|inaudible|coughing|cough|cheering|cheers|"
    r"sighs|sigh|gasps|gasp|background noise|noise"
    r")(?:\s+(?:playing|continues|sounds?))?[\s\W_]*$",
    re.IGNORECASE,
)
_YOUTUBE_HOSTS = {
    "youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
}


class TranscriptError(Exception):
    """A transcript could not be retrieved or contained no usable text."""


def parse_video_id(source: str) -> str:
    """Return the 11-character video id from a URL or a bare id."""

    text = source.strip()
    if _VIDEO_ID.fullmatch(text):
        return text

    parsed = urlparse(text)
    host = parsed.netloc.lower().removeprefix("www.")
    video_id = ""
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif host in _YOUTUBE_HOSTS:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
                video_id = parts[1]
    if not _VIDEO_ID.fullmatch(video_id):
        raise TranscriptError(f"Could not find a YouTube video id in {source!r}.")
    return video_id


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def normalize_caption(text: str) -> str:
    """Drop caption markup and redundant whitespace without changing the words."""

    cleaned = _TAG.sub("", unescape(text))
    return collapse_whitespace(cleaned)


def is_non_speech(text: str) -> bool:
    """True only when the whole caption is a non-speech cue.

    Introductions, sponsorships, and asides stay in the transcript. The model
    decides whether they belong in the summary.
    """

    return _NON_SPEECH.fullmatch(text) is not None


def format_timestamp(start: float) -> str:
    """Format a segment start as MM:SS, or H:MM:SS past one hour."""

    total = max(0, int(start))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_segment_line(segment: Segment) -> str:
    """One prompt line. The id in brackets is what the model may cite.

    The clock time is left out. Python computes every timestamp the user sees
    from the cited segments, and a model copies a clock time as an id.
    """

    marker = "[non-speech] " if segment.non_speech else ""
    return f"[{segment.id}] {marker}{segment.text}"


def utterance_continues(segment: Segment, following: Segment | None) -> bool:
    """True when the next caption is still the same spoken thought."""

    return following is not None and not _utterance_boundary(segment, following)


def build_transcript(
    *,
    video_id: str,
    language: str,
    language_code: str,
    is_generated: bool,
    snippets: list[tuple[float, float, str]] | list[FetchedTranscriptSnippet],
) -> Transcript:
    """Assign stable s1, s2, ... ids after empty captions have been removed."""

    segments: list[Segment] = []
    for snippet in snippets:
        if isinstance(snippet, FetchedTranscriptSnippet):
            start, duration, raw = snippet.start, snippet.duration, snippet.text
        else:
            start, duration, raw = snippet
        text = normalize_caption(raw)
        if not text:
            continue
        segments.append(
            Segment(
                id=f"s{len(segments) + 1}",
                start=start,
                duration=duration,
                text=text,
                non_speech=is_non_speech(text),
            )
        )
    if not segments:
        raise TranscriptError("The transcript did not contain any usable text.")
    return Transcript(
        video_id=video_id,
        language=language,
        language_code=language_code,
        is_generated=is_generated,
        segments=tuple(segments),
    )


def caption_chars(transcript: Transcript) -> int:
    """Caption characters, including non-speech cues. This is the one-pass budget."""

    return sum(len(segment.text) for segment in transcript.segments)


def chunk_transcript(transcript: Transcript) -> list[Chunk]:
    """Split a long transcript on thought boundaries without reordering it.

    A transcript that fits in one pass stays one chunk. A longer one is packed
    from utterances, and each later chunk repeats a short lookback in context.
    The new spans still cover every segment once, in order.
    """

    segments = list(transcript.segments)
    if caption_chars(transcript) <= SINGLE_PASS_CHAR_LIMIT:
        return [Chunk(id="c1", segments=tuple(segments))]

    packed = _pack_utterances(_group_utterances(segments))
    chunks: list[Chunk] = []
    for index, utterances in enumerate(packed):
        previous = packed[index - 1] if index else None
        chunks.append(
            Chunk(
                id=f"c{index + 1}",
                segments=tuple(
                    segment for utterance in utterances for segment in utterance
                ),
                context=_lookback(previous) if previous else (),
            )
        )
    return chunks


def utterance_tail(transcript: Transcript, segment_id: str) -> tuple[Segment, ...]:
    """Later captions that finish this segment's utterance.

    Empty when the segment already ends that utterance.
    """

    for utterance in _group_utterances(list(transcript.segments)):
        ids = [segment.id for segment in utterance]
        if segment_id not in ids:
            continue
        return tuple(utterance[ids.index(segment_id) + 1:])
    return ()


def _group_utterances(segments: list[Segment]) -> list[list[Segment]]:
    """Group captions that belong to one spoken thought.

    An utterance ends on a sentence, a pause, or just before a non-speech cue.
    A cue is its own utterance so it can sit on the edge of a chunk.
    """

    utterances: list[list[Segment]] = []
    current: list[Segment] = []
    for index, segment in enumerate(segments):
        if segment.non_speech:
            if current:
                utterances.append(current)
                current = []
            utterances.append([segment])
            continue
        current.append(segment)
        following = segments[index + 1] if index + 1 < len(segments) else None
        if _utterance_boundary(segment, following):
            utterances.append(current)
            current = []
    if current:
        utterances.append(current)
    return utterances


def _utterance_boundary(segment: Segment, following: Segment | None) -> bool:
    if following is None or following.non_speech:
        return True
    gap = following.start - (segment.start + segment.duration)
    if gap >= PAUSE_SPLIT_SECONDS:
        return True
    return _ends_sentence(segment.text)


def _ends_sentence(text: str) -> bool:
    """True when the caption ends a sentence, ignoring titles and initials."""

    stripped = text.rstrip()
    if stripped and stripped[-1] in _CLOSING_QUOTE:
        stripped = stripped[:-1].rstrip()
    if not stripped or stripped[-1] not in ".!?":
        return False
    if stripped[-1] in "!?":
        return True
    core = stripped.split()[-1].rstrip(".").lower()
    if not core:
        return True
    if core in _ABBREVIATIONS or re.fullmatch(r"[a-z]", core):
        return False
    return True


def _pack_utterances(utterances: list[list[Segment]]) -> list[list[list[Segment]]]:
    """Pack utterances up to the target size, keeping each thought intact."""

    chunks: list[list[list[Segment]]] = []
    current: list[list[Segment]] = []
    current_length = 0

    def close() -> None:
        nonlocal current, current_length
        if not current:
            return
        chunks.append(current)
        current = []
        current_length = 0

    def append(utterance: list[Segment]) -> None:
        nonlocal current_length
        current.append(utterance)
        current_length += sum(len(segment.text) for segment in utterance)

    for utterance in utterances:
        for piece in _split_long_utterance(utterance):
            piece_length = sum(len(segment.text) for segment in piece)
            non_speech = all(segment.non_speech for segment in piece)
            if not current:
                append(piece)
                continue
            # A cue stays with the speech before it. The next speech opens a chunk.
            if non_speech and _speech_length(current) >= MIN_CHUNK_CHARS:
                append(piece)
                continue
            if (
                _trailing_non_speech(current)
                and not non_speech
                and _speech_length(current) >= MIN_CHUNK_CHARS
            ):
                close()
                append(piece)
                continue
            if (
                not non_speech
                and _gap(current, piece) >= PAUSE_SPLIT_SECONDS
                and _speech_length(current) >= MIN_CHUNK_CHARS
            ):
                close()
                append(piece)
                continue
            if current_length + piece_length > CHUNK_CHAR_LIMIT:
                # Finish a sentence that has already started, up to the hard cap.
                if (
                    not _chunk_ends_sentence(current)
                    and _ends_sentence(piece[-1].text)
                    and current_length + piece_length <= CHUNK_HARD_CHAR_LIMIT
                ):
                    append(piece)
                else:
                    close()
                    append(piece)
                continue
            append(piece)
    close()
    _merge_short_tail(chunks)
    return chunks


def _split_long_utterance(utterance: list[Segment]) -> list[list[Segment]]:
    """Break an utterance past the hard cap. A single caption stays whole."""

    if sum(len(segment.text) for segment in utterance) <= CHUNK_HARD_CHAR_LIMIT:
        return [utterance]
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    current_length = 0
    for segment in utterance:
        size = len(segment.text)
        if current and current_length + size > CHUNK_CHAR_LIMIT:
            groups.append(current)
            current = []
            current_length = 0
        current.append(segment)
        current_length += size
    if current:
        groups.append(current)
    return groups


def _merge_short_tail(chunks: list[list[list[Segment]]]) -> None:
    if len(chunks) < 2:
        return
    tail_length = _utterances_length(chunks[-1])
    if tail_length >= MIN_TAIL_CHARS:
        return
    if _utterances_length(chunks[-2]) + tail_length <= CHUNK_HARD_CHAR_LIMIT:
        chunks[-2].extend(chunks[-1])
        chunks.pop()


def _lookback(utterances: list[list[Segment]]) -> tuple[Segment, ...]:
    """The end of the previous thought, when it fits in the lookback budget."""

    for utterance in reversed(utterances):
        if all(segment.non_speech for segment in utterance):
            continue
        chosen: list[Segment] = []
        total = 0
        for segment in reversed(utterance):
            size = len(segment.text)
            if size > LOOKBACK_CHAR_LIMIT or total + size > LOOKBACK_CHAR_LIMIT:
                break
            chosen.append(segment)
            total += size
        chosen.reverse()
        return tuple(chosen)
    return ()


def _gap(current: list[list[Segment]], piece: list[Segment]) -> float:
    previous = current[-1][-1]
    return piece[0].start - (previous.start + previous.duration)


def _speech_length(utterances: list[list[Segment]]) -> int:
    return sum(
        len(segment.text)
        for utterance in utterances
        for segment in utterance
        if not segment.non_speech
    )


def _utterances_length(utterances: list[list[Segment]]) -> int:
    return sum(len(segment.text) for utterance in utterances for segment in utterance)


def _trailing_non_speech(utterances: list[list[Segment]]) -> bool:
    return bool(utterances) and all(segment.non_speech for segment in utterances[-1])


def _chunk_ends_sentence(utterances: list[list[Segment]]) -> bool:
    for utterance in reversed(utterances):
        for segment in reversed(utterance):
            if not segment.non_speech:
                return _ends_sentence(segment.text)
    return False


def fetch_transcript(
    source: str,
    *,
    language: str = "en",
    api: YouTubeTranscriptApi | None = None,
) -> Transcript:
    """Retrieve the preferred caption track and preserve its order."""

    video_id = parse_video_id(source)
    logger.info("Fetching the transcript for %s.", video_id)
    client = api or YouTubeTranscriptApi()
    try:
        listing = client.list(video_id)
    except CouldNotRetrieveTranscript as exc:
        raise TranscriptError(_retrieval_message(exc, language)) from exc

    # Human captions first. Generated captions are the fallback, not a blend.
    manuals = [item for item in listing if not item.is_generated]
    generated = [item for item in listing if item.is_generated]
    try:
        chosen = listing.find_manually_created_transcript(
            _language_codes(language, [item.language_code for item in manuals])
        )
    except NoTranscriptFound:
        try:
            chosen = listing.find_generated_transcript(
                _language_codes(language, [item.language_code for item in generated])
            )
        except NoTranscriptFound as exc:
            raise TranscriptError(
                f"No {language} transcript is available for this video. "
                "tldrtube only summarises an existing transcript and will not "
                "reconstruct the video from its title or description."
            ) from exc

    try:
        fetched = chosen.fetch()
    except CouldNotRetrieveTranscript as exc:
        raise TranscriptError(_retrieval_message(exc, language)) from exc

    transcript = build_transcript(
        video_id=fetched.video_id,
        language=fetched.language,
        language_code=fetched.language_code,
        is_generated=fetched.is_generated,
        snippets=list(fetched.snippets),
    )
    kind = "generated" if transcript.is_generated else "human-written"
    segment_count = len(transcript.segments)
    segment_word = "segment" if segment_count == 1 else "segments"
    logger.info(
        "Using %s %s captions (%s %s).",
        kind,
        transcript.language,
        segment_count,
        segment_word,
    )
    return transcript


def _language_codes(requested: str, available: list[str]) -> list[str]:
    """Prefer an exact language code, then a regional variant such as en-GB for en."""

    requested_lower = requested.lower()
    base = requested_lower.split("-", 1)[0]
    ranked: list[str] = []
    for code in available:
        if code.lower() == requested_lower and code not in ranked:
            ranked.append(code)
    for code in available:
        lower = code.lower()
        if code not in ranked and (lower == base or lower.startswith(base + "-")):
            ranked.append(code)
    return ranked or [requested]


def _retrieval_message(exc: CouldNotRetrieveTranscript, language: str) -> str:
    """Turn the caption library's long errors into one actionable sentence."""

    if isinstance(exc, VideoUnavailable):
        return "This video is unavailable."
    if isinstance(exc, TranscriptsDisabled):
        return "Captions are disabled for this video."
    if isinstance(exc, AgeRestricted):
        return "This video is age-restricted and its transcript cannot be retrieved."
    if isinstance(exc, (RequestBlocked, IpBlocked)):
        return "YouTube blocked the transcript request."
    if isinstance(exc, NoTranscriptFound):
        return (
            f"No {language} transcript is available for this video. "
            "tldrtube only summarises an existing transcript and will not "
            "reconstruct the video from its title or description."
        )
    cause = getattr(exc, "cause", "") or str(exc)
    first = cause.strip().split("\n\n", 1)[0].strip()
    return first or "Could not retrieve a transcript for this video."
