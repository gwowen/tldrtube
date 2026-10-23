"""Transcript retrieval, normalisation, and chunking."""

import json
from pathlib import Path

import httpx
import pytest
from youtube_transcript_api import (
    FetchedTranscript,
    FetchedTranscriptSnippet,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from tldrtube.transcript import (
    CHUNK_CHAR_LIMIT,
    CHUNK_HARD_CHAR_LIMIT,
    SINGLE_PASS_CHAR_LIMIT,
    TranscriptError,
    build_transcript,
    chunk_transcript,
    fetch_transcript,
    fetch_video_title,
    format_timestamp,
    parse_video_id,
    watch_url,
)

FIXTURES = Path(__file__).parent / "fixtures"
VIDEO_ID = "dQw4w9WgXcQ"


def load_fixture(name: str):
    payload = json.loads((FIXTURES / name).read_text())
    snippets = [
        (item["start"], item["duration"], item["text"]) for item in payload["segments"]
    ]
    transcript = build_transcript(
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=False,
        snippets=snippets,
    )
    return payload, transcript


class _Caption:
    def __init__(self, text: str, *, generated: bool, language_code: str = "en") -> None:
        self.text = text
        self.is_generated = generated
        self.language_code = language_code
        self.language = "English"
        self.fetched = False

    def fetch(self) -> FetchedTranscript:
        self.fetched = True
        return FetchedTranscript(
            snippets=[FetchedTranscriptSnippet(text=self.text, start=1.25, duration=2.0)],
            video_id=VIDEO_ID,
            language=self.language,
            language_code=self.language_code,
            is_generated=self.is_generated,
        )


class _Listing:
    def __init__(self, captions: list[_Caption]) -> None:
        self.captions = captions
        self.video_id = VIDEO_ID

    def __iter__(self):
        return iter(self.captions)

    def __str__(self) -> str:
        return "listing"

    def find_manually_created_transcript(self, codes: list[str]) -> _Caption:
        for caption in self.captions:
            if not caption.is_generated and caption.language_code in codes:
                return caption
        raise NoTranscriptFound(self.video_id, codes, self)

    def find_generated_transcript(self, codes: list[str]) -> _Caption:
        for caption in self.captions:
            if caption.is_generated and caption.language_code in codes:
                return caption
        raise NoTranscriptFound(self.video_id, codes, self)


class _Api:
    def __init__(self, listing: _Listing | None = None, error: Exception | None = None) -> None:
        self.listing = listing
        self.error = error

    def list(self, video_id: str):
        assert video_id == VIDEO_ID
        if self.error is not None:
            raise self.error
        return self.listing


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (VIDEO_ID, VIDEO_ID),
        (f"https://www.youtube.com/watch?v={VIDEO_ID}", VIDEO_ID),
        (f"https://www.youtube.com/watch?v={VIDEO_ID}&t=30s", VIDEO_ID),
        (f"https://youtu.be/{VIDEO_ID}", VIDEO_ID),
        (f"https://www.youtube.com/shorts/{VIDEO_ID}", VIDEO_ID),
        (f"https://www.youtube.com/embed/{VIDEO_ID}", VIDEO_ID),
        (f"https://www.youtube.com/live/{VIDEO_ID}", VIDEO_ID),
        (f"https://www.youtube-nocookie.com/embed/{VIDEO_ID}?rel=0", VIDEO_ID),
    ],
)
def test_parse_video_id(source: str, expected: str) -> None:
    assert parse_video_id(source) == expected


@pytest.mark.parametrize(
    "source",
        ["nope", "https://example.com/watch?v=dQw4w9WgXcQ", "https://youtu.be/short"],
)
def test_parse_video_id_rejects_invalid_sources(source: str) -> None:
    with pytest.raises(TranscriptError):
        parse_video_id(source)


def test_prefers_manual_captions() -> None:
    manual = _Caption("Human caption about bridges.", generated=False)
    generated = _Caption("Automatic caption.", generated=True)
    transcript = fetch_transcript(
        VIDEO_ID,
        api=_Api(_Listing([generated, manual])),
    )
    assert transcript.is_generated is False
    assert transcript.segments[0].text == "Human caption about bridges."
    assert manual.fetched is True
    assert generated.fetched is False


def test_falls_back_to_generated_captions() -> None:
    generated = _Caption("Automatic caption about bridges.", generated=True)
    transcript = fetch_transcript(VIDEO_ID, api=_Api(_Listing([generated])))
    assert transcript.is_generated is True
    assert transcript.segments[0].text == "Automatic caption about bridges."


def test_matches_regional_language_code() -> None:
    manual = _Caption("British English caption.", generated=False, language_code="en-GB")
    transcript = fetch_transcript(VIDEO_ID, language="en", api=_Api(_Listing([manual])))
    assert transcript.language_code == "en-GB"
    assert manual.fetched is True


def test_missing_language_fails_without_inventing_a_summary() -> None:
    german = _Caption("Hallo.", generated=False, language_code="de")
    with pytest.raises(TranscriptError, match="will not reconstruct"):
        fetch_transcript(VIDEO_ID, language="en", api=_Api(_Listing([german])))


def test_disabled_and_unavailable_videos_fail_clearly() -> None:
    with pytest.raises(TranscriptError, match="Captions are disabled"):
        fetch_transcript(VIDEO_ID, api=_Api(error=TranscriptsDisabled(VIDEO_ID)))
    with pytest.raises(TranscriptError, match="unavailable"):
        fetch_transcript(VIDEO_ID, api=_Api(error=VideoUnavailable(VIDEO_ID)))


def test_empty_transcript_fails() -> None:
    caption = _Caption("   <c></c>  ", generated=False)
    with pytest.raises(TranscriptError, match="usable text"):
        fetch_transcript(VIDEO_ID, api=_Api(_Listing([caption])))


def test_normalisation_keeps_order_and_flags_non_speech() -> None:
    transcript = build_transcript(
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=False,
        snippets=[
            (0.0, 1.0, "  Hello   <c>world</c> "),
            (2.0, 1.0, "[Music]"),
            (3.0, 1.0, "Hello [Music]"),
            (4.0, 1.0, "   "),
        ],
    )
    assert [segment.text for segment in transcript.segments] == [
        "Hello world",
        "[Music]",
        "Hello [Music]",
    ]
    assert [segment.id for segment in transcript.segments] == ["s1", "s2", "s3"]
    assert [segment.non_speech for segment in transcript.segments] == [False, True, False]
    assert transcript.segments[0].start == 0.0


def test_fixtures_preserve_sponsor_conflict_and_injection() -> None:
    _, explanation = load_fixture("short_explanation.json")
    assert explanation.segments[-1].non_speech is True
    assert "oxygen" in explanation.segments[1].text

    _, sponsor = load_fixture("sponsor.json")
    assert [segment.text for segment in sponsor.segments].count(
        "This video is sponsored by Acme Widgets."
    ) == 2

    _, contradictory = load_fixture("contradictory.json")
    assert "cut costs" in contradictory.segments[0].text
    assert "raise costs" in contradictory.segments[1].text

    notes, injection = load_fixture("injection.json")
    assert "invented statistic" in injection.segments[0].text
    assert "hash table" in injection.segments[1].text
    assert "Adversarial" in notes["notes"]


def _transcript_from(snippets: list[tuple[float, float, str]]):
    return build_transcript(
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=True,
        snippets=snippets,
    )


def _assert_partition(chunks, transcript) -> None:
    flattened = [segment.id for chunk in chunks for segment in chunk.segments]
    assert flattened == [segment.id for segment in transcript.segments]
    assert chunks[0].context == ()


def test_short_transcript_stays_in_one_chunk() -> None:
    _, transcript = load_fixture("short_explanation.json")
    chunks = chunk_transcript(transcript)
    assert len(chunks) == 1
    assert chunks[0].segments == transcript.segments
    assert chunks[0].context == ()


def test_long_transcript_is_split_on_segment_boundaries() -> None:
    transcript = _transcript_from([(float(index), 1.0, "a" * 1000) for index in range(13)])
    assert sum(len(segment.text) for segment in transcript.segments) > SINGLE_PASS_CHAR_LIMIT
    chunks = chunk_transcript(transcript)
    assert [chunk.id for chunk in chunks] == ["c1", "c2"]
    assert [len(chunk.segments) for chunk in chunks] == [6, 7]
    assert sum(len(segment.text) for segment in chunks[0].segments) <= CHUNK_CHAR_LIMIT
    second = sum(len(segment.text) for segment in chunks[1].segments)
    assert CHUNK_CHAR_LIMIT < second <= CHUNK_HARD_CHAR_LIMIT
    _assert_partition(chunks, transcript)


def test_sentence_that_crosses_the_budget_stays_together() -> None:
    ending = "because the cable snapped."
    snippets: list[tuple[float, float, str]] = [(float(index), 0.4, "a" * 1000) for index in range(6)]
    snippets.append((6.0, 0.4, ending))
    for index in range(7):
        snippets.append((7.0 + index, 0.4, "b" * 1000))
    transcript = _transcript_from(snippets)
    chunks = chunk_transcript(transcript)
    assert ending in chunks[0].segments[-1].text
    assert sum(len(segment.text) for segment in chunks[0].segments) > CHUNK_CHAR_LIMIT
    assert all(ending not in segment.text for chunk in chunks[1:] for segment in chunk.segments)
    _assert_partition(chunks, transcript)


def test_pause_starts_a_new_chunk() -> None:
    snippets: list[tuple[float, float, str]] = [(float(index), 1.0, "a" * 1000) for index in range(3)]
    # The third caption ends at 3.0s, so the next one opens on a 3s pause.
    for index in range(10):
        snippets.append((6.0 + index, 1.0, "b" * 1000))
    transcript = _transcript_from(snippets)
    chunks = chunk_transcript(transcript)
    assert [segment.text[0] for segment in chunks[0].segments] == ["a", "a", "a"]
    assert chunks[1].segments[0].text.startswith("b")
    _assert_partition(chunks, transcript)


def test_non_speech_cue_ends_a_chunk() -> None:
    snippets: list[tuple[float, float, str]] = [(float(index), 0.5, "a" * 1000) for index in range(3)]
    snippets.append((3.0, 0.5, "[Music]"))
    for index in range(10):
        snippets.append((3.5 + index, 0.5, "b" * 1000))
    transcript = _transcript_from(snippets)
    chunks = chunk_transcript(transcript)
    assert chunks[0].segments[-1].non_speech is True
    assert chunks[1].segments[0].non_speech is False
    assert chunks[1].segments[0].text.startswith("b")
    _assert_partition(chunks, transcript)


def test_lookback_is_only_on_later_chunks() -> None:
    snippets: list[tuple[float, float, str]] = []
    moment = 0.0
    for _ in range(5):
        snippets.append((moment, 0.4, "a" * 999 + "."))
        moment += 0.4
    snippets.append((moment, 0.4, "The cable snapped."))
    moment += 0.4
    for _ in range(7):
        snippets.append((moment, 0.4, "b" * 1000))
        moment += 0.4
    transcript = _transcript_from(snippets)
    chunks = chunk_transcript(transcript)
    previous_ids = {segment.id for segment in chunks[0].segments}
    assert chunks[1].context
    assert [segment.text for segment in chunks[1].context] == ["The cable snapped."]
    assert all(segment.id in previous_ids for segment in chunks[1].context)
    _assert_partition(chunks, transcript)


def test_abbreviation_does_not_split_a_sentence() -> None:
    snippets: list[tuple[float, float, str]] = [(float(index), 0.4, "a" * 1000) for index in range(6)]
    snippets.append((6.0, 0.4, "Ask Dr."))
    snippets.append((7.0, 0.4, "Smith finished the explanation."))
    for index in range(7):
        snippets.append((8.0 + index, 0.4, "b" * 1000))
    transcript = _transcript_from(snippets)
    chunks = chunk_transcript(transcript)
    together = next(
        chunk
        for chunk in chunks
        if any(segment.text == "Ask Dr." for segment in chunk.segments)
    )
    assert any(segment.text == "Smith finished the explanation." for segment in together.segments)
    _assert_partition(chunks, transcript)


def test_caption_longer_than_the_budget_stays_whole() -> None:
    huge = "a" * 9_000
    transcript = _transcript_from([(0.0, 0.5, huge), (0.5, 0.5, "b" * 4_000)])
    chunks = chunk_transcript(transcript)
    assert chunks[0].segments[0].text == huge
    assert len(chunks[0].segments) == 1
    _assert_partition(chunks, transcript)


@pytest.mark.parametrize(
    ("start", "label"),
    [(0, "00:00"), (14, "00:14"), (134, "02:14"), (3600, "1:00:00"), (3725, "1:02:05")],
)
def test_format_timestamp(start: int, label: str) -> None:
    assert format_timestamp(start) == label


class _TitleResponse:
    def __init__(
        self,
        *,
        payload: object | None = None,
        text: str = "",
        error: Exception | None = None,
    ) -> None:
        self._payload = payload
        self.text = text
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> object:
        return self._payload


class _TitleClient:
    def __init__(self, responses: list[_TitleResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get(self, url: str, *, params: dict[str, str] | None = None) -> _TitleResponse:
        self.calls.append((url, params))
        return self._responses.pop(0)


def test_watch_url_uses_the_video_id() -> None:
    assert watch_url(VIDEO_ID) == f"https://www.youtube.com/watch?v={VIDEO_ID}"


def test_fetch_video_title_uses_oembed() -> None:
    client = _TitleClient([_TitleResponse(payload={"title": "  Bioware Closes  "})])
    assert fetch_video_title(VIDEO_ID, client=client) == "Bioware Closes"
    assert client.calls == [
        (
            "https://www.youtube.com/oembed",
            {"url": f"https://www.youtube.com/watch?v={VIDEO_ID}", "format": "json"},
        )
    ]


def test_fetch_video_title_falls_back_to_og_title() -> None:
    pages = [
        '<meta content="Bioware Closes &amp; More" property="og:title">',
        '<meta property="og:title" content="Bioware Closes &amp; More">',
    ]
    for html in pages:
        client = _TitleClient(
            [
                _TitleResponse(error=httpx.HTTPError("unauthorized")),
                _TitleResponse(text=html),
            ]
        )
        assert fetch_video_title(VIDEO_ID, client=client) == "Bioware Closes & More"
        assert client.calls[1] == (f"https://www.youtube.com/watch?v={VIDEO_ID}", None)


def test_fetch_video_title_fails_when_both_sources_fail() -> None:
    client = _TitleClient(
        [
            _TitleResponse(error=httpx.HTTPError("unauthorized")),
            _TitleResponse(error=httpx.HTTPError("unavailable")),
        ]
    )
    with pytest.raises(TranscriptError, match="Could not retrieve the video title"):
        fetch_video_title(VIDEO_ID, client=client)
