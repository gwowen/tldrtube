"""Reject summary claims that are not tied to the transcript.

These checks catch unknown ids, missing citations, mismatched quotations,
unsupported numbers, and unknown names. They do not prove that a paraphrase
says what the cited lines mean. A claim that fails is dropped. When the main
idea fails, the summary is refused.
"""

import re

from tldrtube.models import MAX_EVIDENCE_IDS, Claim, LengthPreset, Segment, SummaryResponse, Transcript
from tldrtube.transcript import collapse_whitespace, utterance_continues

_QUOTE = re.compile(r'["“”]([^"“”]{4,})["“”]')
# Explicit numerals only. Spelled-out numbers such as "forty" are left alone.
_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\b\d+(?:\.\d+)?%?")
# Clock times the model may echo are not factual claims.
_TIMECODE = re.compile(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?")
_TITLE_PHRASE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_WORD = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
# A model often writes a caption span instead of one id per line. The leading
# s is optional because a small model drops it: 633 and s633 name one line.
_SPAN = re.compile(r"^s?\s*(\d+)\s*[-–—]\s*s?\s*(\d+)$", re.IGNORECASE)
_SINGLE = re.compile(r"^s?\s*-?\s*(\d+)$", re.IGNORECASE)
# A model often copies the clock time that follows the segment id.
_CLOCK_SUFFIX = re.compile(r"\s+\[?\d{1,2}:\d{2}(?::\d{2})?\]?$")
# A bare clock time is a pointer to the caption that starts at that second.
_CLOCK = re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})$")
_BRACKETS = "[](){}"


class VerificationError(Exception):
    """A faithful summary could not be established from the transcript."""


def word_count(summary: SummaryResponse) -> int:
    """Count the prose the reader sees, not the JSON keys or evidence ids."""
    parts = [summary.main_idea.text, *(point.text for point in summary.key_points)]
    if summary.conclusion is not None:
        parts.append(summary.conclusion.text)
    return len(" ".join(parts).split())


def point_count(summary: SummaryResponse) -> int:
    """Count key points plus a conclusion, because the conclusion is rendered as a bullet."""
    return len(summary.key_points) + (1 if summary.conclusion is not None else 0)


def review_summary(
    summary: SummaryResponse,
    transcript: Transcript,
    preset: LengthPreset,
) -> list[str]:
    """Return problems that should be sent back for one corrected generation.

    Being under the word or point target is a hint for that retry. A short
    transcript may genuinely have fewer points, so a shortfall does not by
    itself reject the summary.
    """

    segments = _index(transcript)
    problems: list[str] = []
    problems.extend(claim_problems("The main idea", summary.main_idea, segments))
    for index, point in enumerate(summary.key_points, start=1):
        problems.extend(claim_problems(f"Key point {index}", point, segments))
    if summary.conclusion is not None:
        problems.extend(claim_problems("The conclusion", summary.conclusion, segments))

    words = word_count(summary)
    points = point_count(summary)
    if words > preset.max_words:
        problems.append(
            f"The summary is {words} words, above the limit of {preset.max_words}."
        )
    elif words < preset.min_words:
        problems.append(
            f"The summary is {words} words, below the target of {preset.min_words}. "
            "Add supported points if the transcript contains them. "
            "Do not add unsupported material to reach the target."
        )
    if points > preset.max_points:
        problems.append(
            f"The summary has {points} points, above the limit of {preset.max_points}."
        )
    elif points < preset.min_points:
        problems.append(
            f"The summary has {points} points. Include at least {preset.min_points} "
            "when the transcript supports them, and do not invent points to fill the list."
        )
    return problems


def normalize_summary(summary: SummaryResponse, transcript: Transcript) -> SummaryResponse:
    """Store parsed segment ids on every claim the summary will keep."""

    segments = _index(transcript)
    conclusion = summary.conclusion
    return summary.model_copy(
        update={
            "main_idea": normalize_claim(summary.main_idea, segments),
            "key_points": [normalize_claim(point, segments) for point in summary.key_points],
            "conclusion": None if conclusion is None else normalize_claim(conclusion, segments),
        }
    )


def normalize_claim(claim: Claim, segments: dict[str, Segment]) -> Claim:
    """Turn brackets, comma lists, and inclusive spans into real segment ids.

    An entry that matches no transcript line is left unchanged so the retry can
    quote what the model wrote.
    """

    resolved, unresolved = _resolve_evidence_ids(claim.evidence_ids, claim.text, segments)
    if unresolved or not resolved or resolved == [item.strip() for item in claim.evidence_ids]:
        return claim
    return claim.model_copy(update={"evidence_ids": resolved})


def order_key_points(summary: SummaryResponse, transcript: Transcript) -> SummaryResponse:
    """Put key points in the order their evidence was spoken."""

    segments = _index(transcript)
    ordered = sorted(
        enumerate(summary.key_points),
        key=lambda item: _earliest(item[1], segments, item[0]),
    )
    points = [point for _, point in ordered]
    if points == list(summary.key_points):
        return summary
    return summary.model_copy(update={"key_points": points})


def salvage(
    summary: SummaryResponse,
    transcript: Transcript,
    preset: LengthPreset,
) -> SummaryResponse:
    """Drop unsupported claims and extra points. Fail if the main idea cannot be kept.

    Key points are ordered as spoken. When the list is too long, the opening
    and closing points stay and points are removed from the middle.
    """

    summary = normalize_summary(summary, transcript)
    segments = _index(transcript)
    summary = SummaryResponse(
        main_idea=pin_continuations(summary.main_idea, segments),
        key_points=[pin_continuations(point, segments) for point in summary.key_points],
        conclusion=(
            None
            if summary.conclusion is None
            else pin_continuations(summary.conclusion, segments)
        ),
        limitations=summary.limitations,
    )
    limitations = _clean_limitations(summary.limitations)
    cited_extra = False
    main_idea = summary.main_idea
    main_problems = claim_problems("The main idea", main_idea, segments)
    if main_problems:
        repaired = cite_supporting_lines(main_idea, segments)
        if repaired is None:
            raise VerificationError(
                "A faithful summary could not be produced. " + " ".join(main_problems)
            )
        main_idea = repaired
        cited_extra = True

    key_points: list[Claim] = []
    for index, point in enumerate(summary.key_points, start=1):
        problems = claim_problems(f"Key point {index}", point, segments)
        if problems:
            repaired = cite_supporting_lines(point, segments)
            if repaired is None:
                limitations.append(
                    f"Removed key point {index} because it was not supported by the cited transcript."
                )
                continue
            point = repaired
            cited_extra = True
        key_points.append(point)

    conclusion = summary.conclusion
    if conclusion is not None and claim_problems("The conclusion", conclusion, segments):
        repaired = cite_supporting_lines(conclusion, segments)
        if repaired is None:
            limitations.append(
                "Removed the conclusion because it was not supported by the cited transcript."
            )
            conclusion = None
        else:
            conclusion = repaired
            cited_extra = True
    if cited_extra:
        limitations.append(
            "Cited extra transcript lines that contain names or numbers used in the summary."
        )

    ordered = order_key_points(
        SummaryResponse(
            main_idea=main_idea,
            key_points=key_points,
            conclusion=conclusion,
            limitations=limitations,
        ),
        transcript,
    )
    trimmed = _trim_to_limits(ordered, preset, segments)
    return trimmed.model_copy(update={"limitations": _dedupe(trimmed.limitations)})


def cite_supporting_lines(
    claim: Claim,
    segments: dict[str, Segment],
    allowed: set[str] | None = None,
) -> Claim | None:
    """Add lines that contain a name, number, or quotation the claim already uses.

    Returns None when one of those tokens appears nowhere in the searched lines,
    or when the claim still fails after the lines are added. The claim text stays
    unchanged. ``allowed`` limits the search to one chunk and its lookback.
    """

    if not claim.text.strip():
        return None
    resolved, unresolved = _resolve_evidence_ids(claim.evidence_ids, claim.text, segments)
    needed = _lines_for_missing_anchors(claim.text, resolved, segments, allowed)
    if needed is None:
        return None
    if not needed:
        if not unresolved or not resolved:
            return None
        repaired = pin_continuations(
            claim.model_copy(update={"evidence_ids": resolved}),
            segments,
            allowed,
        )
        if claim_problems("Claim", repaired, segments, allowed):
            return None
        return repaired
    chosen: list[str] = []
    for sid in needed + resolved:
        if allowed is not None and sid not in allowed:
            continue
        if sid not in chosen:
            chosen.append(sid)
        if len(chosen) == MAX_EVIDENCE_IDS:
            break
    if not chosen:
        return None
    ordered = sorted(chosen, key=lambda sid: (segments[sid].start, sid))
    repaired = pin_continuations(
        claim.model_copy(update={"evidence_ids": ordered}),
        segments,
        allowed,
    )
    if claim_problems("Claim", repaired, segments, allowed):
        return None
    return repaired


def pin_continuations(
    claim: Claim,
    segments: dict[str, Segment],
    allowed: set[str] | None = None,
) -> Claim:
    """Cite the next caption when an anchor matches only together with it.

    A wrapped auto-caption splits a name across two short lines. Citing the
    first line is enough for the check; this stores the continuation as well.
    """

    resolved, unresolved = _resolve_evidence_ids(claim.evidence_ids, claim.text, segments)
    if unresolved or not resolved:
        return claim
    if _anchors_supported(claim.text, _join_ids(resolved, segments)):
        if resolved == [item.strip() for item in claim.evidence_ids]:
            return claim
        return claim.model_copy(update={"evidence_ids": resolved})
    extended = list(resolved)
    for sid in list(resolved):
        nxt = _continuation_id(sid, segments)
        if nxt is None or nxt in extended:
            continue
        if allowed is not None and nxt not in allowed:
            continue
        if len(extended) >= MAX_EVIDENCE_IDS:
            break
        extended.append(nxt)
        if _anchors_supported(claim.text, _join_ids(extended, segments)):
            break
    if not _anchors_supported(claim.text, _join_ids(extended, segments)):
        return claim
    ordered = sorted(dict.fromkeys(extended), key=lambda sid: (segments[sid].start, sid))
    return claim.model_copy(update={"evidence_ids": ordered})


def _lines_for_missing_anchors(
    text: str,
    evidence_ids: list[str],
    segments: dict[str, Segment],
    allowed: set[str] | None = None,
) -> list[str] | None:
    """Line ids to add. None means a token in the claim appears nowhere in scope."""

    evidence = _window_text(evidence_ids, segments)
    folded = collapse_whitespace(evidence).casefold()
    needed: list[str] = []
    cited = set(evidence_ids)

    def add_first(hits: list[tuple[tuple[str, ...], str]]) -> bool:
        if not hits:
            return False
        for sid in hits[0][0]:
            if sid not in cited and sid not in needed:
                needed.append(sid)
        return True

    for quote in _quotes(text):
        needle = collapse_whitespace(quote).casefold()
        if needle in folded:
            continue
        hits = _anchor_hits(
            segments,
            lambda line, needle=needle: needle in collapse_whitespace(line).casefold(),
            allowed,
        )
        if not add_first(hits):
            return None
    for number in _numbers(text):
        if _number_supported(number, evidence):
            continue
        hits = _anchor_hits(
            segments,
            lambda line, number=number: _number_supported(number, line),
            allowed,
        )
        if not add_first(hits):
            return None
    for name in _names(text):
        forms = _name_forms(name)
        if _folded_has_name(folded, forms):
            continue
        hits = _anchor_hits(
            segments,
            lambda line, forms=forms: _folded_has_name(collapse_whitespace(line).casefold(), forms),
            allowed,
        )
        if not add_first(hits):
            return None
    return needed


def claim_problems(
    label: str,
    claim: Claim,
    segments: dict[str, Segment],
    allowed: set[str] | None = None,
) -> list[str]:
    """Check one claim against the segments it cites.

    Unknown ids stop the check. A cited line that does not end its utterance
    also searches the next caption, so a name wrapped across those two lines
    still counts. ``allowed`` limits where a missing name or number is looked up.
    """

    problems: list[str] = []
    if not claim.text.strip():
        problems.append(f"{label} is empty.")
    raw_ids = [item.strip() for item in claim.evidence_ids]
    if not raw_ids or any(not item for item in raw_ids):
        problems.append(f"{label} has no transcript evidence.")
        return problems
    evidence_ids, unresolved = _resolve_evidence_ids(raw_ids, claim.text, segments)
    if unresolved:
        problems.append(_unknown_id_problem(label, unresolved, segments))
        return problems
    if not evidence_ids:
        problems.append(f"{label} has no transcript evidence.")
        return problems

    evidence = _window_text(evidence_ids, segments)
    folded_evidence = collapse_whitespace(evidence).casefold()
    for quote in _quotes(claim.text):
        if collapse_whitespace(quote).casefold() not in folded_evidence:
            problems.append(
                f"{label} quotes text that is not in the cited transcript: {quote!r}."
            )
    for number in _numbers(claim.text):
        if not _number_supported(number, evidence):
            problems.append(
                _missing_number_problem(label, number, evidence_ids, segments, allowed)
            )
    for name in _names(claim.text):
        if not _folded_has_name(folded_evidence, _name_forms(name)):
            problems.append(
                _missing_name_problem(label, name, evidence_ids, segments, allowed)
            )
    return problems


def _index(transcript: Transcript) -> dict[str, Segment]:
    return {segment.id: segment for segment in transcript.segments}


def _ordered_segments(segments: dict[str, Segment]) -> list[Segment]:
    return sorted(segments.values(), key=lambda segment: (segment.start, segment.id))


def _continuation_id(segment_id: str, segments: dict[str, Segment]) -> str | None:
    """The next caption when this one does not end the utterance."""

    ordered = _ordered_segments(segments)
    for index, segment in enumerate(ordered):
        if segment.id != segment_id:
            continue
        if index + 1 >= len(ordered):
            return None
        nxt = ordered[index + 1]
        if utterance_continues(segment, nxt):
            return nxt.id
        return None
    return None


def _window_ids(evidence_ids: list[str], segments: dict[str, Segment]) -> list[str]:
    """Cited ids plus the next caption when the utterance continues."""

    ordered = _ordered_segments(segments)
    position = {segment.id: index for index, segment in enumerate(ordered)}
    chosen: set[str] = set()
    for sid in evidence_ids:
        if sid not in position:
            continue
        chosen.add(sid)
        index = position[sid]
        if index + 1 >= len(ordered):
            continue
        nxt = ordered[index + 1]
        if utterance_continues(ordered[index], nxt):
            chosen.add(nxt.id)
    return [segment.id for segment in ordered if segment.id in chosen]


def _join_ids(ids: list[str], segments: dict[str, Segment]) -> str:
    chosen = set(ids)
    ordered = _ordered_segments(segments)
    return "\n".join(segment.text for segment in ordered if segment.id in chosen)


def _window_text(evidence_ids: list[str], segments: dict[str, Segment]) -> str:
    return _join_ids(_window_ids(evidence_ids, segments), segments)


def _anchors_supported(text: str, evidence: str) -> bool:
    folded = collapse_whitespace(evidence).casefold()
    for quote in _quotes(text):
        if collapse_whitespace(quote).casefold() not in folded:
            return False
    for number in _numbers(text):
        if not _number_supported(number, evidence):
            return False
    for name in _names(text):
        if not _folded_has_name(folded, _name_forms(name)):
            return False
    return True


def _anchor_hits(
    segments: dict[str, Segment],
    predicate,
    allowed: set[str] | None = None,
) -> list[tuple[tuple[str, ...], str]]:
    """Lines that match. A wrapped pair is used only when no single line does."""

    ordered = _ordered_segments(segments)
    singles: list[tuple[tuple[str, ...], str]] = []
    for segment in ordered:
        if allowed is not None and segment.id not in allowed:
            continue
        if predicate(segment.text):
            singles.append(((segment.id,), segment.text))
    if singles:
        return singles
    pairs: list[tuple[tuple[str, ...], str]] = []
    for index, segment in enumerate(ordered[:-1]):
        nxt = ordered[index + 1]
        if not utterance_continues(segment, nxt):
            continue
        if allowed is not None and (segment.id not in allowed or nxt.id not in allowed):
            continue
        joined = f"{segment.text} {nxt.text}"
        if predicate(joined):
            pairs.append(((segment.id, nxt.id), joined))
    return pairs


def _fresh_hits(
    hits: list[tuple[tuple[str, ...], str]],
    cited: set[str],
) -> list[tuple[tuple[str, ...], str]]:
    """Hits that include a line the claim did not already cite."""

    return [hit for hit in hits if any(sid not in cited for sid in hit[0])]


def _missing_anchor_message(
    problem: str,
    hits: list[tuple[tuple[str, ...], str]],
    allowed: set[str] | None,
    kind: str,
) -> str:
    if not hits:
        scope = "in this span" if allowed is not None else "anywhere in the transcript"
        return f"{problem} It does not appear {scope}. Remove the {kind}."
    ids: list[str] = []
    for hit_ids, _text in hits:
        for sid in hit_ids:
            if sid not in ids:
                ids.append(sid)
            if len(ids) == 3:
                break
        if len(ids) == 3:
            break
    id_list = ", ".join(ids)
    return (
        f"{problem} It appears in {id_list} ({hits[0][1]!r}). "
        f"Cite one of these segments or remove the {kind}."
    )


def _resolve_evidence_ids(
    raw_ids: list[str],
    text: str,
    segments: dict[str, Segment],
) -> tuple[list[str], list[str]]:
    """Expand each entry into known segment ids, in transcript order.

    The second list is the entries that named no transcript line.
    """

    resolved: list[str] = []
    unresolved: list[str] = []
    for raw in raw_ids:
        found = _ids_from_entry(raw, segments)
        if found:
            resolved.extend(found)
        elif raw.strip():
            unresolved.append(raw.strip())
    if not resolved:
        return [], unresolved
    return _cap_ids(resolved, text, segments), unresolved


def _ids_from_entry(raw: str, segments: dict[str, Segment]) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    found: list[str] = []
    parts = [part for part in text.split(",") if part.strip()]
    if not parts:
        return []
    for part in parts:
        token = _without_clock(_strip_brackets(part))
        clock = _CLOCK.fullmatch(token)
        if clock:
            sid = _segment_at_clock(clock, segments)
            if sid is None:
                return []
            found.append(sid)
            continue
        span = _SPAN.fullmatch(token)
        if span:
            found.extend(_expand_span(int(span.group(1)), int(span.group(2)), segments))
            continue
        single = _SINGLE.fullmatch(token)
        if single:
            sid = f"s{int(single.group(1))}"
            if sid in segments:
                found.append(sid)
            continue
        return []
    return found


def _segment_at_clock(clock: re.Match[str], segments: dict[str, Segment]) -> str | None:
    """The caption that starts at a clock time the model wrote instead of an id.

    The time is a pointer, not evidence. The cited caption is still checked.
    When several captions start in the same second the first one is used.
    """

    hours, minutes, seconds = clock.groups()
    total = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
    matches = [segment for segment in segments.values() if int(segment.start) == total]
    if not matches:
        return None
    return min(matches, key=lambda segment: (segment.start, segment.id)).id


def _strip_brackets(text: str) -> str:
    token = text.strip()
    while token.startswith("[") and token.endswith("]") and len(token) >= 2:
        token = token[1:-1].strip()
    # [s13] 00:24 keeps the clock outside the brackets.
    if token.startswith("[") and "]" in token:
        head, _, tail = token.partition("]")
        token = f"{head[1:].strip()} {tail.strip()}".strip()
    # Mismatched or stray brackets such as [481) or (s481] still name a line.
    return token.strip(_BRACKETS + " ")


def _without_clock(token: str) -> str:
    """Drop a trailing clock time so ``s13 00:24`` resolves to ``s13``."""

    return _CLOCK_SUFFIX.sub("", token).strip()


def _expand_span(start: int, end: int, segments: dict[str, Segment]) -> list[str]:
    lo, hi = sorted((start, end))
    found = [f"s{number}" for number in range(lo, hi + 1) if f"s{number}" in segments]
    return sorted(found, key=lambda sid: (segments[sid].start, sid))


def _cap_ids(ids: list[str], text: str, segments: dict[str, Segment]) -> list[str]:
    ordered = sorted(dict.fromkeys(ids), key=lambda sid: (segments[sid].start, sid))
    if len(ordered) <= MAX_EVIDENCE_IDS:
        return ordered
    anchors = [sid for sid in ordered if _line_supports(text, _window_text([sid], segments))]
    chosen: list[str] = []
    for sid in anchors + ordered:
        if sid in chosen:
            continue
        chosen.append(sid)
        if len(chosen) == MAX_EVIDENCE_IDS:
            break
    return sorted(chosen, key=lambda sid: (segments[sid].start, sid))


def _line_supports(text: str, line: str) -> bool:
    folded_line = collapse_whitespace(line).casefold()
    if any(collapse_whitespace(quote).casefold() in folded_line for quote in _quotes(text)):
        return True
    if any(_number_supported(number, line) for number in _numbers(text)):
        return True
    return any(_folded_has_name(folded_line, _name_forms(name)) for name in _names(text))


def _unknown_id_problem(
    label: str,
    unresolved: list[str],
    segments: dict[str, Segment],
) -> str:
    """Tell the retry what was wrong with each entry and what a line id looks like."""

    shown = ", ".join(unresolved)
    example = _example_id(segments)
    message = (
        f"{label} cites {shown}, which is not a line id. "
        f"Each evidence entry is one line id: the letter s and a number, such as {example}."
    )
    times = [
        item
        for item in unresolved
        if _CLOCK.fullmatch(_without_clock(_strip_brackets(item)))
    ]
    if times:
        message += (
            f" {', '.join(times)} is a clock time. Cite the id in brackets before the line."
        )
    ends = _valid_span_ends(unresolved, segments)
    if ends:
        message += f" The lines at the ends of that span are {', '.join(ends)}."
    return message


def _example_id(segments: dict[str, Segment]) -> str:
    """A real id from the middle of the transcript, so a long id looks normal."""

    ordered = _ordered_segments(segments)
    if not ordered:
        return "s1"
    return ordered[len(ordered) // 2].id


def _valid_span_ends(raw_entries: list[str], segments: dict[str, Segment]) -> list[str]:
    numbers = sorted(
        int(match.group(1))
        for sid in segments
        if (match := _SINGLE.fullmatch(sid))
    )
    if not numbers:
        return []
    chosen: list[int] = []
    for raw in raw_entries:
        for part in raw.split(","):
            span = _SPAN.fullmatch(_strip_brackets(part))
            if span is None:
                continue
            for end in (int(span.group(1)), int(span.group(2))):
                nearest = min(numbers, key=lambda number: (abs(number - end), number))
                if nearest not in chosen:
                    chosen.append(nearest)
    return [f"s{number}" for number in chosen]


def _earliest(claim: Claim, segments: dict[str, Segment], index: int) -> tuple[float, str, int]:
    cited = [segments[item] for item in claim.evidence_ids if item in segments]
    if not cited:
        return (float("inf"), "", index)
    first = min(cited, key=lambda segment: (segment.start, segment.id))
    return (first.start, first.id, index)


def _select_spread(points: list[Claim], keep: int) -> list[Claim]:
    """Keep the opening, closing, and evenly spaced points between them."""

    return [points[index] for index in _spread_indexes(len(points), keep)]


def _spread_indexes(count: int, keep: int) -> list[int]:
    if keep >= count:
        return list(range(count))
    if keep <= 0:
        return []
    if keep == 1:
        return [0]
    indexes = [round(step * (count - 1) / (keep - 1)) for step in range(keep)]
    ordered: list[int] = []
    for index in indexes:
        if index not in ordered:
            ordered.append(index)
    cursor = 0
    while len(ordered) < keep and cursor < count:
        if cursor not in ordered:
            ordered.append(cursor)
        cursor += 1
    ordered.sort()
    return ordered


def _drop_redundant(points: list[Claim], segments: dict[str, Segment]) -> list[Claim]:
    """Drop the point closest to its neighbours, keeping the opening and closing."""

    if len(points) <= 1:
        return []
    if len(points) == 2:
        return [points[0]]
    starts = [_start(point, segments) for point in points]
    best = 1
    best_gap: float | None = None
    for index in range(1, len(points) - 1):
        gap = min(starts[index] - starts[index - 1], starts[index + 1] - starts[index])
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = index
    return points[:best] + points[best + 1:]


def _start(claim: Claim, segments: dict[str, Segment]) -> float:
    cited = [segments[item].start for item in claim.evidence_ids if item in segments]
    return min(cited) if cited else 0.0


def _quotes(text: str) -> list[str]:
    return [match.group(1).strip() for match in _QUOTE.finditer(text) if match.group(1).strip()]


def _numbers(text: str) -> list[str]:
    visible = _TIMECODE.sub(" ", text)
    return _NUMBER.findall(visible)


def _number_forms(token: str) -> set[str]:
    """Equate 1,200 with 1200, and 87% with 87, without matching a shorter number inside a longer one."""
    forms = {token, token.replace(",", "")}
    if token.endswith("%"):
        forms.add(token[:-1])
        forms.add(token[:-1].replace(",", ""))
    return {form for form in forms if form}


def _number_supported(token: str, evidence: str) -> bool:
    evidence_forms: set[str] = set()
    for match in _NUMBER.findall(evidence):
        evidence_forms.update(_number_forms(match))
    return bool(_number_forms(token) & evidence_forms)


def _name_forms(name: str) -> set[str]:
    """Spellings that count as the same name.

    The name itself is kept, so a shorter stem does not match: Carters
    matches Carter, and it does not match Cart. Dropping one trailing s
    yields the singular. A name that does not end in s is also accepted
    with one added, so Carter still matches a longer form such as Carters.
    """

    folded = collapse_whitespace(name).casefold()
    forms = {folded}
    if folded.endswith("s") and not folded.endswith("ss"):
        singular = folded[:-1]
        if len(singular) >= 4:
            forms.add(singular)
    elif folded:
        forms.add(folded + "s")
    return {form for form in forms if form}


def _folded_has_name(folded: str, forms: set[str]) -> bool:
    return any(form in folded for form in forms)


def _missing_number_problem(
    label: str,
    number: str,
    evidence_ids: list[str],
    segments: dict[str, Segment],
    allowed: set[str] | None = None,
) -> str:
    """Point the retry at lines that contain the number, or tell the model to drop it."""

    problem = f"{label} uses the number {number}, which is not in the cited transcript."
    hits = _fresh_hits(
        _anchor_hits(segments, lambda line: _number_supported(number, line), allowed),
        set(evidence_ids),
    )
    return _missing_anchor_message(problem, hits, allowed, "number")


def _missing_name_problem(
    label: str,
    name: str,
    evidence_ids: list[str],
    segments: dict[str, Segment],
    allowed: set[str] | None = None,
) -> str:
    """Point the retry at lines that contain the name, or tell the model to drop it."""

    problem = f"{label} names {name!r}, which is not in the cited transcript."
    forms = _name_forms(name)
    hits = _fresh_hits(
        _anchor_hits(
            segments,
            lambda line: _folded_has_name(collapse_whitespace(line).casefold(), forms),
            allowed,
        ),
        set(evidence_ids),
    )
    return _missing_anchor_message(problem, hits, allowed, "name")


def _names(text: str) -> list[str]:
    """Names that must appear in the cited lines.

    Multi-word names and acronyms count anywhere. A single capitalised word at
    the start of a sentence is ordinary English, so it is not treated as a name.
    """
    phrases = _TITLE_PHRASE.findall(text)
    # Hide those phrases so their individual words are not reported again.
    masked = _TITLE_PHRASE.sub(lambda match: " " * len(match.group(0)), text)
    acronyms = _ACRONYM.findall(masked)
    singles: list[str] = []
    for sentence in _SENTENCE_BREAK.split(masked):
        tokens = _WORD.findall(sentence)
        for index, token in enumerate(tokens):
            if index == 0:
                continue
            if re.fullmatch(r"[A-Z][a-z]{2,}", token):
                singles.append(token)
    return phrases + acronyms + singles


def _trim_to_limits(
    summary: SummaryResponse,
    preset: LengthPreset,
    segments: dict[str, Segment],
) -> SummaryResponse:
    key_points = list(summary.key_points)
    conclusion = summary.conclusion
    limitations = list(summary.limitations)
    dropped_points = False
    dropped_words = False

    def current() -> SummaryResponse:
        return SummaryResponse(
            main_idea=summary.main_idea,
            key_points=key_points,
            conclusion=conclusion,
            limitations=limitations,
        )

    budget = preset.max_points - (1 if conclusion is not None else 0)
    if len(key_points) > max(budget, 0):
        key_points = _select_spread(key_points, max(budget, 0))
        dropped_points = True
    while key_points and word_count(current()) > preset.max_words:
        key_points = _drop_redundant(key_points, segments)
        dropped_words = True
    # The main idea stays even when it alone exceeds the word ceiling.
    if conclusion is not None and word_count(current()) > preset.max_words:
        conclusion = None
        limitations.append("Removed the conclusion to meet the length limit.")
    if dropped_points:
        limitations.append(
            "Omitted points to meet the point limit while keeping the opening and closing of the transcript."
        )
    if dropped_words:
        limitations.append(
            "Omitted points to meet the length limit while keeping the opening and closing of the transcript."
        )
    if word_count(current()) > preset.max_words:
        limitations.append(
            "The central statement exceeds the length preset; it was kept because "
            "it is supported by the transcript."
        )
    return current()


def _clean_limitations(items: list[str]) -> list[str]:
    return [item.strip() for item in items if item and item.strip()]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
