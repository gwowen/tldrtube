"""Reject summary claims that are not tied to the transcript.

These checks catch unknown ids, missing citations, mismatched quotations,
unsupported numbers, and unknown names. They do not prove that a paraphrase
says what the cited lines mean. A claim that fails is dropped. When the main
idea fails, the summary is refused.
"""

import re

from tldrtube.models import MAX_EVIDENCE_IDS, Claim, LengthPreset, Segment, SummaryResponse, Transcript
from tldrtube.transcript import collapse_whitespace

_QUOTE = re.compile(r'["“”]([^"“”]{4,})["“”]')
# Explicit numerals only. Spelled-out numbers such as "forty" are left alone.
_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\b\d+(?:\.\d+)?%?")
# Clock times the model may echo are not factual claims.
_TIMECODE = re.compile(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?")
_TITLE_PHRASE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_WORD = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
# A model often writes a caption span instead of one id per line.
_SPAN = re.compile(r"^s(\d+)\s*[-–—]\s*s?(\d+)$", re.IGNORECASE)
_SINGLE = re.compile(r"^s(\d+)$", re.IGNORECASE)


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
    limitations = _clean_limitations(summary.limitations)
    cited_extra = False
    main_idea = summary.main_idea
    main_problems = claim_problems("The main idea", main_idea, segments)
    if main_problems:
        repaired = _cite_supporting_lines(main_idea, segments)
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
            repaired = _cite_supporting_lines(point, segments)
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
        repaired = _cite_supporting_lines(conclusion, segments)
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


def _cite_supporting_lines(claim: Claim, segments: dict[str, Segment]) -> Claim | None:
    """Add lines that contain a name, number, or quotation the claim already uses.

    Returns None when one of those tokens appears nowhere in the transcript, or
    when the claim still fails after the lines are added. The claim text stays
    unchanged.
    """

    if not claim.text.strip():
        return None
    resolved, unresolved = _resolve_evidence_ids(claim.evidence_ids, claim.text, segments)
    needed = _lines_for_missing_anchors(claim.text, resolved, segments)
    if needed is None:
        return None
    if not needed:
        if not unresolved or not resolved:
            return None
        repaired = claim.model_copy(update={"evidence_ids": resolved})
        if claim_problems("Claim", repaired, segments):
            return None
        return repaired
    chosen: list[str] = []
    for sid in needed + resolved:
        if sid not in chosen:
            chosen.append(sid)
        if len(chosen) == MAX_EVIDENCE_IDS:
            break
    ordered = sorted(chosen, key=lambda sid: (segments[sid].start, sid))
    repaired = claim.model_copy(update={"evidence_ids": ordered})
    if claim_problems("Claim", repaired, segments):
        return None
    return repaired


def _lines_for_missing_anchors(
    text: str,
    evidence_ids: list[str],
    segments: dict[str, Segment],
) -> list[str] | None:
    """Line ids to add. None means a token in the claim appears nowhere."""

    evidence = "\n".join(segments[sid].text for sid in evidence_ids if sid in segments)
    folded = collapse_whitespace(evidence).casefold()
    needed: list[str] = []
    cited = set(evidence_ids)

    def add_first(found: list[Segment]) -> bool:
        for segment in found:
            if segment.id not in cited and segment.id not in needed:
                needed.append(segment.id)
                return True
        return bool(found)

    for quote in _quotes(text):
        if collapse_whitespace(quote).casefold() in folded:
            continue
        needle = collapse_whitespace(quote).casefold()
        found = [
            segment
            for segment in segments.values()
            if needle in collapse_whitespace(segment.text).casefold()
        ]
        if not add_first(found):
            return None
    for number in _numbers(text):
        if _number_supported(number, evidence):
            continue
        found = [
            segment
            for segment in segments.values()
            if _number_supported(number, segment.text)
        ]
        if not add_first(found):
            return None
    for name in _names(text):
        forms = _name_forms(name)
        if _folded_has_name(folded, forms):
            continue
        found = [
            segment
            for segment in segments.values()
            if _folded_has_name(collapse_whitespace(segment.text).casefold(), forms)
        ]
        if not add_first(found):
            return None
    return needed


def claim_problems(
    label: str,
    claim: Claim,
    segments: dict[str, Segment],
) -> list[str]:
    """Check one claim against the segments it cites.

    Unknown ids stop the check. Comparing a quotation with lines the model did
    not cite would make a bad citation look supported.
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

    evidence = "\n".join(segments[item].text for item in evidence_ids)
    folded_evidence = collapse_whitespace(evidence).casefold()
    for quote in _quotes(claim.text):
        if collapse_whitespace(quote).casefold() not in folded_evidence:
            problems.append(
                f"{label} quotes text that is not in the cited transcript: {quote!r}."
            )
    for number in _numbers(claim.text):
        if not _number_supported(number, evidence):
            problems.append(
                _missing_number_problem(label, number, evidence_ids, segments)
            )
    for name in _names(claim.text):
        if not _folded_has_name(folded_evidence, _name_forms(name)):
            problems.append(_missing_name_problem(label, name, evidence_ids, segments))
    return problems


def _index(transcript: Transcript) -> dict[str, Segment]:
    return {segment.id: segment for segment in transcript.segments}


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
        token = _strip_brackets(part)
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


def _strip_brackets(text: str) -> str:
    token = text.strip()
    while token.startswith("[") and token.endswith("]") and len(token) >= 2:
        token = token[1:-1].strip()
    return token


def _expand_span(start: int, end: int, segments: dict[str, Segment]) -> list[str]:
    lo, hi = sorted((start, end))
    found = [f"s{number}" for number in range(lo, hi + 1) if f"s{number}" in segments]
    return sorted(found, key=lambda sid: (segments[sid].start, sid))


def _cap_ids(ids: list[str], text: str, segments: dict[str, Segment]) -> list[str]:
    ordered = sorted(dict.fromkeys(ids), key=lambda sid: (segments[sid].start, sid))
    if len(ordered) <= MAX_EVIDENCE_IDS:
        return ordered
    anchors = [sid for sid in ordered if _line_supports(text, segments[sid].text)]
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
    """Tell the retry that each evidence entry is one line id."""

    shown = ", ".join(unresolved)
    message = (
        f"{label} cites {shown}, which is not a line id. "
        "Each evidence entry is one line id."
    )
    ends = _valid_span_ends(unresolved, segments)
    if ends:
        message += f" The lines at the ends of that span are {', '.join(ends)}."
    return message


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
) -> str:
    """Point the retry at lines that contain the number, or tell the model to drop it."""

    problem = f"{label} uses the number {number}, which is not in the cited transcript."
    cited = set(evidence_ids)
    found = [
        segment
        for segment in segments.values()
        if segment.id not in cited and _number_supported(number, segment.text)
    ]
    if not found:
        return f"{problem} It does not appear anywhere in the transcript. Remove the number."
    shown = found[:3]
    id_list = ", ".join(segment.id for segment in shown)
    return (
        f"{problem} It appears in {id_list} ({shown[0].text!r}). "
        "Cite one of these segments or remove the number."
    )


def _missing_name_problem(
    label: str,
    name: str,
    evidence_ids: list[str],
    segments: dict[str, Segment],
) -> str:
    """Point the retry at lines that contain the name, or tell the model to drop it."""

    problem = f"{label} names {name!r}, which is not in the cited transcript."
    forms = _name_forms(name)
    cited = set(evidence_ids)
    found = [
        segment
        for segment in segments.values()
        if segment.id not in cited
        and _folded_has_name(collapse_whitespace(segment.text).casefold(), forms)
    ]
    if not found:
        return f"{problem} It does not appear anywhere in the transcript. Remove the name."
    shown = found[:3]
    id_list = ", ".join(segment.id for segment in shown)
    return (
        f"{problem} It appears in {id_list} ({shown[0].text!r}). "
        "Cite one of these segments or remove the name."
    )


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
