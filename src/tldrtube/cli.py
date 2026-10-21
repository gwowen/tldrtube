"""Command-line interface for tldrtube.

The summary goes to stdout. Errors go to stderr so a successful summary can
be piped. Exit 1 is a usage or transcript failure, 2 means the summary was
not supported, and 3 means Ollama could not be used.
"""

import argparse
import json
import logging
import sys

from tldrtube.models import DEFAULT_MODEL, PRESETS, Claim, SummaryResponse, Transcript
from tldrtube.summarizer import ModelError, resolve_base_url, summarize
from tldrtube.transcript import TranscriptError, fetch_transcript, format_timestamp
from tldrtube.verifier import VerificationError

logger = logging.getLogger("tldrtube")


class _StderrHandler(logging.StreamHandler):
    """Write to whatever stderr is current, including after a test capture ends."""

    def __init__(self) -> None:
        super().__init__(stream=sys.stderr)

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def configure_logging() -> None:
    """Print progress to stderr. The summary itself stays on stdout."""

    logger.setLevel(logging.INFO)
    logger.propagate = False
    if any(isinstance(handler, _StderrHandler) for handler in logger.handlers):
        return
    handler = _StderrHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse exits with 2. Usage failures share exit code 1 with transcript failures.
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="tldrtube",
        description="Summarise a YouTube video from its transcript with a local model.",
    )
    parser.add_argument("url", help="YouTube URL or video id")
    parser.add_argument(
        "--length",
        choices=tuple(PRESETS),
        default="medium",
        help="summary length preset (default: medium)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="output format (default: text)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="preferred caption language (default: en)",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL (default: OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    """Fetch, summarise, and print. The return value is the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    try:
        transcript = fetch_transcript(args.url, language=args.language)
        summary = summarize(
            transcript,
            preset=PRESETS[args.length],
            model=args.model,
            base_url=resolve_base_url(args.ollama_url),
        )
    except TranscriptError as exc:
        # Missing captions are a data failure, not a reason to invent a summary.
        print(f"tldrtube: {exc}", file=sys.stderr)
        return 1
    except VerificationError as exc:
        # Fail closed: an unsupported main idea is not printed as a finished summary.
        print(f"tldrtube: {exc}", file=sys.stderr)
        return 2
    except ModelError as exc:
        print(f"tldrtube: {exc}", file=sys.stderr)
        return 3
    sys.stdout.write(render(summary, transcript, args.format))
    return 0


def render(summary: SummaryResponse, transcript: Transcript, fmt: str) -> str:
    """Render text, Markdown, or JSON. Timestamps are added here, not by the model."""
    if fmt == "json":
        return json.dumps(_payload(summary, transcript), indent=2, ensure_ascii=False) + "\n"
    return _render_text(summary, transcript, markdown=fmt == "markdown")


def main(argv: list[str] | None = None) -> None:
    sys.exit(run(argv))


def _render_text(summary: SummaryResponse, transcript: Transcript, *, markdown: bool) -> str:
    lines = [
        _heading("TL;DR", markdown),
        "",
        summary.main_idea.text.strip(),
        "",
        _heading("Key points", markdown),
        "",
    ]
    for claim in _bullets(summary):
        lines.append(f"- {claim.text.strip()} — [{_timestamp(claim, transcript)}]")
    if summary.limitations:
        lines.extend(["", _heading("Limitations", markdown), ""])
        for item in summary.limitations:
            lines.append(f"- {item.strip()}")
    return "\n".join(lines).rstrip() + "\n"


def _heading(text: str, markdown: bool) -> str:
    return f"**{text}**" if markdown else text


def _bullets(summary: SummaryResponse) -> list[Claim]:
    """Key points first, then the conclusion as the last bullet when one was kept."""
    claims = list(summary.key_points)
    if summary.conclusion is not None:
        claims.append(summary.conclusion)
    return claims


def _timestamp(claim: Claim, transcript: Transcript) -> str:
    """Use the earliest cited caption. The model does not choose the clock time."""
    segments = {segment.id: segment for segment in transcript.segments}
    cited = [segments[item] for item in claim.evidence_ids if item in segments]
    earliest = min(cited, key=lambda segment: (segment.start, segment.id))
    return format_timestamp(earliest.start)


def _payload(summary: SummaryResponse, transcript: Transcript) -> dict[str, object]:
    def claim_payload(claim: Claim) -> dict[str, object]:
        return {
            "text": claim.text,
            "evidence_ids": claim.evidence_ids,
            "timestamp": _timestamp(claim, transcript),
        }

    conclusion = None if summary.conclusion is None else claim_payload(summary.conclusion)
    return {
        "video_id": transcript.video_id,
        "main_idea": claim_payload(summary.main_idea),
        "key_points": [claim_payload(point) for point in summary.key_points],
        "conclusion": conclusion,
        "limitations": summary.limitations,
    }


if __name__ == "__main__":
    main()
