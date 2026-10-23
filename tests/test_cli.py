"""Command-line rendering and exit codes."""

import json

import httpx
import pytest

from tldrtube.cli import render, run
from tldrtube.models import DEFAULT_MODEL, PRESETS, Claim, SummaryResponse
from tldrtube.summarizer import ModelError
from tldrtube.transcript import SINGLE_PASS_CHAR_LIMIT, TranscriptError, build_transcript
from tldrtube.verifier import VerificationError

VIDEO_ID = "dQw4w9WgXcQ"


def sample_transcript():
    return build_transcript(
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=False,
        snippets=[
            (134.0, 2.0, "The central example starts here."),
            (3725.0, 2.0, "The closing example comes much later."),
        ],
    )


def sample_summary() -> SummaryResponse:
    return SummaryResponse(
        main_idea=Claim(text="The video explains the central example.", evidence_ids=["s1"]),
        key_points=[
            Claim(text="The closing example comes later.", evidence_ids=["s2", "s1"]),
        ],
        conclusion=Claim(text="The speaker closes on the later example.", evidence_ids=["s2"]),
        limitations=["Some transcript spans did not yield a supported point."],
    )


def test_text_markdown_and_json_use_transcript_timestamps() -> None:
    source = sample_transcript()
    summary = sample_summary()
    text = render(summary, source, "text")
    assert text.startswith("TL;DR\n")
    assert "**TL;DR**" not in text
    assert "- The closing example comes later. — [02:14]" in text
    assert "- The speaker closes on the later example. — [1:02:05]" in text
    assert "Limitations" in text

    markdown = render(summary, source, "markdown")
    assert "**TL;DR**" in markdown
    assert "**Key points**" in markdown
    assert "**Limitations**" in markdown

    payload = json.loads(render(summary, source, "json"))
    assert payload["video_id"] == VIDEO_ID
    assert payload["main_idea"]["timestamp"] == "02:14"
    assert payload["key_points"][0]["timestamp"] == "02:14"
    assert payload["key_points"][0]["evidence_ids"] == ["s2", "s1"]
    assert payload["conclusion"]["timestamp"] == "1:02:05"


def skip_model_check(monkeypatch) -> None:
    monkeypatch.setattr("tldrtube.cli.require_installed_model", lambda *args, **kwargs: None)


class _TagsResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_cli_writes_the_summary_to_stdout(monkeypatch, capsys) -> None:
    source = sample_transcript()
    summary = sample_summary()
    skip_model_check(monkeypatch)
    monkeypatch.setattr("tldrtube.cli.fetch_transcript", lambda *args, **kwargs: source)
    monkeypatch.setattr("tldrtube.cli.summarize", lambda *args, **kwargs: summary)
    assert run([f"https://youtu.be/{VIDEO_ID}", "--format", "markdown"]) == 0
    captured = capsys.readouterr()
    assert "**TL;DR**" in captured.out
    assert captured.err == ""


def test_cli_exit_codes(monkeypatch, capsys) -> None:
    source = sample_transcript()
    skip_model_check(monkeypatch)

    def fail_transcript(*args, **kwargs):
        raise TranscriptError("Captions are disabled for this video.")

    monkeypatch.setattr("tldrtube.cli.fetch_transcript", fail_transcript)
    assert run([VIDEO_ID]) == 1
    captured = capsys.readouterr()
    assert "Captions are disabled" in captured.err
    assert captured.out == ""

    monkeypatch.setattr("tldrtube.cli.fetch_transcript", lambda *args, **kwargs: source)

    def fail_verification(*args, **kwargs):
        raise VerificationError("A faithful summary could not be produced.")

    monkeypatch.setattr("tldrtube.cli.summarize", fail_verification)
    assert run([VIDEO_ID]) == 2
    assert "faithful summary" in capsys.readouterr().err

    def fail_model(*args, **kwargs):
        raise ModelError("Could not reach Ollama.")

    monkeypatch.setattr("tldrtube.cli.summarize", fail_model)
    assert run([VIDEO_ID, "--model", DEFAULT_MODEL]) == 3
    assert "Ollama" in capsys.readouterr().err


def transcript_of_length(chars: int):
    """Two captions whose text totals chars, so the sample summary can cite both."""

    first = chars // 2
    second = chars - first
    return build_transcript(
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=False,
        snippets=[(0.0, 1.0, "a" * first), (1.0, 1.0, "b" * second)],
    )


def test_omitted_length_uses_detailed_only_past_one_pass(monkeypatch, capsys) -> None:
    summary = sample_summary()
    seen: dict[str, object] = {}

    def capture(*args, **kwargs):
        seen["preset"] = kwargs["preset"]
        return summary

    skip_model_check(monkeypatch)
    monkeypatch.setattr("tldrtube.cli.summarize", capture)

    def chosen(source, argv: list[str]):
        monkeypatch.setattr("tldrtube.cli.fetch_transcript", lambda *args, **kwargs: source)
        assert run(argv) == 0
        captured = capsys.readouterr()
        return seen["preset"], captured.err

    preset, err = chosen(sample_transcript(), [VIDEO_ID])
    assert preset == PRESETS["medium"]
    assert "longer than one pass" not in err

    preset, err = chosen(transcript_of_length(SINGLE_PASS_CHAR_LIMIT), [VIDEO_ID])
    assert preset == PRESETS["medium"]
    assert "longer than one pass" not in err

    long = transcript_of_length(SINGLE_PASS_CHAR_LIMIT + 1)
    preset, err = chosen(long, [VIDEO_ID])
    assert preset == PRESETS["detailed"]
    assert "longer than one pass" in err

    preset, err = chosen(long, [VIDEO_ID, "--length", "short"])
    assert preset == PRESETS["short"]
    assert "longer than one pass" not in err


def test_usage_error_exits_with_code_1() -> None:
    with pytest.raises(SystemExit) as exc:
        run([])
    assert exc.value.code == 1


def test_config_supplies_model_and_temperature(monkeypatch, tmp_path, capsys) -> None:
    (tmp_path / "tldrtube_config.toml").write_text(
        'model = "config-model"\ntemperature = 0.2\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    source = sample_transcript()
    summary = sample_summary()
    seen: dict[str, object] = {}

    def capture(*args, **kwargs):
        seen["model"] = kwargs["model"]
        seen["temperature"] = kwargs["temperature"]
        return summary

    skip_model_check(monkeypatch)
    monkeypatch.setattr("tldrtube.cli.fetch_transcript", lambda *args, **kwargs: source)
    monkeypatch.setattr("tldrtube.cli.summarize", capture)
    assert run([VIDEO_ID]) == 0
    assert seen == {"model": "config-model", "temperature": 0.2}
    capsys.readouterr()

    assert run([VIDEO_ID, "--model", "flag-model"]) == 0
    assert seen["model"] == "flag-model"
    assert seen["temperature"] == 0.2


def test_invalid_config_exits_before_fetch(monkeypatch, tmp_path, capsys) -> None:
    (tmp_path / "tldrtube_config.toml").write_text("temperature = -1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("transcript fetch should not run")

    monkeypatch.setattr("tldrtube.cli.fetch_transcript", fail_fetch)
    assert run([VIDEO_ID]) == 1
    captured = capsys.readouterr()
    assert "temperature must be 0 or greater" in captured.err
    assert captured.out == ""


def test_unreachable_ollama_skips_the_transcript(monkeypatch, capsys) -> None:
    def fail_fetch(*args, **kwargs):
        raise AssertionError("transcript fetch should not run")

    def fail_tags(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("tldrtube.cli.fetch_transcript", fail_fetch)
    monkeypatch.setattr("tldrtube.summarizer.httpx.get", fail_tags)
    assert run([VIDEO_ID]) == 3
    err = capsys.readouterr().err
    assert "Start Ollama" in err
    assert "ollama pull" not in err


def test_missing_model_skips_the_transcript_and_lists_installed_models(monkeypatch, capsys) -> None:
    def fail_fetch(*args, **kwargs):
        raise AssertionError("transcript fetch should not run")

    def tags(*args, **kwargs):
        return _TagsResponse({"models": [{"name": "gemma3:4b"}, {"model": "llama3.2:latest"}]})

    monkeypatch.setattr("tldrtube.cli.fetch_transcript", fail_fetch)
    monkeypatch.setattr("tldrtube.summarizer.httpx.get", tags)
    assert run([VIDEO_ID, "--model", "missing-model"]) == 3
    err = capsys.readouterr().err
    assert "gemma3:4b" in err
    assert "llama3.2:latest" in err
    assert "tldrtube_config.toml" in err
    assert "ollama pull missing-model" in err


def test_installed_model_still_fetches_the_transcript(monkeypatch, capsys) -> None:
    source = sample_transcript()
    fetched = {"called": False}

    def fetch(*args, **kwargs):
        fetched["called"] = True
        return source

    def tags(*args, **kwargs):
        return _TagsResponse({"models": [{"name": DEFAULT_MODEL}]})

    monkeypatch.setattr("tldrtube.cli.fetch_transcript", fetch)
    monkeypatch.setattr("tldrtube.cli.summarize", lambda *args, **kwargs: sample_summary())
    monkeypatch.setattr("tldrtube.summarizer.httpx.get", tags)
    assert run([VIDEO_ID, "--model", DEFAULT_MODEL]) == 0
    assert fetched["called"] is True
    capsys.readouterr()
