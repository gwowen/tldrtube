# tldrtube

Too many videos, too little time? Wondering about whether that epic hour long YouTube video is for you and just want to get the gist of it? 

Well, tldrtube is for you!

tldrtube is a command-line tool that writes a short, evidence-grounded summary of a YouTube video from its transcript when provided with a YouTube URL. 

A local model, served by [Ollama](https://ollama.com/), condenses the words that were spoken. Python checks that every retained claim points at a real transcript segment.

The model may condense, reorganise, and paraphrase. It must not add facts, names, statistics, or recommendations that the transcript does not support. tldrtube does not watch the video, read the title or description, or check claims against the web. Material that appears only on screen is missing from the summary.

This project is licensed under the MIT License. Copyright (c) 2026 Gareth Owen.

## Requirements

- Python 3.11 or newer
- [Ollama](https://ollama.com/) running locally
- The default model:

```bash
ollama pull LiquidAI/lfm2.5-2.6b
```

## Install

`uv sync` only fills the project virtualenv, so the `tldrtube` command is not on your `PATH` afterwards. Install the command with:

```bash
uv tool install --editable .
```

That places `tldrtube` in `~/.local/bin`, and editable mode picks up later changes in this checkout. From the project directory, without installing the command, run:

```bash
uv run tldrtube "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Use

```bash
tldrtube "https://www.youtube.com/watch?v=VIDEO_ID"
```

A bare video id, a `youtu.be` link, and YouTube shorts, embed, and live URLs are accepted. Human-written captions are preferred. Automatically generated captions are used when no human track exists for the requested language.

```bash
tldrtube URL --length short
tldrtube URL --length medium
tldrtube URL --length detailed
tldrtube URL --format text
tldrtube URL --format markdown
tldrtube URL --format json
tldrtube URL --model MODEL_NAME
tldrtube URL --language en
tldrtube URL --ollama-url http://127.0.0.1:11434
```

`--model` replaces the model named in `tldrtube_config.toml`. The default length is medium and the default format is plain text. Progress is printed to stderr while the summary is written to stdout. `OLLAMA_HOST` is used when `--ollama-url` is omitted. If the host has no scheme, `http://` is added.

| Preset | Words | Points |
| --- | --- | --- |
| short | 50–80 | 3 |
| medium | 100–150 | 3–5 |
| detailed | 200–300 | 5–7 |

Timestamps such as `[02:14]` come from the cited caption, not from the model. JSON includes the same timestamps alongside the evidence ids.

## Configuration

tldrtube reads `tldrtube_config.toml` from the current working directory. A missing file uses the built-in defaults, so an installed command still runs from any directory. This repository includes a file with those defaults:

```toml
model = "LiquidAI/lfm2.5-2.6b"
temperature = 0.8
```

`model` is the Ollama model name. `temperature` is sent to Ollama. `0.8` is Ollama's own default, so two runs of the same video can differ. A present file may omit either key. An unknown key, a model that is not a non-empty string, a temperature that is not a number, or a negative temperature stops the program with exit code 1.

## What a failure means

- Exit code 1: the URL is invalid, captions are disabled, no transcript exists in the requested language, or `tldrtube_config.toml` cannot be used. The title and description are not sent to the model.
- Exit code 2: the summary could not be tied to the transcript. tldrtube does not print a polished answer in that case.
- Exit code 3: Ollama could not be reached, or the model did not return a readable summary. When the requested model is not installed, tldrtube lists the models Ollama has and points at `tldrtube_config.toml` or `ollama pull`.

The checker rejects unknown segment ids, claims with no evidence, quotations that do not appear in the cited lines, and explicit numbers or names that the cited lines do not contain. A paraphrase can still misread a real quotation. When that cannot be ruled out, the claim is dropped rather than checked by a second model.

## Development

```bash
uv run pytest
```

The tests do not call YouTube or Ollama.
