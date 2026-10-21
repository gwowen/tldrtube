# tldrtube agent guide

tldrtube summarises a YouTube transcript with a local model. The model may condense, reorganise, and paraphrase that transcript. It must not introduce information the transcript does not support.

## Responsibilities

Python owns retrieval, normalisation, chunking, segment ids, timestamps, verification, and rendering.

`langchain_ollama.ChatOllama` is the only model client. Bind it with `with_structured_output(..., method="json_schema")`, `temperature=0`, a bounded `num_predict`, and `reasoning=False`. Do not add agents, tools, embeddings, vector stores, or a second model that checks the summary.

## Transcripts

Prefer manually created captions, then generated captions. Keep the original order and timestamps. Flag obvious non-speech cues such as `[Music]`. Do not drop introductions, sponsorships, or digressions before the model sees them.

If no suitable transcript can be retrieved, fail with an informative error. Do not summarise from the title or description.

## Summaries

Short transcripts are one pass. Longer transcripts are extracted chunk by chunk, then condensed. Pass B must cite the original transcript segments. Text generated in pass A is not evidence.

Ask for one corrected response when verification fails. Then drop invalid claims. If the main idea still has no valid evidence, fail closed and do not print a polished summary.

Timestamps shown to the user are computed from the cited segments.

## Layout

Keep the `src/` layout and Python 3.11. The modules are `models`, `transcript`, `summarizer`, `verifier`, and `cli`. Tests stay offline: mock transcript fetch and inject a fake chat model.

## License

The license stays MIT, copyright Gareth Owen.
