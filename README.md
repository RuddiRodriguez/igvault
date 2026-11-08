# igvault

Turn your **saved Instagram videos** into a clean, searchable folder library — automatically grouped by topic.

```
Instagram saves
      │
      ▼
 Instagram API ──► data/raw/      (videos + captions + metadata)
      │
      ▼
  index ────────► data/work/records.jsonl
      │
      ▼
  Whisper ──────► data/work/transcripts/*.txt
      │
      ▼
  LLM: describe each video ──► data/work/descriptions.json (cached)
      │      (caption, hashtags, transcript; video frames when the caption says little)
      ▼
  LLM: design topic folders ──► LLM: file every video
      │
      ▼
  organize ─────► library/<Topic>/<date>_<shortcode>_<author>.mp4
                  library/_index.csv
                  library/README.md
```

Topics are **discovered from your content** — you don't have to define them (but you can). An LLM reads every video on its own, writes an English summary and tags *for that video*, designs a set of non-overlapping folders from all the summaries, and files each video (e.g. `Volleyball`, `Cooking & Meal Prep`, `AI & Machine Learning`).

> **v0.2 — why the classifier was rewritten.** v0.1 embedded raw captions, ran k-means and asked an LLM to name each cluster. On a real library of 283 saved videos the results were poor: Spanish and English posts clustered by *language*, unrelated videos were forced into the nearest cluster, and every video inherited its cluster's tags. In a 45-video manual spot check, 11 were clearly misfiled (home workouts under *Volleyball*, a fruit ad under *Cuban Literature*, a health wristband under *Investing*). The LLM-first pipeline misfiled none of the same 45 and gives every video its own tags (280 distinct tag lists instead of 11). The old method is still available offline with `--no-llm`.

---

## Requirements

- macOS, Linux or Windows with **Python 3.10+**
- **ffmpeg** (`brew install ffmpeg`)
- An Instagram account (the one whose saved videos you want)
- An OpenAI-compatible API key (OpenAI, OpenRouter, Groq, Together, local vLLM, …)

> **Heads-up:** Instagram has no official API for saved posts. Collection uses the private mobile API (`feed/saved/posts/` via `instagrapi`). It works, but automated access is against Instagram's terms and can trigger rate limits or a temporary checkpoint. Keep the request rate low, reuse the saved session, and don't run it on an account you can't afford to lose access to.
>
> `instaloader` is still supported as a fallback with `--engine instaloader`, but its web endpoint is heavily throttled and often returns `401 Please wait a few minutes`.

---

## Quick start

```bash
# 1. install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-transcribe.txt   # optional, for audio transcripts

# 2. configure
cp .env.example .env
#    edit .env → OPENAI_API_KEY=... and IGVAULT_LOGIN=your_username

# 3. test on 20 videos first
python -m igvault run --limit 20

# 4. full run
python -m igvault run
```

The first run asks for your Instagram password (and 2FA code if enabled). It is **not stored**; only a session file is written to `data/work/igapi_session.json` (permissions `600`).

If the API login asks for a challenge, open Instagram in the app or a browser, approve the login attempt, then rerun. To use the web engine instead:

```bash
python -m igvault collect --engine instaloader
```

---

## Commands

| Command | What it does |
| --- | --- |
| `python -m igvault run` | Full pipeline: collect → index → transcribe → classify → organize |
| `python -m igvault collect` | Only download saved posts (resumable) |
| `python -m igvault index` | Rebuild `records.jsonl` from downloaded metadata |
| `python -m igvault transcribe` | Transcribe audio (skips existing transcripts) |
| `python -m igvault classify` | Re-cluster and rename topics |
| `python -m igvault organize` | Copy videos into topic folders + rebuild indexes |
| `python -m igvault status` | Show counts for every stage |

Useful flags:

```bash
python -m igvault run --limit 20            # test on 20 videos
python -m igvault collect --engine api      # default: private mobile API (recommended)
python -m igvault collect --engine instaloader  # fallback: web GraphQL endpoint
python -m igvault collect --fast-update     # stop at the first already-downloaded post
python -m igvault transcribe --model small  # tiny | base | small | medium | large-v3
python -m igvault classify --clusters 8     # ask for exactly 8 topics
python -m igvault organize --dry-run        # preview folders, write nothing
python -m igvault organize --prune          # remove stale copies after re-classification
python -m igvault organize --mode move      # move instead of copy
```

Re-syncing later is incremental: `collect` skips existing files and `transcribe` skips existing transcripts.

### Keeping it updated

```bash
python -m igvault collect --fast-update   # fetch newly saved videos only
python -m igvault transcribe              # transcribe only the new ones
python -m igvault classify                # re-cluster the whole library
python -m igvault organize --prune        # re-copy, remove stale topic copies
```

Or just `python -m igvault run --prune` — every stage skips work that is already done.

---

## Configuration

Everything lives in [`config.yaml`](config.yaml). The most useful knobs:

```yaml
collect:
  videos_only: true      # skip photo posts
  limit: 0               # 0 = all

transcribe:
  enabled: auto          # auto | true | false
  model: small           # bigger = more accurate, slower
  language: auto         # or "en", "es", …

llm:
  base_url: https://api.openai.com/v1
  chat_model: gpt-5.6-luna      # describes and files each video
  taxonomy_model: gpt-5.6-luna  # designs the folders (2 calls per run)
  reasoning_effort: low         # GPT-5 / o-series only
  # works with any OpenAI-compatible provider and model, e.g. gpt-4.1-mini

classify:
  use_frames: true       # send 3 frames to the LLM when the caption says little (needs ffmpeg)
  topics: []             # optional fixed folder list: [{name: ..., description: ...}]
  clusters: auto         # auto, or an exact number of topics
  max_clusters: 12
  min_cluster_size: 3
  naming_language: English

organize:
  mode: copy             # copy | move | symlink
```

Environment overrides: `IGVAULT_LOGIN`, `IGVAULT_LLM_BASE_URL`, `IGVAULT_LLM_MODEL`, `IGVAULT_EMBED_MODEL`, `IGVAULT_LIBRARY`, `IGVAULT_RAW`, `IGVAULT_WORK`.

### Which provider?

| Provider | `base_url` | `chat_model` | `embedding_model` |
| --- | --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | `text-embedding-3-small` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` | `openai/text-embedding-3-small` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` | *(leave empty → TF-IDF fallback)* |
| Ollama | `http://localhost:11434/v1` | `llama3.2` | *(leave empty → TF-IDF fallback)* |

Any chat model with JSON output works; vision is used for low-text videos when available. Without an API key (or with `--no-llm`) the pipeline falls back to the v0.1 method: embeddings or TF-IDF + k-means + hashtag names.

---

## What you get

```
library/
├── README.md                     ← topic table with counts
├── _index.csv                    ← open in Excel / Numbers / Sheets
├── _index.json                   ← machine-readable index
├── Home Cooking/
│   ├── _about.md
│   ├── 2025-03-14_CxYzAbc123_johncook.mp4
│   └── 2025-03-14_CxYzAbc123_johncook.md   ← caption + transcript + source link
├── AI Tools/
│   └── …
└── Unsorted/
    └── …
```

`_index.csv` columns: `shortcode, topic, posted_at, author, duration_seconds, url, caption, hashtags, tags, confidence, video_file, info_file, source_file`.

Every video also gets a small Markdown sidecar with its caption, transcript, tags and original Instagram link — useful for full-text search with Spotlight, Alfred, Obsidian, etc.

---

## How topics are chosen

1. **Describe.** Each video is sent to the LLM in batches of 12: author handle, caption, hashtags and transcript (if any). Videos whose caption has fewer than ~12 real words and no transcript are sent individually with **3 frames** extracted by ffmpeg. The model returns an English `summary`, a broad `subject`, 3–6 specific `tags`, and an `unclear` flag. Results are cached per video (keyed on the input text and model), so re-runs only describe new videos.
2. **Design folders.** All summaries plus a subject-frequency table go to the LLM in one call. It proposes up to `max_clusters` folders, told to merge related subjects, avoid overlap, never group by language or by "is an ad", and keep `Other` under ~10 %.
3. **File.** Videos are assigned in batches of 30 to exactly one folder (or `Other`) with a confidence score. Anything the model skips is retried.
4. **Refine.** If more than ~8 % of videos land in `Other`, the LLM proposes extra folders for just those videos and they are filed again.
5. Videos the model marks `unclear` go to `Unsorted`.

Want control? Set `classify.topics` in `config.yaml` to a fixed folder list, or pass `--clusters N`, then re-run `classify` + `organize --prune`.

## Privacy & cost

- **Local by default**: videos, transcripts and the library never leave your machine.
- **What is sent to the LLM**: captions, hashtags, author handles, transcripts and, for low-text videos, 3 low-resolution frames. If that is a concern, point `llm.base_url` at a local server (Ollama, vLLM).
- **Cost**: one describe call per 12 videos, one per low-text video, two taxonomy calls and one assign call per 30 videos — roughly 150 requests for a 300-video library, typically well under $1. Re-runs reuse the description cache.
- **Instagram password**: asked interactively, never written to disk. Only the session file (`data/work/igapi_session.json`) is saved.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `challenge_required` | Open Instagram in the app or a browser, approve the login attempt, then rerun `collect`. |
| `LoginRequiredException` | The session expired. Delete `data/work/igapi_session.json` and re-login. |
| Web engine returns `401 Unauthorized` | Use the default `--engine api`; the Instaloader web endpoint is heavily throttled. |
| Download stops with `429` / `Please wait a few minutes` | Instagram rate limit. Wait 1–6 hours, then resume with `collect --fast-update`. |
| `ffmpeg not found` | `brew install ffmpeg` |
| `faster-whisper` install fails | Skip it (`transcribe.enabled: false`); classification still uses captions and hashtags. |
| Topics look too broad / too narrow | Raise or lower `max_clusters`, force `--clusters N`, or fix the list with `classify.topics`. |
| `Unsupported parameter: max_tokens` | Reasoning models (GPT-5, o-series) are detected by name; check `llm.chat_model` is spelled as the provider lists it. |
| LLM requests fail | Check `OPENAI_API_KEY` and `llm.base_url`; with no LLM the pipeline falls back to k-means + hashtag names. |
| Old topic folders remain after re-classifying | `python -m igvault organize --prune` |

---

## Project layout

```
igvault/
├── cli.py          command-line interface
├── config.py       YAML + env configuration
├── collect.py      Instaloader web fallback (--engine instaloader)
├── collect_api.py  private mobile API collector (default)
├── metadata.py     JSON sidecars → records.jsonl
├── transcribe.py   ffmpeg + faster-whisper
├── llm.py          OpenAI-compatible chat (incl. reasoning models) + embeddings
├── classify.py     classify command; offline k-means fallback
├── classify_llm.py LLM describe → design folders → assign
├── organize.py     copy/move/symlink + CSV/JSON/Markdown index
└── utils.py        helpers
config.yaml
requirements.txt
requirements-transcribe.txt
```

Data directories are git-ignored: `data/` (raw + work) and `library/` (final output). No videos, captions, transcripts, sessions or keys are part of this repository.

## License

MIT
