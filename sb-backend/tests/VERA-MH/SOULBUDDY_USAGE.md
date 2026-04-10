# Using VERA-MH In This Repo

`tests/VERA-MH` is a standalone Python project inside the backend repo.

Do not use the backend virtualenv for it. Use the VERA-MH env from inside `tests/VERA-MH`.

- VERA-MH has its own `pyproject.toml`
- VERA-MH has its own `.venv`
- VERA-MH requires Python `>=3.11`
- SoulBuddy already exposes a VERA-compatible endpoint at `/api/v1/chat/vera`

## The important rule

Run VERA commands from `tests/VERA-MH`, not from the repo root.

```bash
cd tests/VERA-MH
source .venv/bin/activate
```

If the env ever needs to be rebuilt:

```bash
cd tests/VERA-MH
uv sync
source .venv/bin/activate
```

## Environment variables

Your local `.env` should point the endpoint provider at SoulBuddy's wrapper:

```env
ENDPOINT_URL=http://127.0.0.1:8000/api/v1/chat/vera
ENDPOINT_API_KEY=anything

```

Notes:

- `ENDPOINT_START_URL` should be commented out unless you add a separate start endpoint
- `ENDPOINT_API_KEY` can be left blank like so: `ENDPOINT_API_KEY=`
- Recommended models: Google `gemini-2.5-pro` for persona generation and Anthropic `claude-sonnet-4-6` for judging

## What you can do without API keys

Run VERA's own local tests:

```bash
cd tests/VERA-MH
source .venv/bin/activate
pytest -m "not live"
```

You can also inspect or rescore existing outputs already in:

- `tests/VERA-MH/conversations/`
- `tests/VERA-MH/evaluations/`

## Quickest real evaluation of SoulBuddy

1. Start the Soulbuddy Backend from the repo root in one terminal:

```bash
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

2. In a second terminal, run VERA from its own folder:

```bash
cd tests/VERA-MH
source .venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib
python run_pipeline.py \
  --user-agent gemini-2.5-pro \
  --provider-agent endpoint-app \
  --runs 2 \
  --turns 10 \
  --judge-model claude-sonnet-4-6 \
  --max-personas 3
```

What that does:

- `--user-agent gemini-2.5-pro` uses Google to simulate the persona/user
- `--provider-agent endpoint-app` sends provider turns to SoulBuddy at `/api/v1/chat/vera`
- `--judge-model claude-sonnet-4-6` uses Anthropic to evaluate the finished conversations

## Minimal manual flow

If you want each step separately instead of the full pipeline:

```bash
cd tests/VERA-MH
source .venv/bin/activate

python generate.py \
  -u gemini-2.5-pro \
  -p endpoint-app \
  -t 6 \
  -r 1 \
  -mp 3

python judge.py \
  -f conversations/<generated-folder> \
  -j claude-sonnet-4-6

python -m judge.score \
  -r evaluations/<evaluation-folder>/results.csv
```

## Where Results can be Found

- `Conversations` contains the generated raw chat transcripts for each persona/run in `conversations/<run-folder>/`
- `Evaluations` contains the judge outputs in `evaluations/<eval-folder>/`, including per-conversation rubric TSVs and a `results.csv` file that `judge.score` reads

## Model name rules

- Any provider model name containing `endpoint` uses VERA's endpoint adapter
- `endpoint-app` is the right choice for SoulBuddy because VERA will send `"model": "app"` to `/api/v1/chat/vera`
- Judge models cannot be `endpoint-*`; use `gpt-*`, `claude-*`, `gemini-*`, or `azure-*`
- Current Anthropic API IDs are versioned, for example `claude-sonnet-4-6` and `claude-opus-4-6`; `claude-opus-4` is not a valid model ID

## Common confusion points

- If you run `judge.py` or `run_pipeline.py` from the repo root, relative paths like `data/rubric.tsv` can break or behave oddly
- If you do not set an API key for the persona/judge model, generation or judging will fail
- If the backend server is not running on port `8000`, endpoint-based provider runs will fail
- The first `run_pipeline.py` run may spend time building the matplotlib font cache
- If Anthropic returns `404 not_found_error model: claude-opus-4`, the model name is stale; use `claude-sonnet-4-6` or `claude-opus-4-6`
