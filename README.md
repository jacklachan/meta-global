# Hindsight

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**The postmortem, with its evidence.** Hindsight investigates a production incident from its telemetry, names what started the outage, and writes the postmortem. Every claim in the document carries a tag back to the log line, diff or trace it came from.

Built for the **Meta Global AI Developer Hackathon** (Push the Frontier: software). It runs on **Muse Spark** through Meta Model API, and fine-tunes **Muse Glimmer**, Meta's open-weights model, for the many small decisions an investigation makes.

> **Status: in development.** The pipeline, API, MCP server and UI work end to end and are covered by tests. No investigation has run on a real Meta model yet (the Model API key is pending), so the results table below has empty rows. They will be filled from committed benchmark files, not before.

## What it does

1. **Takes an incident.** A sample incident, a generated one, or your own. `python -m copilot bundle` builds one from your services' git history and log files; the [bundle format](static/copilot/bundle-format.html) also takes traces, config changes and infrastructure events.
2. **Investigates it blind.** The agent has read-only evidence tools and nothing else. Each lookup becomes a numbered exhibit: E1, E2, E3.
3. **Names the cause.** It commits to a root cause and the chain of failures that followed, citing exhibits for each hop, and says what it ruled out and what it could not settle.
4. **Checks the wider world.** Optionally, it searches for published guidance on that failure mode, without sending anything incident-specific to the search engine.
5. **Writes the postmortem.** Summary, impact, root cause, how it spread, timeline, action items, open questions, evidence appendix and what the investigation cost. Download it as Markdown.

For incidents with a known answer, a deterministic grader scores the investigation, so accuracy is measured, not asserted.

## How it uses Meta's models

An investigation is three jobs, and each role can run on its own model and its own server ([copilot/config.py](copilot/config.py)).

| Stage | Job | Model |
|---|---|---|
| Triage | Choose the next lookup. 5–12 short calls per incident. | Muse Glimmer 30B, fine-tuned for this job, served with vLLM on Modal. Muse Spark until then. |
| Diagnosis | Read the whole evidence ledger, name the cause and chain. One or two calls. | `muse-spark-1.3` on Meta Model API |
| Postmortem | Write the summary, impact and action items. One call. | `muse-spark-1.3` on Meta Model API |

Triage is where most calls go, so it's the job worth making small and cheap. Diagnosis is where reasoning quality decides the outcome, so it stays on Muse Spark. The UI shows which model made each move and the running cost per stage.

### Fine-tuning Glimmer for triage

1. **Training data** ([scripts/make_triage_data.py](scripts/make_triage_data.py)). The oracle (below) investigates 600 generated incidents that are not in the benchmark. Its evidence is replayed in the order an engineer without the answer would gather it: logs along the failure path, a trace, then every recent change on the first service to fail, decoys included. Each step becomes one example: the exact triage prompt and the next action. 4,151 examples. The prompt comes from the same function the live investigator uses, and a test checks they match.
2. **Training** ([training/sft.py](training/sft.py), [training/modal_train.py](training/modal_train.py)). QLoRA on `meta-models/Muse-Glimmer-30B` with TRL, one H100 on Modal. Loss on the action only; LoRA on the text decoder only.
3. **Serving** ([training/modal_serve.py](training/modal_serve.py)). One vLLM server exposes both `glimmer-base` and `glimmer-triage` (base + adapter), so the untrained and fine-tuned rows run on identical hardware.

## Results

Generated incidents, scored by [copilot/evaluate.py](copilot/evaluate.py). Score is 0–1; cause is the share with the right root cause.

| Investigator | Incidents | Cause correct | Score | Lookups | Cost / incident |
|---|---|---|---|---|---|
| Random change | 600 | 5.7% | 0.03 | 0 | $0 |
| Most recent change | 600 | 25.0% | 0.14 | 0 | $0 |
| Most recent change on the reddest service | 600 | 42.8% | 0.24 | 0 | $0 |
| Muse Spark (all roles) | – | *pending* | | | |
| Glimmer base (triage) + Muse Spark | – | *pending* | | | |
| **Glimmer fine-tuned (triage) + Muse Spark** | – | *pending* | | | |
| Oracle (given the answer) | 600 | 100% | 1.00 | 5.2 | $0 |

Baseline and oracle rows: seeds 0–199 at easy, medium and hard ([benchmarks/](benchmarks/)). The baselines never look at evidence, which is why grounding keeps their scores low even when they name the right change. Training uses seeds 1000+; the data script refuses benchmark seeds.

## Design choices that matter

- **No oracle.** The agent's workspace offers read-only evidence tools and nothing else ([copilot/workspace.py](copilot/workspace.py)). There is no way to ask whether a guess is right, and nothing it returns carries the answer or the labels that mark which facts matter.
- **The model cannot put facts in the document.** Root cause, chain, timeline and citations are assembled in code from a validated diagnosis. IDs, services and citations the model invents are dropped. The writer model contributes prose only ([copilot/report.py](copilot/report.py)).
- **Nothing internal reaches web search.** Search queries are written to be generic, and any query containing a service name, change ID, trace ID or email address from the incident is dropped before it is sent.
- **Private logs stay off the cheap tier.** Muse Spark's `-contributor` mode costs ~12x less, but Meta uses that traffic to improve its products. The default is the standard model; contributor is for the synthetic benchmark only.
- **Scored by rule, not by a model, and guessing does not pay.** Five scores: root cause, failure path (the right services in order), failure modes (the right label at each hop), grounding, and efficiency. Grounding checks the work: did it open the change it blames, and do the exhibits it cites actually bear on the incident? Naming the right commit without investigating scores 0.55; the same answer properly investigated scores 1.00.
- **The test incidents do not give the answer away.** Generated incidents ([data/incident_generator.py](data/incident_generator.py)) have a symptom-only brief, a realistic diff on every commit, harmless changes that land closer to the outage than the culprit, and alarming-sounding changes that are innocent. In most of them the most recent change is not the cause.
- **A real ceiling, and proof the incidents are solvable.** The oracle ([copilot/oracle.py](copilot/oracle.py)) is given the answer but still has to go through the same evidence tools, open every change it blames, and find an exhibit for each step of the failure. It scores 1.00 on every generated incident in about five lookups. A test fails if any generated incident cannot support its own answer.
- **It says when it does not know.** An investigation that cannot name a cause produces a document that says so and lists the open questions.

## Run it

Requires Python 3.11 or later.

```bash
git clone https://github.com/jacklachan/meta-global.git
cd meta-global
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt    # macOS/Linux: .venv/bin/python
cp .env.example .env                                        # then add your keys
```

Put a Meta Model API key in `.env` as `MODEL_API_KEY`. `TAVILY_API_KEY` is optional and enables the research step.

```bash
# Check the configured models exist on your account
python -m copilot models

# Web UI at http://localhost:7860
uvicorn app:app --port 7860

# One investigation from the command line
python -m copilot investigate --seed 42 --difficulty medium --out postmortem.md

# Your own incident: build a bundle from git history and log files, then investigate it
python -m copilot bundle --start 2026-10-01T10:00:00Z --end 2026-10-01T10:20:00Z \
  --repo api=../api --repo web=../web --logs api=logs/api.log \
  --services services.json --out my_incident.json
python -m copilot investigate --file my_incident.json --out postmortem.md

# Score the investigator on 20 generated incidents
python -m copilot bench --seeds 0-19 --difficulty medium --out benchmarks/spark-medium.json
```

Without a key the app still shows real investigations: every sample incident has a **See the reference answer** button, which runs the oracle with no model and no spend. `python scripts/dev_ui_server.py` runs the UI with a scripted stand-in for the models; it demonstrates the interface and nothing about model quality.

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

### Fine-tune and serve Glimmer

These spend Modal GPU credits. Install the tools from [training/requirements.txt](training/requirements.txt) and run `modal setup` first.

```bash
python scripts/make_triage_data.py --seeds 1000-1199        # free, ~5 s
modal run training/modal_train.py                           # smoke run: 64 examples, 20 steps
modal run training/modal_train.py --full --run glimmer-v1   # the real run
modal secret create hindsight-vllm VLLM_API_KEY=<long random string>
modal deploy training/modal_serve.py
```

Then point the triage role at it in `.env` (see [.env.example](.env.example)): `COPILOT_BASE_URL_TRIAGE`, `COPILOT_API_KEY_TRIAGE` and `COPILOT_MODEL_TRIAGE=glimmer-triage`.

To check the training code for free, run [training/sft.py](training/sft.py) on CPU against TRL's tiny Glimmer (command in its docstring).

## Use it from a coding agent (MCP)

Hindsight is also an MCP server, so a coding agent such as Muse Code can investigate an incident without leaving the repository it is working in.

```json
{
  "mcpServers": {
    "hindsight": {
      "command": "/path/to/meta-global/.venv/bin/python",
      "args": ["-m", "copilot", "mcp"],
      "cwd": "/path/to/meta-global"
    }
  }
}
```

| Tool | What it does |
|---|---|
| `build_incident_bundle` | Collects the day before an incident from local git repositories and log files into a bundle file. |
| `investigate_bundle_file` | Investigates that file and returns the root cause, how it spread, open questions and the postmortem. |
| `investigate_incident` | The same, with the bundle passed inline. |
| `investigate_sample`, `list_sample_incidents` | Try it on incidents with a known answer; the result is graded. |

The web app serves the same thing over Streamable HTTP at `/api/copilot/mcp`, under the same spending limits as the UI (`python -m copilot mcp --http` runs it on its own). The two tools that read local paths are only offered over stdio, where the caller is the machine's own user.

## Layout

```
copilot/            the product
  llm.py            chat client: per-role routing and servers, retries, cost metering
  workspace.py      blind evidence tools over the environment
  investigator.py   triage, diagnosis, validation, grading
  research.py       Tavily search with private terms kept out of queries
  report.py         the postmortem document
  bundle.py         build an incident bundle from git repos and log files
  mcp_server.py     the same investigator as MCP tools
  evaluate.py       deterministic scoring, including grounding
  oracle.py         reference investigator: the ceiling, and a solvability check
  taxonomy.py       failure-mode vocabulary
  bench.py          benchmark runner
training/           Glimmer fine-tuning (TRL) and serving (vLLM), both on Modal
scripts/make_triage_data.py  triage training data from oracle runs
web/copilot_api.py  HTTP API and live event stream
static/copilot/     the UI
data/incident_generator.py   incidents with coherent telemetry and decoys
web/lab.py, engine/ the original PostmortemEnv, attached at /lab (optional)
```

## What existed before this hackathon

This is disclosed in full so judges can see what was built here.

- **PostmortemEnv** (April 2026, Meta PyTorch OpenEnv Hackathon): the reinforcement-learning environment this started from. Its README is kept at [docs/POSTMORTEMENV.md](docs/POSTMORTEMENV.md); its history is the start of this repository's commit log. Carried over: the five hand-written sample incidents and the original environment, now an optional add-on at `/lab` that the product does not import.
- **Hindsight on Nemotron** (October 2026, built for the Nebius x NVIDIA hackathon, never submitted): the product layer, meaning the evidence store, evaluator, failure-mode vocabulary, incident generator, investigator, research step, postmortem writer, bundle builder, MCP server, benchmark runner, web API and UI. It targeted Nemotron models on Nebius and was never run against a real model.
Both were team work (L Mohit Jain, Tanush Deepak, Utkarsh Singh Yadav).

- **Built for this hackathon, solo** (from October 26, 2026; every commit after `b791dc5`): the move to Meta Model API and Muse Spark, per-role model servers, the triage training data and its prompt-match test, the Glimmer fine-tuning and serving on Modal, and every real-model result in the table above.

## Known limits

- Generated incidents cover five failure modes (connection pool exhaustion, memory leak, out-of-memory crash loop, config change, failover bug triggered by a network event). They are useful for measuring the agent and are simpler than real outages.
- The triage training data comes from an oracle that knows the answer. Its lookups are reordered to hide that, but the demonstrations still stop reading changes once the culprit is open, which a real investigator can't know to do. Mixing in successful Muse Spark investigations is the next step if this shows up in the results.
- Failure-mode scoring needs exact labels, so the agent labels each hop from a fixed vocabulary ([copilot/taxonomy.py](copilot/taxonomy.py)). The five hand-written incidents use free-text labels, so that one score understates the agent there; root cause and failure path are the figures to read.
- The agent can only name a cause that is present in the bundle. If the change that broke production was never exported, it will not be found.

## Team

L Mohit Jain (solo entry, student track).

PostmortemEnv and the Nemotron version of Hindsight were built with Tanush Deepak and Utkarsh Singh Yadav.

## License

[MIT](LICENSE)
