# Hindsight on Meta models: plan to Nov 4

**Track:** Indie / student. **Theme:** Push the Frontier (software).
**Pitch:** Hindsight investigates a production outage and writes a postmortem where every claim cites its evidence. A small open-weights Meta model, fine-tuned on the investigation's step-by-step choices, does the many cheap lookup decisions; Muse Spark does the one hard reasoning call. A deterministic benchmark proves how well each setup works.

**The result that wins it:** one table, same 60 held-out incidents:

| Triage model | Diagnosis model | Score | Cause correct | Cost / incident |
|---|---|---|---|---|
| random / heuristic baselines (already exist) | – | | | $0 |
| small open model, untrained | Muse Spark | | | |
| **small open model, fine-tuned** | Muse Spark | | | |
| Muse Spark | Muse Spark | | | |
| oracle (ceiling, already exists) | – | 1.00 | | $0 |

If the fine-tuned row closes most of the gap to Muse Spark at a fraction of the cost, that's the story. If it doesn't, ship the table anyway: honest numbers still beat most entries.

Source repo: `~/Downloads/nvidia and amazon hackathon/postmortem` (remote `jacklachan/Nebius-x-NVIDIA`).

---

## 1. What to reuse (almost everything)

| Keep as is | Why |
|---|---|
| `copilot/workspace.py` | Read-only evidence tools; nothing model-specific. |
| `copilot/evaluate.py`, `oracle.py`, `taxonomy.py` | The grader, the ceiling, the label set. These are the benchmark. |
| `data/incident_generator.py` | Unlimited incidents with decoys, so training and test sets can be cleanly separated. |
| `copilot/investigator.py` | Triage → diagnosis → report loop. Only prompts may change. |
| `copilot/report.py`, `bundle.py`, `mcp_server.py`, `bench.py` | Writer, real-incident import, MCP tools, benchmark runner. |
| `web/`, `static/copilot/`, `app.py` | UI and API. Update labels and the leaderboard only. |
| `tests/` | Keep green after every change. |

| Drop or demote | Why |
|---|---|
| Nemotron defaults and prices in `copilot/config.py` | Replaced by Meta models. |
| `train.py` (1540 lines, trains on the **old** PostmortemEnv format) | Wrong prompt format for the current investigator. Reuse only its TRL/LoRA setup (`run_sft`, around line 178) as a reference. |
| `training_data/oracle_demos.jsonl` (60 old-format demos) | Same reason. Regenerate. |
| `docs/DEPLOY.md` (Nebius Serverless) | Replace with HF Space or Modal deploy. |
| `research.py` (Tavily) | Keep it optional; it's not part of the story. |

**Repo:** start a new repo, e.g. `jacklachan/hindsight`, from this one **with git history kept**, and write an honest "What existed before" section (the current README already has the right shape). First, check the Official Rules on prior work and team size. PostmortemEnv (April, team of three) is prior work.

## 2. Swap to Muse Spark ✅ done Oct 26 (commit `65a329c`)

Meta Model API uses the same OpenAI chat-completions format as Nebius, so `llm.py` mostly stays the same.

- Base URL: `https://api.meta.ai/v1`. Key env var: `MODEL_API_KEY`.
- Models: `muse-spark-1.3`, $1.25 in / $4.25 out per 1M tokens. `muse-spark-1.3-contributor` costs $0.10 / $0.20, but Meta uses that traffic to improve its products: fine for the synthetic benchmark, never for real incident data.

Changes:
1. **`copilot/config.py`**
   - Default base URL → `https://api.meta.ai/v1`.
   - Read `MODEL_API_KEY` (keep `NEBIUS_API_KEY` as a fallback only if old tests need it).
   - `DEFAULT_MODELS`: all three roles → `muse-spark-1.3` to start.
   - `PRICES_PER_MTOK`: add Muse Spark prices from the Pricing page. Unknown models cost $0, which would make the cost column lie.
   - **Per-role endpoint:** add `COPILOT_BASE_URL_<ROLE>` / `COPILOT_API_KEY_<ROLE>`, so the triage role can point at your own vLLM server on Modal while diagnosis stays on Meta. This is the one real code change. `Settings.model_for` becomes `endpoint_for(role)` returning (url, key, model).
2. **`copilot/llm.py`:** use the per-role endpoint in `_request`. Rename `TokenFactoryClient` → `ChatClient` (5 call sites: `mcp_server.py:45`, `__main__.py:82,117,164`, `web/copilot_api.py:143`, plus 2 tests). Update error messages.
3. **`.env.example`, README, UI labels:** Nemotron → Muse Spark.
4. **Check:** `python -m copilot models` lists the Spark IDs, then `python -m copilot investigate --seed 42 --difficulty medium` runs end to end, then `pytest tests -q`.

Leave the three-role split in place. Once the fine-tuned model exists, it takes the triage role.

## 3. Fine-tune a small open-weights Meta model on Modal

**Model choice (decide Day 1, morning):**
The participant guide lists only **Muse Glimmer** and **Muse Spark Open** as open-weights models, so Llama is out.
- **Muse Glimmer 30B** (Apache 2.0, dense) is the target. Check that it's on Hugging Face, loads in `transformers`/TRL, and fits QLoRA on one 80 GB GPU.
- Not Muse Spark Open: 816B MoE, far beyond a $100 budget.
- **Fallback if Glimmer can't be fine-tuned in time:** serve Glimmer *untrained* as the cheap triage model (still a Meta open-weights model doing real work) and present the fine-tune as attempted, with the reason it stopped.

**What it learns:** the **triage** step only. Input: system prompt + incident brief + evidence ledger so far. Output: one JSON action (`{"thought", "tool", "args"}` or `done`). It runs ~5–12 times per incident, so it's where most calls and cost are. Diagnosis stays on Muse Spark.

**Training data:**
1. Pull the prompt construction out of `investigator.py` into one function, `triage_messages(brief, evidence)`. Inference and data generation must use the **same** function. Add a test that they match.
2. `scripts/make_triage_data.py`: for generated incidents on **seeds 1000–1999** (eval stays on 0–19), run `copilot/oracle.py`. After each oracle lookup, write one example: messages so far → the oracle's next action as JSON. ~5 lookups × 1000 incidents ≈ 5k examples, at no cost.
3. Optional, if the oracle-only model is weak: also run Muse Spark on seeds 2000–2199 and keep the triage steps from runs that score ≥ 0.9 (distillation). Costs API money, so check the price first.
4. Known weakness: the oracle never makes mistakes, so the model never sees how to recover from one. If it gets stuck repeating lookups, step 3 is the fix.

**Training (`modal_train.py`, one file):**
- Modal image: `torch`, `transformers`, `trl`, `peft`, `bitsandbytes`, `datasets`. Data and output go on a Modal Volume.
- TRL `SFTTrainer`, LoRA r=16, α=32, on attention and MLP projections. 2–3 epochs, max length ~4k tokens, loss on the assistant reply only.
- GPU: A100-80GB or H100 for Glimmer QLoRA; L40S/A10G is enough for 8B LoRA.
- Smoke run first: 50 examples, 20 steps, to prove the pipeline end to end before spending.
- Save the adapter to the Volume **and** push it to HF (`jacklachan/hindsight-triage-<model>`), so a crash can't lose it.
- Budget: $100 covers roughly one or two real training runs plus a few hours of serving. Check Modal's per-GPU prices before the full run.

**Serving (`modal_serve.py`):** vLLM with the base model + LoRA adapter, OpenAI-compatible, `@modal.web_server`. Point `COPILOT_BASE_URL_TRIAGE` at it. Scale to zero when idle. Serve the untrained base the same way for the comparison row (or use Baseten's $100 credits for the base model).

**Eval:** `python -m copilot bench --seeds 0-19 --difficulty {easy,medium,hard}` for each row of the table. Save each to `benchmarks/` and show them on the existing leaderboard.

## 4. Muse Code

Use Muse Code as your coding agent during the build (it's one of the listed tools). For the demo, register Hindsight's MCP server with it and show Muse Code investigating an incident from inside a repo with `build_incident_bundle` → `investigate_bundle_file`. The MCP server already exists, so this is setup, not code.

## 5. Day by day

Times are IST. College runs 9–5, so weekday evenings give about 4 hours, and Sat/Sun are full days.
**Deadline:** Nov 4, 9 PM ET = **Nov 5, 6:30 AM IST**. Aim to submit on the night of Nov 4 IST.

| Day | Goal | Done when |
|---|---|---|
| **Mon Oct 26** | Kickoff (9:30 PM IST). Read Official Rules: prior work, team size, deliverables. Get the Model API key and claim Modal/Baseten credits. Check Glimmer on HF. Create the new repo. | Rules read; keys in `.env`; model chosen. |
| **Tue Oct 27** | Section 2: swap to Muse Spark, add per-role endpoints, rename the client. | One real investigation on Spark; tests green. |
| **Wed Oct 28** | Muse Spark baseline bench on 60 incidents. Start `triage_messages()` refactor. | `benchmarks/spark-*.json` committed. **First real-model numbers.** |
| **Thu Oct 29** | `make_triage_data.py` + test that prompts match. Generate ~5k examples. | JSONL on disk; a quick manual look at 5 random examples. |
| **Fri Oct 30** | `modal_train.py` smoke run. `modal_serve.py` serving the untrained base. | Smoke adapter trains; base model answers through Hindsight. |
| **Sat Oct 31** | Full training run. Bench untrained-base and fine-tuned rows. | Two more rows in the table. |
| **Sun Nov 1** | Fix the biggest failure (bad JSON, loops, missed lookups): more data or distillation, retrain once. **Freeze numbers tonight.** | Final table. |
| **Mon Nov 2** | UI: leaderboard rows, "which model made this move", cost per incident. Muse Code + MCP demo set up. | Demo flows work on the deployed app. |
| **Tue Nov 3** | README rewrite (pitch, table, method, honest limits, prior-work section). Deploy to HF Space. Record the ≤2-min demo video. | Public link + video uploaded. |
| **Wed Nov 4** | Buffer. Fresh-clone test of the README steps. Submit. | Submitted before midnight IST. |

**Cut lines, if behind:**
- **Thu night:** no training data yet? Drop distillation for good, oracle-only data.
- **Sat night:** fine-tuning not working? Ship with Spark vs baselines vs oracle, and present fine-tuning as a measured negative result.
- **Mon night:** UI polish not done? Skip it. The README table and the video matter more.

**Keep DSA going:** at least 1 problem a day, even during the hackathon. A light week is fine; a stopped week isn't.

## Open questions to settle Day 1

- [ ] Official Rules: is building on PostmortemEnv / Hindsight allowed with disclosure? Solo or team? (Guide pasted so far stops at section 2.)
- [x] Muse Spark pricing: $1.25 / $4.25 per 1M (standard), $0.10 / $0.20 (contributor; data used by Meta). Rate limits still unknown.
- [x] Open weights allowed: Muse Glimmer and Muse Spark Open only.
- [x] Muse Glimmer: `meta-models/Muse-Glimmer-30B`, Apache 2.0, not gated, BF16 ~60 GB. Supported in transformers 5.20.2, TRL 1.16.0 and vLLM 0.32.0 (with LoRA). `training/sft.py` passed a CPU smoke run on TRL's tiny Glimmer (Oct 26). Model card: not intended for users under 18.
- [ ] Modal GPU prices; is $100 enough for Glimmer QLoRA?
- [ ] Is the Model API open to developers in India? (A third-party guide said US-only preview; the hackathon presumably grants access.)
- [ ] Oct exam dates (there's an RVCE Oct exam config on your desktop). If any fall this week, shift the schedule.
