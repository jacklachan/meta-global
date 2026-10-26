"""Triage training data: same prompts as production, no answer leaked up front."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from copilot import incidents
from copilot.config import ROLE_TRIAGE
from copilot.investigator import Investigator
from copilot.llm import ChatResult, UsageMeter
from copilot.workspace import Workspace

_spec = importlib.util.spec_from_file_location(
    "make_triage_data", Path(__file__).resolve().parent.parent / "scripts" / "make_triage_data.py")
make = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make)


class Replay:
    """Answers triage with the demo's actions and records the full prompts."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.seen = []
        self.meter = UsageMeter()

    async def chat(self, role, messages, max_tokens=1024, temperature=0.2):
        if role == ROLE_TRIAGE:
            self.seen.append(messages)
            text = self.actions.pop(0)
        else:
            text = json.dumps({"root_cause_ids": [], "chain": []})
        result = ChatResult(text=text, role=role, model="replay", input_tokens=0,
                            output_tokens=0, cost_usd=0.0, latency_s=0.0)
        self.meter.add(result)
        return result


@pytest.mark.parametrize("difficulty", incidents.DIFFICULTIES)
def test_examples_use_the_investigators_exact_prompts(difficulty):
    examples = make.examples_for(1000, difficulty)
    model = Replay(e["messages"][-1]["content"] for e in examples)
    inv = Investigator(Workspace(incidents.from_seed(1000, difficulty)), model, write_report=False)

    async def run():
        return [e async for e in inv.run()]
    asyncio.run(run())

    assert [e["messages"][:-1] for e in examples] == model.seen


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
@pytest.mark.parametrize("difficulty", incidents.DIFFICULTIES)
def test_demo_reads_logs_first_and_opens_every_blamed_change(seed, difficulty):
    scenario = incidents.from_seed(seed, difficulty)
    actions = [json.loads(e["messages"][-1]["content"]) for e in make.examples_for(seed, difficulty)]

    assert actions[0]["tool"] == "search_logs"
    assert actions[-1]["tool"] == "done"
    opened = {v for a in actions for v in a.get("args", {}).values()}
    assert set(scenario["ground_truth"]["cause"].split("+")) <= opened


def test_benchmark_seeds_are_refused(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["x", "--seeds", "5", "--out", str(tmp_path / "x.jsonl")])
    with pytest.raises(SystemExit):
        make.main()
