"""Build supervised training data for the triage model from oracle runs.

Each example is one triage turn: the exact prompt the investigator sends
(``triage_messages``) and the JSON action a careful engineer would take next.

The oracle knows the answer, so its own lookup order gives the answer away:
it opens the guilty change first and never reads a decoy. Training on that
would teach a model to guess the culprit from the brief. Instead this replays
the oracle's evidence in the order an investigator without the answer would
take it: logs along the failure path from the origin, one end-to-end trace,
then every recent change on the origin service (newest first, decoys
included) and any other blamed change, then ``done``.

shortcut: the change reads still stop once the culprit is opened, which a
model can't know to do; distil from real Muse Spark runs if that shows.

Usage:
    python scripts/make_triage_data.py --seeds 1000-1199 --out training_data/triage_sft.jsonl

Keep training seeds away from the benchmark seeds (0-199, see benchmarks/).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import incidents
from copilot.bench import parse_seeds
from copilot.core import brief_text
from copilot.investigator import triage_messages
from copilot.oracle import Oracle
from copilot.workspace import Workspace

MAX_CHANGE_READS = 4      # recent changes read on the origin service

_CHANGE_TOOLS = {"get_commit": "commit_hash", "get_config": "config_id",
                 "get_infra_event": "event_id"}


def _thought(tool: str, args: dict[str, Any], services: dict[str, str]) -> str:
    if tool == "search_logs":
        return (f"Check {args['service']} {args.get('level', 'all')} logs "
                f"{args.get('time_window', '').replace('_', ' ')} to see how it failed.").replace("  ", " ")
    if tool == "get_trace":
        return "Follow one failing request end to end to confirm the failure path."
    if tool == "get_infra_event":
        return f"Check whether infrastructure event {args['event_id']} lines up with the onset."
    change = args.get("commit_hash") or args.get("config_id")
    return f"Read {change}, a change that landed on {services.get(change, 'a suspect service')} before the outage."


def plan_lookups(scenario: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The oracle's evidence, reordered as an investigator without the answer would gather it."""
    oracle = Oracle(scenario)
    gathered = [(e.tool, e.args) for e in oracle.investigate()]
    logs = [c for c in gathered if c[0] == "search_logs"]
    traces = [c for c in gathered if c[0] == "get_trace"]
    blamed = [c for c in gathered if c[0] in _CHANGE_TOOLS]

    origin = scenario["ground_truth"]["chain"][0]["service"]
    start = scenario["incident_window"]["start"]
    recent = [(c["timestamp"], "get_commit", {"commit_hash": c["hash"]})
              for c in scenario.get("commits", []) if c.get("service") == origin]
    recent += [(c["timestamp"], "get_config", {"config_id": c["config_id"]})
               for c in scenario.get("config_changes", []) if c.get("service") == origin]
    recent = [(tool, args) for ts, tool, args in sorted(recent, key=lambda r: r[0], reverse=True)
              if ts < start]

    changes: list[tuple[str, dict[str, Any]]] = []
    for call in recent:
        if len(changes) >= MAX_CHANGE_READS or all(b in changes for b in blamed):
            break
        changes.append(call)
    changes += [b for b in blamed if b not in changes]
    return logs + traces + changes


def examples_for(seed: int, difficulty: str) -> list[dict[str, Any]]:
    scenario = incidents.from_seed(seed, difficulty)
    services = {c["hash"]: c.get("service") for c in scenario.get("commits", [])}
    services |= {c["config_id"]: c.get("service") for c in scenario.get("config_changes", [])}
    ws = Workspace(scenario)
    brief = brief_text(ws.brief())
    out = []
    for tool, args in plan_lookups(scenario) + [("done", {})]:
        action: dict[str, Any] = {"thought": _thought(tool, args, services), "tool": tool, "args": args}
        if tool == "done":
            action = {"thought": "The cause and every hop of the failure now have evidence.", "tool": "done"}
        messages = triage_messages(brief, ws.evidence, ws.calls_left)
        messages.append({"role": "assistant", "content": json.dumps(action)})
        out.append({"seed": seed, "difficulty": difficulty, "messages": messages})
        if tool != "done":
            evidence = ws.call(tool, args)
            assert evidence.ok, f"seed {seed} {difficulty}: {tool}({args}) failed: {evidence.result}"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="1000-1199")
    parser.add_argument("--out", default="training_data/triage_sft.jsonl")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if set(seeds) & set(range(200)):
        parser.error("seeds 0-199 are the benchmark; train on others")
    count = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for seed in seeds:
            for difficulty in incidents.DIFFICULTIES:
                for example in examples_for(seed, difficulty):
                    f.write(json.dumps(example) + "\n")
                    count += 1
    print(f"{count} examples from {len(seeds) * len(incidents.DIFFICULTIES)} incidents -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
