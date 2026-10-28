"""Export one generated incident, solved by the oracle, for the home-page story.

The story plays from a static file so the landing page needs no key, costs
nothing and adds nothing to the investigation history.

    python scripts/make_story.py --seed 3 --difficulty medium
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from copilot import incidents
from copilot.oracle import Oracle
from copilot.workspace import Workspace

MAX_LINES = 18    # per exhibit; the story shows excerpts, the app shows everything


def story(seed: int, difficulty: str) -> dict:
    scenario = incidents.from_seed(seed, difficulty)
    brief = Workspace(scenario).brief()
    oracle = Oracle(scenario)
    evidence = oracle.investigate()
    changes = sorted(
        [{"id": c["hash"], "kind": "commit", "service": c["service"], "when": c["timestamp"], "what": c["message"]}
         for c in brief["commits"]]
        + [{"id": c["config_id"], "kind": "config", "service": c["service"], "when": c["timestamp"],
            "what": c["description"]} for c in brief["config_changes"]]
        + [{"id": e["event_id"], "kind": "infra", "service": "infrastructure", "when": e["timestamp"],
            "what": e["description"]} for e in brief["infra_events"]],
        key=lambda c: c["when"])
    truth = scenario["ground_truth"]
    out = {
        "incident_id": brief["incident_id"],
        "description": brief["description"],
        "window": brief["incident_window"],
        "services": [{"name": s["name"], "status": s.get("status"), "deps": s.get("dependencies", []),
                      "error_rate": s.get("error_rate_during_incident")} for s in brief["services"]],
        "onset": brief.get("error_onset", []),
        "changes": changes,
        "cause": truth["cause"].split("+"),
        "chain": [{"service": h["service"], "effect": h["effect"]} for h in truth["chain"]],
        "evidence": [{"id": e.id, "tool": e.tool, "args": e.args,
                      "result": "\n".join(e.result.splitlines()[:MAX_LINES])} for e in evidence],
        "score": oracle.grade["score"],
    }
    ids = {c["id"] for c in changes}
    assert set(out["cause"]) <= ids, "the cause must be one of the changes on the tape"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--out", default=str(ROOT / "static" / "copilot" / "story.json"))
    a = p.parse_args()
    data = story(a.seed, a.difficulty)
    Path(a.out).write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{a.out}: {len(data['changes'])} changes, {len(data['evidence'])} exhibits, cause {data['cause']}")


if __name__ == "__main__":
    main()
