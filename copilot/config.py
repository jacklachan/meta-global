"""Runtime configuration for the copilot.

Everything is read from environment variables so the same code runs locally,
in Docker and on a hosted Space. See ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_API_BASE_URL = "https://api.meta.ai/v1"

# The three jobs an investigation needs, cheapest first. Each runs on the
# model configured for it; by default all three use Muse Spark on Meta Model API.
ROLE_TRIAGE = "triage"    # many small calls: pick the next piece of evidence
ROLE_REASON = "reason"    # one or two big calls: root cause + causal chain
ROLE_WRITER = "writer"    # one call: turn the diagnosis into a postmortem
ROLES = (ROLE_TRIAGE, ROLE_REASON, ROLE_WRITER)

# Standard, not -contributor: real incident bundles hold private logs, and
# contributor traffic is used to improve Meta's products.
DEFAULT_MODELS = {
    ROLE_TRIAGE: "muse-spark-1.3",
    ROLE_REASON: "muse-spark-1.3",
    ROLE_WRITER: "muse-spark-1.3",
}

# USD per 1M tokens: (input, output). Keys are lower-cased model IDs.
# From dev.meta.ai/models/muse-spark as of 2026-10-26.
PRICES_PER_MTOK = {
    "muse-spark-1.3": (1.25, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.20),
}

_ENV_BY_ROLE = {
    ROLE_TRIAGE: "COPILOT_MODEL_TRIAGE",
    ROLE_REASON: "COPILOT_MODEL_REASON",
    ROLE_WRITER: "COPILOT_MODEL_WRITER",
}


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    models: dict[str, str]
    tavily_api_key: str | None
    # Per-role server overrides, e.g. triage on a self-hosted fine-tuned model.
    role_base_urls: dict[str, str] = field(default_factory=dict)
    role_api_keys: dict[str, str] = field(default_factory=dict)

    def model_for(self, role: str) -> str:
        return self.models[role]

    @property
    def ready(self) -> bool:
        """Every role has a key, whether for Model API or its own server."""
        return all(self.endpoint_for(role)[1] for role in ROLES)

    def endpoint_for(self, role: str) -> tuple[str, str | None]:
        """(base URL, API key) for ``role``: its override, else Model API."""
        return (self.role_base_urls.get(role, self.base_url),
                self.role_api_keys.get(role, self.api_key))


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a local .env file. Real environment
    variables win, so a deployment's secrets are never overridden."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def load_settings() -> Settings:
    models = {
        role: os.environ.get(_ENV_BY_ROLE[role]) or DEFAULT_MODELS[role]
        for role in ROLES
    }
    urls = {role: os.environ[f"COPILOT_BASE_URL_{role.upper()}"].rstrip("/")
            for role in ROLES if os.environ.get(f"COPILOT_BASE_URL_{role.upper()}")}
    keys = {role: os.environ[f"COPILOT_API_KEY_{role.upper()}"]
            for role in ROLES if os.environ.get(f"COPILOT_API_KEY_{role.upper()}")}
    return Settings(
        api_key=os.environ.get("MODEL_API_KEY") or None,
        base_url=(os.environ.get("MODEL_API_BASE_URL") or MODEL_API_BASE_URL).rstrip("/"),
        models=models,
        tavily_api_key=os.environ.get("TAVILY_API_KEY") or None,
        role_base_urls=urls,
        role_api_keys=keys,
    )


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated cost of one call. Unknown models cost 0 rather than guessing."""
    price = PRICES_PER_MTOK.get(model.lower())
    if price is None:
        return 0.0
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
