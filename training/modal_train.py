"""QLoRA fine-tune of Muse Glimmer 30B for the triage role, on Modal.

Costs GPU credits. Smoke run first (64 examples, 20 steps), then the full run:

    python scripts/make_triage_data.py --seeds 1000-1199
    modal run training/modal_train.py                          # smoke
    modal run training/modal_train.py --full --run glimmer-v1  # real run

The data goes to the ``hindsight-triage`` Volume, the adapter to
``/runs/<run>`` on it, where training/modal_serve.py picks it up. Glimmer's
weights (~60 GB) are cached on the ``hf-cache`` Volume after the first run.
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "training_data" / "triage_sft.jsonl"

app = modal.App("hindsight-triage-train")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.20.2", "trl==1.16.0", "peft==0.21.2",
                    "bitsandbytes==0.50.2", "accelerate==1.15.0", "datasets==5.1.0",
                    "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(HERE / "sft.py", "/root/sft.py")
)
vol = modal.Volume.from_name("hindsight-triage", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


# QLoRA keeps the 30B base at ~17 GB, so an A100-80GB would also do if it's cheaper.
@app.function(image=image, gpu="H100", timeout=6 * 3600,
              volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache})
def train(run: str, epochs: float, max_steps: int, limit: int | None) -> str:
    import sys
    sys.path.insert(0, "/root")
    import sft

    out = sft.train("/vol/data/triage_sft.jsonl", out=f"/vol/runs/{run}",
                    epochs=epochs, max_steps=max_steps, limit=limit)
    vol.commit()
    return out


@app.local_entrypoint()
def main(run: str = "smoke", full: bool = False, epochs: float = 1.0):
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found. Run scripts/make_triage_data.py first.")
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(DATA), "/data/triage_sft.jsonl")
    if full:
        out = train.remote(run, epochs, -1, None)
    else:
        out = train.remote(run, 1, 20, 64)
    print(f"Adapter saved to Volume hindsight-triage at {out.removeprefix('/vol')}")
