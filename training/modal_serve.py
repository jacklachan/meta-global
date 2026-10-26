"""Serve Muse Glimmer 30B with vLLM on Modal, OpenAI-compatible.

One server, two models, so the untrained and fine-tuned rows of the
benchmark run on identical hardware:

    glimmer-base     Muse Glimmer as released
    glimmer-triage   the same model with the LoRA adapter from modal_train.py

Costs GPU credits while a container is up; it scales to zero after 5 idle minutes.

    modal secret create hindsight-vllm VLLM_API_KEY=<a long random string>
    modal deploy training/modal_serve.py

Then in .env:

    COPILOT_BASE_URL_TRIAGE=https://<workspace>--hindsight-triage-serve.modal.run/v1
    COPILOT_API_KEY_TRIAGE=<the same VLLM_API_KEY>
    COPILOT_MODEL_TRIAGE=glimmer-triage        # or glimmer-base
"""

from __future__ import annotations

import modal

GLIMMER = "meta-models/Muse-Glimmer-30B"
RUN = "glimmer-v1"          # adapter folder on the Volume, from modal_train.py --run
PORT = 8000

app = modal.App("hindsight-triage")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.32.0", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
vol = modal.Volume.from_name("hindsight-triage", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


# BF16 weights are ~60 GB, so this needs an 80 GB card.
@app.function(image=image, gpu="H100", scaledown_window=300, timeout=24 * 3600,
              volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("hindsight-vllm", required_keys=["VLLM_API_KEY"])])
@modal.concurrent(max_inputs=16)
@modal.web_server(port=PORT, startup_timeout=30 * 60)
def serve():
    import subprocess
    from pathlib import Path

    cmd = [
        "vllm", "serve", GLIMMER,
        "--served-model-name", "glimmer-base",
        "--host", "0.0.0.0", "--port", str(PORT),
        "--max-model-len", "8192",           # triage prompts are ~3k tokens, replies <=1.5k
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',   # text only; skips vision memory
    ]
    # Before the first training run there is no adapter; serve the base alone.
    if Path(f"/vol/runs/{RUN}/adapter_config.json").exists():
        cmd += ["--enable-lora", "--max-lora-rank", "16",
                "--lora-modules", f"glimmer-triage=/vol/runs/{RUN}"]
    # vLLM reads VLLM_API_KEY from the environment and rejects requests without it.
    subprocess.Popen(cmd)
