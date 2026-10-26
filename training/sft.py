"""LoRA fine-tuning of the triage model on oracle demos (scripts/make_triage_data.py).

Plain TRL, no Modal: ``modal_train.py`` runs this on a GPU, and the same code
runs on CPU against TRL's tiny Glimmer for a free smoke test:

    python training/sft.py --data training_data/triage_sft.jsonl --model \
        trl-internal-testing/tiny-MuseGlimmerForConditionalGeneration \
        --out runs/smoke --max-steps 2 --limit 8 --no-qlora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GLIMMER = "meta-models/Muse-Glimmer-30B"

# Text decoder only; Glimmer's vision tower never sees an image here.
LORA_TARGETS = (r".*language_model\.layers\.\d+\."
                r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")


def load_rows(path: str, limit: int | None = None) -> list[dict]:
    """Prompt/completion pairs, so loss falls on the action only. Glimmer's chat
    template has no {% generation %} markers, which assistant_only_loss needs."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            messages = json.loads(line)["messages"]
            rows.append({"prompt": messages[:-1], "completion": messages[-1:]})
            if limit and len(rows) >= limit:
                break
    return rows


def train(data: str, model: str = GLIMMER, out: str = "runs/triage-lora", *,
          qlora: bool = True, epochs: float = 2, max_steps: int = -1,
          limit: int | None = None, lr: float = 1e-4, batch: int = 1,
          grad_accum: int = 16, max_length: int = 4096) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    init: dict = {"dtype": torch.bfloat16}
    if qlora:
        from transformers import BitsAndBytesConfig
        init["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    gpu = torch.cuda.is_available()

    args = SFTConfig(
        output_dir=out,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,        # a float is a ratio of total steps
        completion_only_loss=True,
        max_length=max_length,
        gradient_checkpointing=gpu,
        bf16=gpu,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        model_init_kwargs=init,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(load_rows(data, limit)),
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                               target_modules=LORA_TARGETS, task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(out)
    Path(out, "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--model", default=GLIMMER)
    p.add_argument("--out", default="runs/triage-lora")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--limit", type=int)
    p.add_argument("--no-qlora", action="store_true")
    a = p.parse_args()
    train(a.data, a.model, a.out, qlora=not a.no_qlora, epochs=a.epochs,
          max_steps=a.max_steps, limit=a.limit)


if __name__ == "__main__":
    main()
