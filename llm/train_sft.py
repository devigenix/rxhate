"""Stage 1 for the LLM: supervised warm-up on gold JSON answers (LoRA).

    python llm/train_sft.py --data data/synhatemm --model Qwen/Qwen2.5-1.5B-Instruct --out runs/sft

Requires a GPU and:  pip install -r requirements-llm.txt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synhatemm")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--only-gold-rationales", action="store_true",
                    help="warm up only on posts that carry human rationales (low-feedback setting)")
    a = ap.parse_args()

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    from llm.data import hf_dataset, load_records

    recs = load_records(a.data, "train")
    if a.only_gold_rationales:
        recs = [r for r in recs if r.get("has_gold_rationale")]
    ds = hf_dataset(recs, with_answer=True).select_columns(["messages"])
    trainer = SFTTrainer(
        model=a.model,
        train_dataset=ds,
        args=SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
                       per_device_train_batch_size=a.bs, gradient_accumulation_steps=2, logging_steps=20,
                       save_strategy="epoch", bf16=True, report_to="none"),
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                               task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(a.out)


if __name__ == "__main__":
    main()
