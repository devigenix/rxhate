"""Stage 2 for the LLM: reward optimisation with GRPO (TRL) using the shared rewards.

    # 0) build the frozen faithfulness judge (char n-gram LR, seconds on CPU)
    python llm/train_grpo.py --data data/synhatemm --build-scorer runs/scorer.pkl
    # 1) GRPO from the SFT adapter (or directly from the instruct model)
    python llm/train_grpo.py --data data/synhatemm --model runs/sft --scorer runs/scorer.pkl --out runs/grpo

Requires a GPU and:  pip install -r requirements-llm.txt
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_scorer(data, out):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from llm.data import load_records
    recs = load_records(data, "train")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    X = vec.fit_transform([r["text"] for r in recs])
    clf = LogisticRegression(C=4.0, max_iter=2000).fit(X, [r["label"] for r in recs])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pickle.dump((vec, clf), open(out, "wb"))
    print("saved faithfulness judge to", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synhatemm")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--build-scorer", default=None)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-completion", type=int, default=160)
    ap.add_argument("--no-fair", action="store_true")
    ap.add_argument("--no-faith", action="store_true")
    ap.add_argument("--no-plaus", action="store_true")
    a = ap.parse_args()

    if a.build_scorer:
        build_scorer(a.data, a.build_scorer)
        return

    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from llm import reward_funcs as R
    from llm.data import hf_dataset, load_records

    if a.scorer:
        R.set_scorer(a.scorer)
    funcs = [R.format_reward, R.correctness_reward, R.rationale_shape_reward]
    if not a.no_fair:
        funcs.append(R.fairness_reward)
    if not a.no_plaus:
        funcs.append(R.plausibility_reward)
    if not a.no_faith and a.scorer:
        funcs.append(R.faithfulness_reward)

    ds = hf_dataset(load_records(a.data, "train"))
    args = GRPOConfig(
        output_dir=a.out, learning_rate=a.lr, per_device_train_batch_size=a.bs, gradient_accumulation_steps=4,
        num_generations=a.num_generations, max_completion_length=a.max_completion, beta=a.beta,
        num_train_epochs=a.epochs, logging_steps=10, save_steps=200, bf16=True, report_to="none", temperature=1.0,
    )
    peft_cfg = None
    if not os.path.isdir(a.model):  # fresh LoRA on top of a hub model; an SFT adapter dir is loaded as-is
        peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                              task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=a.model, reward_funcs=funcs, args=args, train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(a.out)


if __name__ == "__main__":
    main()
