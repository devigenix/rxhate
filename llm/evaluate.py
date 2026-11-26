"""Evaluate an LLM moderator (base, SFT or GRPO checkpoint) with the same metrics as the
NumPy policy: macro-F1, bias AUCs, identity FPR, rationale plausibility and format validity.

    python llm/evaluate.py --data data/synhatemm --model runs/grpo --split test --out results/llm_grpo.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synhatemm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm.data import load_records
    from llm.prompts import build_messages
    from rxhate.evaluate import evaluate_probs
    from rxhate.metrics import plausibility
    from rxhate.rewards import parse_completion, rationale_words_to_mask

    if os.path.exists(os.path.join(a.model, "adapter_config.json")):
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
        tok = AutoTokenizer.from_pretrained(model.peft_config["default"].base_model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
        tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    recs = load_records(a.data, a.split)[: a.limit]
    outs = []
    for s in range(0, len(recs), a.bs):
        batch = recs[s:s + a.bs]
        prompts = [tok.apply_chat_template(build_messages(r), tokenize=False, add_generation_prompt=True)
                   for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=160, do_sample=False)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    parsed = [parse_completion(o) for o in outs]
    y = np.array([r["label"] for r in recs])
    p1 = np.array([1.0 if (p["label"] == 1) else 0.0 for p in parsed])
    res = {k: v for k, v in evaluate_probs(recs, y, p1).items()}
    res["format_valid"] = float(np.mean([p["ok"] for p in parsed]))
    hate = [i for i, r in enumerate(recs) if r["label"] == 1 and parsed[i]["label"] == 1]
    masks = [rationale_words_to_mask(recs[i]["tokens"], parsed[i]["rationale"]) for i in hate]
    res["plausibility"] = plausibility([m.astype(float) for m in masks], [np.array(recs[i]["rationale"]) for i in hate],
                                       masks)
    res["examples"] = [{"text": r["text"], "label": r["label"], "output": o} for r, o in zip(recs[:20], outs[:20])]
    print(json.dumps(res["overall"], indent=1), "format_valid", res["format_valid"])
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
