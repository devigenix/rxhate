# RX-Hate: Explainable, Reward-Optimised Multimodal Hate Speech Detection

Source code for the project **"Hate Speech Detection through Explainable AI and Reward-Optimized LLMs"**.

A social-media post is a multimodal input **x = {T, V, C}**: text *T* (English, romanised Hindi or
code-mixed Hinglish), an image/meme *V* and context *C* (parent post, community, thread toxicity).
We learn a policy that outputs a **label and its evidence**, π(y, z | x) = π(y | x) · π(z | x, y), and
optimise it with **GRPO** against a reward that combines correctness, fairness, faithfulness, plausibility
and conciseness. The same reward functions also drive an **LLM** (Qwen2.5 + LoRA, TRL `GRPOTrainer`).

```
          ┌──────── text T ──► hashed word + char n-grams ──► h_T ─┐
 post x ──┼──────── image V ─► colour/HOG descriptor (or CLIP) ► h_V ─┼─► fusion ─► π(y|x)   label
          └──────── context C ► parent post + metadata ─────────► h_C ─┘
                       tokens ─► token + neighbour features ─────────────────────► π(z|x,y=1) rationale
 Stage 1  SFT:  cross-entropy + rationale BCE (on the few posts with human rationales)
 Stage 2  GRPO: sample G actions (y,z) per post → reward R(x,y,z) → group-normalised advantage
                → clipped surrogate + KL(π‖π_SFT) + counterfactual logit pairing
 XAI      Integrated Gradients · LIME · KernelSHAP · occlusion · learned rationale · modality Shapley
```

## Repository layout

| Path | What it is |
|---|---|
| `rxhate/data/synthetic.py` | **SynHateMM** generator: a HateCheck-style multimodal, code-mixed benchmark with gold rationales, counterfactual identity swaps and held-out templates |
| `rxhate/data/loaders.py` | Loaders for HateXplain, HASOC, Hateful Memes and generic CSV (Hinglish) → unified schema |
| `rxhate/text.py` | Normalisation for code-mixed text (de-leet, elongation, Hinglish spelling variants) + signed feature hashing |
| `rxhate/vision.py` | Dependency-free image descriptor (swap for CLIP in production) |
| `rxhate/model.py` | NumPy fusion policy with label + rationale heads, Adam, SFT loop |
| `rxhate/rewards.py` | **Shared reward functions** (used by both the NumPy policy and the LLM) |
| `rxhate/grpo.py` | GRPO / PPO-clip trainer with exact KL and counterfactual logit pairing |
| `rxhate/xai.py` | IG, occlusion, LIME, KernelSHAP, learned rationale, exact modality Shapley, natural-language explanations |
| `rxhate/metrics.py` | Macro-F1/AUROC, bias AUCs (subgroup/BPSN/BNSP, GMB), FPED/FNED, counterfactual fairness, ERASER plausibility & faithfulness |
| `llm/` | LLM path: prompts, TRL reward wrappers, `train_sft.py`, `train_grpo.py`, `evaluate.py` |
| `scripts/run_experiments.py` | Reproduces every table and figure in the paper |
| `scripts/make_report.py` | Builds LaTeX tables + figures from `results/*.json` |
| `scripts/prepare_real_data.py`, `scripts/train_real.py` | Run the whole pipeline on real datasets |
| `scripts/demo.py` | Explain a decision for any post you type |
| `tests/` | Unit tests (incl. a finite-difference gradient check of the hand-written backprop) |

## Quick start (CPU, ~2 h for everything)

```bash
pip install -r requirements.txt
python scripts/run_experiments.py --stage data      # generate SynHateMM (9,000 posts + images)
python scripts/run_experiments.py --stage main      # baselines, SFT, GRPO (3 seeds)
python scripts/run_experiments.py --stage ablation  # reward/optimiser ablations
python scripts/run_experiments.py --stage lowres    # low-resource Hindi curve
python scripts/run_experiments.py --stage feedback  # human-rationale budget curve
python scripts/run_experiments.py --stage xai       # explainer comparison, modality Shapley, case studies
python scripts/make_report.py                       # tables -> paper/tables, figures -> figures/
pytest -q tests
```
`SMOKE=1 python scripts/run_experiments.py --stage main` runs a one-minute end-to-end check.

Explain a post:
```bash
python scripts/demo.py --text "get rid of all of them" --parent "a muslim family just moved into our street"
```

## Real datasets

```bash
# HateXplain (token rationales + target groups)
git clone https://github.com/hate-alert/HateXplain
python scripts/prepare_real_data.py hatexplain --dataset-json HateXplain/Data/dataset.json \
       --divisions-json HateXplain/Data/post_id_divisions.json --out data/hatexplain
python scripts/train_real.py --data data/hatexplain --out results/hatexplain

# Facebook Hateful Memes (multimodal)
python scripts/prepare_real_data.py memes --root /path/to/hateful_memes --out data/memes
python scripts/train_real.py --data data/memes --images-root /path/to/hateful_memes --out results/memes

# Any code-mixed CSV (e.g. Hinglish corpora)
python scripts/prepare_real_data.py csv --path hinglish.csv --text-col text --label-col label --positive 1 \
       --lang hien --out data/hinglish
```

## LLM path (GPU)

```bash
pip install -r requirements-llm.txt
python llm/train_grpo.py --data data/synhatemm --build-scorer runs/scorer.pkl    # frozen faithfulness judge
python llm/train_sft.py  --data data/synhatemm --model Qwen/Qwen2.5-1.5B-Instruct --out runs/sft
python llm/train_grpo.py --data data/synhatemm --model runs/sft --scorer runs/scorer.pkl --out runs/grpo
python llm/evaluate.py   --data data/synhatemm --model runs/grpo --out results/llm_grpo.json
```
The LLM answers in JSON (`label`, `target`, `rationale` words copied from the post, `explanation`).
Rewards: format validity, correctness, identity-FP fairness penalty, rationale plausibility, rationale
shape (sparsity, no evidence for non-hate, no hallucinated evidence) and faithfulness measured with a
frozen judge on the rationale-only / rationale-removed text. Images reach the LLM as captions
(`llm/prompts.py`); a VLM (e.g. Qwen2-VL) can be dropped in with the same rewards.

## Notes on the data

SynHateMM is synthetic and template-based (in the spirit of HateCheck). It contains no slurs, and its
"hate symbols" are invented shapes. It exists so that modality dependence, counter-speech, identity
bias, code-mixing and rationale quality can be measured with exact ground truth; **conclusions should be
confirmed on real corpora** with the loaders above. Use of real hate speech data must follow each
dataset's licence and an appropriate ethics review.

## Citation
See `paper/` for the ACL-format paper.

## License
MIT (code). Datasets keep their own licences.

## Team

Prince Chauhan (S24CSEU2811), Shourya Dagar (S24CSEU2802), Ronit (S24CSEU2825), Aaryan Singh (S24CSEU2225), Gyankush Bordoloi (S24CSEU0770). See [CONTRIBUTORS.md](CONTRIBUTORS.md) for individual contributions and `docs/` for the Milestone-2 progress report.
