# Module guide

Short guide to each part of RX-Hate, for the team and for reviewers.

## 1. Data (Prince): `rxhate/data/synthetic.py`, `rxhate/data/loaders.py`
- Generates SynHateMM: 9,000 posts (6,348 train / 837 val / 1,815 test), 46-49% hateful, 14 functionalities (6 hateful, 8 non-hateful).
- Hateful posts carry gold token rationales; identity-swap counterfactuals are generated; test templates and lexicon items are held out from training.
- Loaders convert HateXplain, HASOC, Hateful Memes and generic CSV files to one schema.
- Run: `python scripts/run_experiments.py --stage data`
- Likely questions: why synthetic data? how are classes balanced? what is held out and why?

## 2. Features and model (Shourya): `text.py`, `vision.py`, `features.py`, `model.py`
- Text: code-mixed normalisation (de-leet, elongation, Hinglish spellings), hashed word and char n-grams.
- Image: colour histogram + HOG descriptor (CLIP can replace it). Context: parent post and thread features.
- `FusionPolicy`: NumPy network with a label head and a rationale head; trained with label cross-entropy + rationale BCE and modality dropout (SFT).
- Likely questions: why hashing? how do the modalities fuse? how is the rationale head trained?

## 3. Reward and GRPO (Ronit): `rewards.py`, `grpo.py`
- Reward = accuracy + fairness + faithfulness + plausibility + sparsity terms (weights 1.0 / 0.5 / 0.5 / 0.5 / 0.3).
- GRPO: G=8 samples per post, group-normalised advantage, PPO clip 0.2, KL to the SFT model (beta 0.04), counterfactual logit pairing (lambda 0.5).
- Likely questions: why group normalisation? what does each reward term punish? what happens without KL?

## 4. Explainability and LLM path (Aaryan): `xai.py`, `llm/`
- Integrated Gradients, occlusion, LIME, KernelSHAP, learned rationale, exact 3-player modality Shapley.
- `llm/`: Qwen2.5 + LoRA with TRL SFT/GRPO trainers that reuse the same reward functions. Not yet run (needs a GPU).
- Run: `python scripts/demo.py --text "your post"`
- Likely questions: how do you measure faithfulness? what does modality Shapley tell you?

## 5. Evaluation (Gyankush): `metrics.py`, `baselines.py`, `evaluate.py`, `tests/`
- Macro-F1, AUROC, subgroup/BPSN/BNSP AUC, FPED/FNED, identity false-positive rate, counterfactual gap, ERASER plausibility and faithfulness.
- Baselines: TF-IDF + LR, early-fusion LR. Eleven unit tests, including a finite-difference gradient check.
- Run: `pytest -q tests`
- Likely questions: why is early-fusion LR competitive? what is identity FPR?
