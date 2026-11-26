# Team

CSET-346 Natural Language Processing, Bennett University. Supervisor: Dr. Pradeepika Verma.

| Member | Enrolment no. | GitHub |
|---|---|---|
| Prince Chauhan | S24CSEU2811 | @devigenix |
| Shourya Dagar | S24CSEU2802 | @ |
| Ronit | S24CSEU2825 | @ |
| Aaryan Singh | S24CSEU2225 | @ |
| Gyankush Bordoloi | S24CSEU0770 | @ |

## Individual contributions

Each member adds their GitHub username above and commits from their own account.

| Member | What I did |
|---|---|
| Prince Chauhan | Dataset and infrastructure: SynHateMM generator (14 functionalities, gold rationales, counterfactuals, held-out templates) in `rxhate/data/synthetic.py`; real-corpus loaders in `rxhate/data/loaders.py`; GitHub repository and Vercel project page. |
| Shourya Dagar | Representation and model: code-mixed text normalisation and hashed n-gram features, image descriptor, context features (`text.py`, `vision.py`, `features.py`); fusion policy with label and rationale heads and SFT training (`model.py`). |
| Ronit | Reward optimisation: composite reward (`rewards.py`) and the GRPO trainer with group-normalised advantage, PPO clip, KL and counterfactual pairing (`grpo.py`); reward ablations. |
| Aaryan Singh | Explainability and LLM path: Integrated Gradients, occlusion, LIME, KernelSHAP, modality Shapley (`xai.py`); Qwen2.5 + LoRA prompts, TRL reward wrappers, SFT/GRPO scripts (`llm/`). |
| Gyankush Bordoloi | Evaluation and writing: metrics, fairness and bias measures, baselines, unit tests (`metrics.py`, `baselines.py`, `tests/`); experiment and report scripts; paper, slides and progress report. |
