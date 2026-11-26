"""Build LaTeX tables (paper/tables/*.tex) and figures (figures/*.pdf|png) from results/*.json."""
from __future__ import annotations

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES, TAB, FIG = "results", "paper/tables", "figures"
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#257de0", "#f47230", "#00b894", "#9a9a9a", "#191919", "#666666"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": "#bbbbbb", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
})


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ms(runs, getter, pct=True, digits=1):
    v = np.array([getter(r) for r in runs], dtype=float) * (100 if pct else 1)
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f}\\std{{{v.std():.{digits}f}}}"


def mean(runs, getter):
    return float(np.mean([getter(r) for r in runs]))


def bold_best(rows, cols, higher):
    """rows: list of [name, val_str...] with parallel numeric arrays; bold best per column."""
    return rows


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def write(name, s):
    """Rows are written WITHOUT the final row terminator; the paper adds `\\\\` after \\input
    (a trailing `\\\\` at end-of-file breaks the following \\bottomrule)."""
    s = re.sub(r"\s*\\\\\s*$", "", s.rstrip("\n"))
    open(os.path.join(TAB, name + ".tex"), "w").write(s + "\n")


main = load("main.json")
abl = load("ablation.json")
low = load("lowres.json")
fb = load("feedback.json")
xai = load("xai.json")
summary = {}

# ---------------------------------------------------------------- Table: main results
if main:
    systems = [(k, [v]) for k, v in main["baselines"].items()]
    systems += [(f"Fusion SFT ({k})", v) for k, v in main["sft"].items()]
    systems += [("\\textbf{RX-Hate: SFT + GRPO (T+V+C)}", main["grpo"]["T+V+C"])]
    lines = []
    for name, runs in systems:
        lines.append(" & ".join([
            name,
            ms(runs, lambda r: r["overall"]["acc"]),
            ms(runs, lambda r: r["overall"]["macro_f1"]),
            ms(runs, lambda r: r["overall"]["auroc"]),
            ms(runs, lambda r: r["by_template_seen"]["True"]["macro_f1"]),
            ms(runs, lambda r: r["by_template_seen"]["False"]["macro_f1"]),
            ms(runs, lambda r: r["by_lang"]["en"]["macro_f1"]),
            ms(runs, lambda r: r["by_lang"]["hien"]["macro_f1"]),
            ms(runs, lambda r: r["by_lang"]["hi"]["macro_f1"]),
        ]) + " \\\\")
        summary[name] = {"macro_f1": mean(runs, lambda r: r["overall"]["macro_f1"]),
                         "auroc": mean(runs, lambda r: r["overall"]["auroc"]),
                         "unseen": mean(runs, lambda r: r["by_template_seen"]["False"]["macro_f1"])}
    lines.insert(3, "\\midrule")
    lines.insert(7, "\\midrule")
    write("main", "\n".join(lines))

    # functional tests
    funcs = ["H_DEROG", "H_THREAT", "H_IMPLICIT", "H_SARC", "MM_HATE", "CTX_HATE",
             "N_NEUTRAL", "N_POSITIVE", "N_COUNTER", "N_NONPROT", "N_SARC", "MM_BENIGN", "MM_EDU", "CTX_BENIGN"]
    pretty = {"H_DEROG": "Derogation", "H_THREAT": "Threat / exclusion", "H_IMPLICIT": "Implicit / coded",
              "H_SARC": "Sarcastic hate", "MM_HATE": "Hate via image", "CTX_HATE": "Hate via context",
              "N_NEUTRAL": "Neutral identity mention", "N_POSITIVE": "Positive statement",
              "N_COUNTER": "Counter-speech", "N_NONPROT": "Abuse, non-protected", "N_SARC": "Benign sarcasm",
              "MM_BENIGN": "Benign image (confounder)", "MM_EDU": "Educational symbol post",
              "CTX_BENIGN": "Benign context (confounder)"}
    cols = [("Char LR", [main["baselines"]["TF-IDF char + LR"]]),
            ("EF-LR", [main["baselines"]["Early-fusion LR (T+V+C)"]]),
            ("SFT-T", main["sft"]["T"]), ("SFT", main["sft"]["T+V+C"]), ("GRPO", main["grpo"]["T+V+C"])]
    lines = []
    for f in funcs:
        lab = "H" if f.startswith(("H_", "MM_HATE", "CTX_HATE")) else "N"
        vals = [mean(r, lambda x, f=f: x["by_func"][f]["acc"]) * 100 for _, r in cols]
        best = max(vals)
        cells = [f"\\textbf{{{v:.1f}}}" if abs(v - best) < 1e-9 else f"{v:.1f}" for v in vals]
        n = main["grpo"]["T+V+C"][0]["by_func"][f]["n"]
        lines.append(f"{pretty[f]} & {lab} & {n} & " + " & ".join(cells) + " \\\\")
        if f == "CTX_HATE":
            lines.append("\\midrule")
    write("functional", "\n".join(lines))

    # fairness
    lines = []
    for name, runs in [("TF-IDF char + LR", [main["baselines"]["TF-IDF char + LR"]]),
                       ("Early-fusion LR", [main["baselines"]["Early-fusion LR (T+V+C)"]]),
                       ("Fusion SFT", main["sft"]["T+V+C"]),
                       ("\\textbf{RX-Hate (GRPO)}", main["grpo"]["T+V+C"])]:
        lines.append(" & ".join([
            name,
            ms(runs, lambda r: r["bias"]["gmb_subgroup_auc"]),
            ms(runs, lambda r: r["bias"]["gmb_bpsn_auc"]),
            ms(runs, lambda r: r["bias"]["gmb_bnsp_auc"]),
            ms(runs, lambda r: r["bias"]["fped"], pct=False, digits=2),
            ms(runs, lambda r: r["bias"]["fned"], pct=False, digits=2),
            ms(runs, lambda r: r["bias"]["identity_fpr"]),
            ms(runs, lambda r: r["counterfactual"]["ctf_gap"]),
            ms(runs, lambda r: r["counterfactual"]["cf_flip_rate"]),
        ]) + " \\\\")
        summary["fair_" + name] = {k: mean(runs, g) for k, g in [
            ("bpsn", lambda r: r["bias"]["gmb_bpsn_auc"]), ("fped", lambda r: r["bias"]["fped"]),
            ("idfpr", lambda r: r["bias"]["identity_fpr"]), ("ctf", lambda r: r["counterfactual"]["ctf_gap"]),
            ("flip", lambda r: r["counterfactual"]["cf_flip_rate"]), ("sub", lambda r: r["bias"]["gmb_subgroup_auc"]),
            ("bnsp", lambda r: r["bias"]["gmb_bnsp_auc"]), ("fned", lambda r: r["bias"]["fned"])]}
    write("fairness", "\n".join(lines))

    # training curves (seed 0)
    hist = main["logs"].get("grpo_s0")
    if hist:
        fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.2))
        ep = [h["epoch"] for h in hist]
        ax[0].plot(ep, [h["train_reward"] for h in hist], color=BLUE, marker="o", ms=4, label="train (sampled)")
        ax[0].plot(ep, [h["val_reward"] for h in hist], color=ORANGE, marker="s", ms=4, label="val (greedy)")
        ax[0].set_xlabel("GRPO epoch")
        ax[0].set_ylabel("mean reward")
        ax[0].legend(fontsize=7)
        for k, c, lab in [("r_acc", BLUE, "correctness"), ("r_faith", ORANGE, "faithfulness"),
                          ("r_plaus", AQUA, "plausibility")]:
            ax[1].plot(ep, [h[k] for h in hist], color=c, marker="o", ms=4, label=lab)
        ax[1].set_xlabel("GRPO epoch")
        ax[1].set_ylabel("mean component (train)")
        ax[1].legend(fontsize=7)
        for a in ax:
            a.set_xticks(ep)
        fig.tight_layout()
        save(fig, "grpo_curves")

# ---------------------------------------------------------------- ablations
if abl:
    order = ["SFT only", "Full (GRPO)", "- fairness reward", "- faithfulness reward", "- plausibility reward",
             "- counterfactual pairing", "- KL to reference", "REINFORCE (no group norm.)"]
    lines = []
    for name in order:
        if name not in abl:
            continue
        runs = abl[name]
        nm = name.replace("- ", "$-$ ")
        if name == "Full (GRPO)":
            nm = "\\textbf{Full RX-Hate (GRPO)}"
        lines.append(" & ".join([
            nm,
            ms(runs, lambda r: r["overall"]["macro_f1"]),
            ms(runs, lambda r: r["by_template_seen"]["False"]["macro_f1"]),
            ms(runs, lambda r: r["bias"]["gmb_bpsn_auc"]),
            ms(runs, lambda r: r["bias"]["identity_fpr"]),
            ms(runs, lambda r: r["counterfactual"]["ctf_gap"]),
            ms(runs, lambda r: r["plausibility"]["token_f1"]),
            ms(runs, lambda r: r["faithfulness"]["comprehensiveness"]),
            ms(runs, lambda r: r["faithfulness"]["sufficiency"]),
        ]) + " \\\\")
        if name == "Full (GRPO)":
            lines.append("\\midrule")
        summary["abl_" + name] = {k: mean(runs, g) for k, g in [
            ("f1", lambda r: r["overall"]["macro_f1"]), ("unseen", lambda r: r["by_template_seen"]["False"]["macro_f1"]),
            ("bpsn", lambda r: r["bias"]["gmb_bpsn_auc"]), ("idfpr", lambda r: r["bias"]["identity_fpr"]),
            ("ctf", lambda r: r["counterfactual"]["ctf_gap"]), ("tokf1", lambda r: r["plausibility"]["token_f1"]),
            ("comp", lambda r: r["faithfulness"]["comprehensiveness"]),
            ("suff", lambda r: r["faithfulness"]["sufficiency"]),
            ("ratlen", lambda r: r["faithfulness"]["rationale_len_frac"])]}
    lines.insert(1, "\\midrule")
    write("ablation", "\n".join(lines))

# ---------------------------------------------------------------- low-resource curve
if low:
    fr = sorted(low, key=float)
    n = [low[f]["n_hi_train"] for f in fr]
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    for key, c, lab, mk in [("char_lr", GRAY, "Char n-gram LR", "^"), ("sft", BLUE, "Fusion SFT", "o"),
                            ("grpo", ORANGE, "RX-Hate (GRPO)", "s")]:
        ax.plot(n, [low[f][key]["hi"]["macro_f1"] * 100 for f in fr], color=c, marker=mk, ms=5, label=lab)
    ax.set_xlabel("# romanised-Hindi training posts")
    ax.set_ylabel("Macro-F1 on Hindi test (%)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "lowres")
    summary["lowres"] = {f: {k: low[f][k]["hi"]["macro_f1"] for k in ("char_lr", "sft", "grpo")} | {"n": low[f]["n_hi_train"]}
                         for f in fr}

# ---------------------------------------------------------------- rationale-feedback curve
if fb:
    fr = sorted(fb, key=float)
    n = [fb[f]["n_rationales"] for f in fr]
    xs = np.arange(len(fr))
    fig, ax = plt.subplots(2, 1, figsize=(3.2, 4.3), sharex=True)
    for key, c, lab, mk in [("sft", BLUE, "SFT", "o"), ("grpo", ORANGE, "SFT + GRPO", "s")]:
        ax[0].plot(xs, [fb[f][key]["plausibility"]["token_f1"] * 100 for f in fr], color=c, marker=mk, ms=5, label=lab)
        ax[1].plot(xs, [fb[f][key]["faithfulness"]["comprehensiveness"] * 100 for f in fr], color=c, marker=mk, ms=5,
                   label=lab)
    for a, yl in zip(ax, ["Rationale token-F1 (%)", "Comprehensiveness (%)"]):
        a.set_xticks(xs)
        a.set_xticklabels([str(v) for v in n])
        a.set_ylabel(yl)
        a.legend(fontsize=7)
    ax[1].set_xlabel("# human rationales in training")
    fig.tight_layout()
    save(fig, "feedback")
    summary["feedback"] = {f: {"n": fb[f]["n_rationales"],
                               **{f"{k}_tokf1": fb[f][k]["plausibility"]["token_f1"] for k in ("sft", "grpo")},
                               **{f"{k}_comp": fb[f][k]["faithfulness"]["comprehensiveness"] for k in ("sft", "grpo")},
                               **{f"{k}_suff": fb[f][k]["faithfulness"]["sufficiency"] for k in ("sft", "grpo")},
                               **{f"{k}_f1": fb[f][k]["overall"]["macro_f1"] for k in ("sft", "grpo")}}
                           for f in fr}

# ---------------------------------------------------------------- explainers + modality Shapley + cases
if xai:
    lines = []
    for name, r in xai["grpo"].items():
        nm = "\\textbf{Learned rationale (ours)}" if "ours" in name else name
        rs = xai["sft"][name]
        lines.append(f"{nm} & {r['auprc'] * 100:.1f} & {r['token_f1'] * 100:.1f} & {r['iou_f1'] * 100:.1f} & "
                     f"{r['comprehensiveness'] * 100:.1f} & {r['sufficiency'] * 100:.1f} & "
                     f"{rs['token_f1'] * 100:.1f} & {rs['comprehensiveness'] * 100:.1f} & {r['sec_per_post'] * 1000:.0f} \\\\")
    write("explainers", "\n".join(lines))
    summary["xai"] = xai["grpo"]
    summary["xai_sft"] = xai["sft"]

    sh = xai["grpo_modality_shapley_abs_share"]
    funcs = [f for f in ["H_DEROG", "H_THREAT", "H_IMPLICIT", "H_SARC", "MM_HATE", "CTX_HATE"] if f in sh]
    pretty = {"H_DEROG": "Derogation", "H_THREAT": "Threat", "H_IMPLICIT": "Implicit", "H_SARC": "Sarcasm",
              "MM_HATE": "Hate via image", "CTX_HATE": "Hate via context"}
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    y = np.arange(len(funcs))
    left = np.zeros(len(funcs))
    for j, (c, lab) in enumerate([(BLUE, "text"), (ORANGE, "image"), (AQUA, "context")]):
        v = np.array([sh[f][j] for f in funcs]) * 100
        ax.barh(y, v, left=left, color=c, label=lab, height=0.62, edgecolor="white", linewidth=1.5)
        for k in range(len(funcs)):
            if v[k] >= 12:
                ax.text(left[k] + v[k] / 2, y[k], f"{v[k]:.0f}", ha="center", va="center", color="white", fontsize=7)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([pretty[f] for f in funcs])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of |Shapley| on hate logit (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, 1.17))
    fig.tight_layout()
    save(fig, "modality_shapley")
    summary["modality_share"] = sh

    # token heatmaps for case studies
    cases = [c for c in xai["cases"] if c["func"] in ("H_THREAT", "H_DEROG", "N_COUNTER", "CTX_HATE")]
    fig, axes = plt.subplots(len(cases), 1, figsize=(6.4, 0.62 * len(cases) + 0.2))
    for ax, c in zip(np.atleast_1d(axes), cases):
        toks = c["tokens"]
        ig = np.array(c["ig"])
        m = max(1e-6, np.abs(ig).max())
        x = 0.0
        for t, a in zip(toks, ig):
            w = 0.012 * len(t) + 0.025
            col = (1, 1 - 0.75 * max(a, 0) / m, 1 - 0.85 * max(a, 0) / m) if a >= 0 else \
                  (1 - 0.8 * (-a) / m, 1 - 0.45 * (-a) / m, 1)
            ax.add_patch(plt.Rectangle((x, 0.1), w - 0.006, 0.8, color=col, lw=0))
            ax.text(x + (w - 0.006) / 2, 0.5, t, ha="center", va="center", fontsize=7.5, color=INK)
            x += w
        lab = "hateful" if c["p"] >= 0.5 else "not hateful"
        ax.text(x + 0.01, 0.5, f"[{c['func']}, {c['lang']}]  p={c['p']:.2f} -> {lab}", va="center", fontsize=7,
                color=INK2)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.tight_layout(h_pad=0.2)
    save(fig, "case_heatmaps")

json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
print("wrote tables to", TAB, "and figures to", FIG)
