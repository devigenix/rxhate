"""Prompt format for the reward-optimised LLM moderator.

The model receives the three modalities of a post as text (the image through a caption /
OCR line produced by any captioner, e.g. BLIP-2 or a VLM) and must answer with JSON:

  {"label": "hateful" | "not_hateful",
   "target": "<identity group or none>",
   "rationale": ["words", "copied", "from", "the", "post"],
   "explanation": "<one sentence>"}
"""
from __future__ import annotations

SYSTEM = (
    "You are a careful, multilingual content-moderation assistant. Posts may be in English, Hindi "
    "(romanised) or code-mixed Hinglish. Decide whether the POST is hate speech: content that attacks, "
    "dehumanises, threatens or calls for exclusion of people based on a protected identity (religion, caste, "
    "gender, sexuality, origin, disability...). Quoting hate in order to condemn it (counter-speech), "
    "insulting non-protected things, and neutral or positive mentions of a group are NOT hate. Use the image "
    "description and the parent post when the text alone is ambiguous. Answer ONLY with a JSON object with "
    "keys label (hateful|not_hateful), target, rationale (list of words copied verbatim from the post; empty "
    "if not hateful) and explanation (one sentence)."
)

SYMBOL_CAPTIONS = {  # captions for the synthetic benchmark's images
    "circle": "a plain coloured circle", "square": "a plain coloured square", "star": "a five-pointed star",
    "heart": "a heart shape", "sun": "a small sun with rays", "triangle_up": "an upright triangle",
    "mark_a": "an inverted dark-red triangle with a black bar, a symbol used by an extremist group",
    "mark_b": "a black double-barred cross, a symbol used by an extremist group",
}


def image_caption(rec, captions=None):
    if captions and rec["id"] in captions:
        return captions[rec["id"]]
    if rec.get("image_symbol"):
        return "meme image showing " + SYMBOL_CAPTIONS[rec["image_symbol"]]
    return "no image"


def build_messages(rec, captions=None):
    user = (f"POST: {rec['text']}\n"
            f"PARENT POST (context): {rec.get('parent_text') or 'none'}\n"
            f"IMAGE: {image_caption(rec, captions)}\n"
            f"COMMUNITY: {rec.get('community', 'unknown')}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def target_answer(rec):
    """Gold JSON answer used for the SFT warm-up."""
    import json
    hate = rec["label"] == 1
    rat = [t for t, m in zip(rec["tokens"], rec["rationale"]) if m] if hate else []
    tgt = rec.get("target_category", "none") if hate else "none"
    expl = ("The post attacks a protected group" + (f" ({tgt})" if tgt != "none" else "") + "."
            if hate else "The post does not attack anyone for a protected identity.")
    return json.dumps({"label": "hateful" if hate else "not_hateful", "target": tgt, "rationale": rat,
                       "explanation": expl}, ensure_ascii=False)
