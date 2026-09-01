"""Provider Privacy Filter local — openai/privacy-filter (50M paramètres actifs, GPU)."""
import re
import torch
from transformers import pipeline

MODEL_NAME = "openai/privacy-filter"
_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        device = 0 if torch.cuda.is_available() else -1
        # Pas d'aggregation_strategy : on gère la fusion BIOES manuellement
        # pour mieux contrôler les limites de span (évite d'inclure la ponctuation)
        _pipe = pipeline("token-classification", model=MODEL_NAME, device=device)
    return _pipe


def _merge_bioes(tokens: list, text: str) -> list:
    """
    Fusionne les tokens B-/I-/E-/S- en spans complets.
    Fusionne aussi les spans consécutifs de même label séparés par des espaces.
    Élimine la ponctuation en bordure des spans.
    """
    spans = []
    current = None

    for tok in tokens:
        tag   = tok["entity"]          # ex: "B-private_person"
        label = re.sub(r"^[BIES]-", "", tag).replace("private_", "").upper()
        start = tok["start"]
        end   = tok["end"]
        score = tok["score"]

        prefix = tag[0]  # B, I, E ou S

        if prefix in ("B", "S"):
            if current:
                spans.append(current)
            current = {"start": start, "end": end, "label": label, "score": score}
        elif prefix in ("I", "E") and current and current["label"] == label:
            current["end"] = end
            current["score"] = min(current["score"], score)
            if prefix == "E":
                spans.append(current)
                current = None
        else:
            if current:
                spans.append(current)
            current = {"start": start, "end": end, "label": label, "score": score}

    if current:
        spans.append(current)

    # Fusionne les spans consécutifs de même label séparés uniquement par des espaces
    merged = []
    for span in spans:
        if merged and merged[-1]["label"] == span["label"]:
            gap = text[merged[-1]["end"]: span["start"]]
            if gap.strip() == "":
                merged[-1]["end"] = span["end"]
                merged[-1]["score"] = min(merged[-1]["score"], span["score"])
                continue
        merged.append(dict(span))

    # Retire la ponctuation en bordure de chaque span
    for s in merged:
        while s["start"] < s["end"] and not text[s["start"]].isalnum():
            s["start"] += 1
        while s["end"] > s["start"] and not text[s["end"] - 1].isalnum():
            s["end"] -= 1

    return [s for s in merged if s["start"] < s["end"]]


def anonymize(text: str, custom_instruction: str = "") -> dict:
    """Détecte et masque les PII via openai/privacy-filter."""
    pipe = _get_pipe()
    raw_tokens = pipe(text)
    spans = _merge_bioes(raw_tokens, text)

    anonymized_parts = []
    last_end = 0
    entities = []

    for s in sorted(spans, key=lambda x: x["start"]):
        anonymized_parts.append(text[last_end: s["start"]])
        anonymized_parts.append(f'[{s["label"]}]')
        last_end = s["end"]
        entities.append({
            "text":  text[s["start"]: s["end"]],
            "label": s["label"],
            "score": round(float(s["score"]), 3),
            "start": s["start"],
            "end":   s["end"],
        })

    anonymized_parts.append(text[last_end:])

    return {
        "provider": "privacy_filter_local",
        "original": text,
        "anonymized": "".join(anonymized_parts),
        "entities": entities,
    }


if __name__ == "__main__":
    sample = "Bonjour, je m'appelle Jean Dupont, né le 12/03/1985. Mon email est jean.dupont@example.com et mon téléphone est 06 12 34 56 78."
    result = anonymize(sample)
    print("Texte anonymisé:", result["anonymized"])
    print("Entités:")
    for e in result["entities"]:
        print(f"  {e['label']}: '{e['text']}' (score={e['score']:.3f})")
