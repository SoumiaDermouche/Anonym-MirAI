"""Provider OpenMed local — privacy-filter-multilingual (54 catégories PII, 16 langues, GPU)."""
import torch
from openmed import extract_pii

MODEL_NAME = "OpenMed/privacy-filter-multilingual"
_initialized = False


def _init():
    global _initialized
    if not _initialized:
        extract_pii("init", model_name=MODEL_NAME)
        _initialized = True


def _merge_spans(entities: list, original: str) -> list:
    """
    Fusionne les tokens consécutifs ayant le même label.
    Le modèle découpe parfois un mot en sous-tokens (ex: Dupont → Dup + ont).
    """
    if not entities:
        return []

    merged = []
    current = {
        "start": entities[0].start,
        "end":   entities[0].end,
        "label": entities[0].label,
        "score": entities[0].confidence,
    }

    for ent in entities[1:]:
        same_label = ent.label == current["label"]
        # Adjacent ou séparé uniquement par des espaces
        gap = original[current["end"]: ent.start]
        contiguous = same_label and gap.strip() == ""

        if contiguous:
            current["end"] = ent.end
            current["score"] = min(current["score"], ent.confidence)
        else:
            merged.append(current)
            current = {
                "start": ent.start,
                "end":   ent.end,
                "label": ent.label,
                "score": ent.confidence,
            }

    merged.append(current)
    return merged


def anonymize(text: str, custom_instruction: str = "") -> dict:
    """Détecte et anonymise les PII via OpenMed/privacy-filter-multilingual."""
    _init()

    result = extract_pii(text, model_name=MODEL_NAME)
    raw_entities = result.entities if hasattr(result, "entities") else []

    spans = _merge_spans(raw_entities, text)

    # Reconstruire le texte anonymisé à partir des spans fusionnés
    anonymized_parts = []
    last_end = 0
    for span in sorted(spans, key=lambda s: s["start"]):
        anonymized_parts.append(text[last_end: span["start"]])
        anonymized_parts.append(f'[{span["label"]}]')
        last_end = span["end"]
    anonymized_parts.append(text[last_end:])
    anonymized = "".join(anonymized_parts)

    entities = [
        {
            "text":  text[s["start"]: s["end"]],
            "label": s["label"].upper(),
            "score": round(float(s["score"]), 3),
            "start": s["start"],
            "end":   s["end"],
        }
        for s in spans
    ]

    return {
        "provider": "openmed_local",
        "original": text,
        "anonymized": anonymized,
        "entities": entities,
    }


if __name__ == "__main__":
    sample = "Bonjour, je m'appelle Jean Dupont, né le 12/03/1985. Mon email est jean.dupont@example.com et mon téléphone est 06 12 34 56 78."
    result = anonymize(sample)
    print("Texte anonymisé:", result["anonymized"])
    print("Entités:")
    for e in result["entities"]:
        print(f"  {e['label']}: '{e['text']}' (score={e['score']:.3f})")
