"""Provider GLiNER2 local — modèle PII multilingue de Fastino (package `gliner2`)
+ regex Presidio pour formats structurés (partagées avec gliner_local_provider).

Nouvelle génération de GLiNER, distincte de `gliner` (v1) : API et poids différents.
fastino/gliner2.5-multi-v1 n'est pas encore chargeable avec gliner2 2.0.0 (PyPI).
"""
import torch
from gliner2 import GLiNER2

from providers.gliner_local_provider import DEFAULT_LABELS, _regex_entities

MODEL_NAME = "fastino/gliner2-privacy-filter-PII-multi"

_model = None


def _get_model() -> GLiNER2:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = GLiNER2.from_pretrained(MODEL_NAME)
        _model = _model.to(device)
    return _model


def anonymize(text: str, custom_instruction: str = "") -> dict:
    model = _get_model()

    has_custom_filter = bool(custom_instruction.strip())
    if has_custom_filter:
        labels = [l.strip() for l in custom_instruction.replace(";", "\n").replace(",", "\n").splitlines() if l.strip()]
    else:
        labels = DEFAULT_LABELS

    # Même seuil que gliner_local_provider : mieux vaut sur-détecter que rater une PII.
    result = model.extract_entities(text, labels, threshold=0.3, include_spans=True, include_confidence=True)
    gliner_entities = [
        {"text": text[e["start"]:e["end"]], "label": label.upper().replace(" ", "_"),
         "score": round(e["confidence"], 3), "start": e["start"], "end": e["end"]}
        for label, ents in result["entities"].items()
        for e in ents
    ]
    regex_entities = _regex_entities(text, labels if has_custom_filter else None)

    # Fusionner et dédoublonner par position (regex prioritaire sur les chevauchements)
    all_entities = regex_entities + gliner_entities
    all_entities.sort(key=lambda e: (e["start"], -e["score"]))
    kept, last_end = [], -1
    for ent in all_entities:
        if ent["start"] >= last_end:
            kept.append(ent)
            last_end = ent["end"]

    # Remplacer en ordre inverse pour ne pas décaler les positions
    anonymized = text
    for ent in sorted(kept, key=lambda e: e["start"], reverse=True):
        anonymized = anonymized[: ent["start"]] + f"[{ent['label']}]" + anonymized[ent["end"]:]

    return {
        "provider": "gliner2_local",
        "original": text,
        "anonymized": anonymized,
        "entities": kept,
    }


if __name__ == "__main__":
    sample = "Bonjour, je m'appelle Jean Dupont, né le 12/03/1985. Mon email est jean.dupont@example.com et mon téléphone est 06 12 34 56 78."
    result = anonymize(sample)
    print("Texte anonymisé:", result["anonymized"])
    print("Entités:", result["entities"])
