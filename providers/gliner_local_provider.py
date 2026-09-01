"""Provider GLiNER local — NER sémantique (GPU) + regex Presidio pour formats structurés."""
import re
import torch
from gliner import GLiNER

MODEL_NAME = "urchade/gliner_multi-v2.1"

# GLiNER excelle sur les entités sémantiques ; les formats structurés sont gérés par regex ci-dessous
DEFAULT_LABELS = [
    "person name",
    "date of birth",
    "age",
    "postal address",
    "zip code",
    "social security number",
    "credit card number",
    "passport number",
    "driver license number",
    "national identity number",
    "organization",
    "location",
    "IBAN",
]

# Regex pour les formats structurés que GLiNER détecte mal
_REGEX_PATTERNS = [
    ("EMAIL",       re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    ("PHONE",       re.compile(r"(?:(?:\+|00)33[\s.\-]?|0)[1-9](?:[\s.\-]?\d{2}){4}")),
    ("IP_ADDRESS",  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

# Mots-clés permettant de savoir si un label demandé par l'utilisateur correspond
# à l'un des patterns regex ci-dessus (pour respecter le filtre d'entités).
_REGEX_LABEL_KEYWORDS = {
    "EMAIL": ("email", "mail", "courriel"),
    "PHONE": ("phone", "téléphone", "telephone", "tél"),
    "IP_ADDRESS": ("ip",),
}

_model = None


def _get_model() -> GLiNER:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = GLiNER.from_pretrained(MODEL_NAME)
        _model = _model.to(device)
    return _model


def _regex_entities(text: str, allowed_labels: list[str] | None = None) -> list[dict]:
    found = []
    labels_lower = [l.lower() for l in allowed_labels] if allowed_labels is not None else None
    for label, pattern in _REGEX_PATTERNS:
        if labels_lower is not None:
            keywords = _REGEX_LABEL_KEYWORDS.get(label, ())
            if not any(kw in l for kw in keywords for l in labels_lower):
                continue
        for m in pattern.finditer(text):
            found.append({"text": m.group(), "label": label, "score": 1.0, "start": m.start(), "end": m.end()})
    return found


def anonymize(text: str, custom_instruction: str = "") -> dict:
    model = _get_model()

    has_custom_filter = bool(custom_instruction.strip())
    if has_custom_filter:
        labels = [l.strip() for l in custom_instruction.replace(";", "\n", ).replace(",", "\n").splitlines() if l.strip()]
    else:
        labels = DEFAULT_LABELS

    # Seuil abaissé (0.5 -> 0.3) : en zero-shot le score varie beaucoup selon la formulation
    # exacte du label choisi par l'utilisateur (ex: "nom, prénom" scorait ~0.40 et passait
    # sous silence avec 0.5). Mieux vaut sur-détecter que rater une PII.
    gliner_entities = model.predict_entities(text, labels, threshold=0.3)
    # Sans filtre custom, la liste par défaut est censée tout couvrir -> regex non restreint.
    # Avec un filtre custom, on ne garde que les patterns regex correspondant aux labels demandés.
    regex_entities = _regex_entities(text, labels if has_custom_filter else None)

    # Fusionner et dédoublonner par position (regex prioritaire sur les chevauchements)
    all_entities = regex_entities + [
        {"text": text[e["start"]:e["end"]], "label": e["label"].upper().replace(" ", "_"),
         "score": round(e["score"], 3), "start": e["start"], "end": e["end"]}
        for e in gliner_entities
    ]
    # Supprimer les chevauchements : garder l'entité avec le score le plus haut
    all_entities.sort(key=lambda e: (e["start"], -e["score"]))
    kept, last_end = [], -1
    for ent in all_entities:
        if ent["start"] >= last_end:
            kept.append(ent)
            last_end = ent["end"]

    # Remplacer en ordre inverse pour ne pas décaler les positions
    anonymized = text
    for ent in sorted(kept, key=lambda e: e["start"], reverse=True):
        placeholder = "[" + ent["label"].replace(" ", "_") + "]"
        anonymized = anonymized[: ent["start"]] + placeholder + anonymized[ent["end"]:]

    return {
        "provider": "gliner_local",
        "original": text,
        "anonymized": anonymized,
        "entities": kept,
    }


if __name__ == "__main__":
    sample = "Bonjour, je m'appelle Jean Dupont, né le 12/03/1985. Mon email est jean.dupont@example.com et mon téléphone est 06 12 34 56 78."
    result = anonymize(sample)
    print("Texte anonymisé:", result["anonymized"])
    print("Entités:", result["entities"])
