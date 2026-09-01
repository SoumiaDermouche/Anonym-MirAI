"""Provider GLiNER fine-tuné FR — même architecture que gliner_local_provider.py,
mais charge le modèle fine-tuné localement sur le subset français d'ai4privacy
(training/finetune_gliner.py) au lieu du checkpoint urchade/gliner_multi-v2.1 de base.

Pour passer d'un checkpoint POC à un checkpoint entraîné sur le train complet,
il suffit de réécrire MODEL_PATH (ou de pointer MODEL_PATH vers le nouveau dossier)
— aucun autre changement de code n'est nécessaire.
"""
import re
from pathlib import Path

import torch
from gliner import GLiNER

MODEL_PATH = Path(__file__).parent.parent / "models" / "gliner-fr-full"

# Labels appris pendant le fine-tuning (cf. training/prepare_gliner_data.py::LABEL_TO_TEXT,
# dérivés des 19 catégories d'ai4privacy/pii-masking-openpii-1m). GLiNER reste zero-shot :
# d'autres labels fonctionneront aussi, mais avec une précision moindre que ceux-ci.
DEFAULT_LABELS = [
    "title",
    "given name",
    "surname",
    "date",
    "city",
    "street",
    "building number",
    "zip code",
    "age",
    "sex",
    "gender",
    "email address",
    "phone number",
    "credit card number",
    "national id card number",
    "passport number",
    "driver license number",
    "tax identification number",
    "social security number",
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
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle fine-tuné introuvable dans {MODEL_PATH}. "
                "Lancez d'abord training/finetune_gliner.py (voir README, section Fine-tuning)."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = GLiNER.from_pretrained(str(MODEL_PATH), local_files_only=True)
        _model = _model.to(device)
    return _model


def _regex_entities(text: str, allowed_labels: list[str] | None = None) -> list[dict]:
    found = []
    labels_lower = [l.lower() for l in allowed_labels] if allowed_labels is not None else None
    for label, pattern in _REGEX_PATTERNS:
        if labels_lower is not None:
            keywords = _REGEX_LABEL_KEYWORDS.get(label, ())
            if not any(kw in l for l in labels_lower for kw in keywords):
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
    # exacte du label choisi par l'utilisateur. Mieux vaut sur-détecter que rater une PII.
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
        "provider": "gliner_fr_local",
        "original": text,
        "anonymized": anonymized,
        "entities": kept,
    }


if __name__ == "__main__":
    sample = "Bonjour, je m'appelle Jean Dupont, né le 12/03/1985. Mon email est jean.dupont@example.com et mon téléphone est 06 12 34 56 78."
    result = anonymize(sample)
    print("Texte anonymisé:", result["anonymized"])
    print("Entités:", result["entities"])
