"""Métriques unifiées d'anonymisation — méthode diff pour tous les providers."""
import re
from difflib import SequenceMatcher


def extract_detected_entities(original: str, anonymized: str) -> list[dict]:
    """
    Retrouve ce qui a été masqué en comparant original et anonymisé.
    Fonctionne pour tous les providers (Presidio, Privacy Filter, OpenMED).
    """
    sm = SequenceMatcher(None, original, anonymized, autojunk=False)
    entities = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            replaced = original[i1:i2].strip()
            replacement = anonymized[j1:j2] if tag == "replace" else ""
            if not replaced:
                continue
            # Tenter d'extraire le label depuis le placeholder
            m = re.search(r"\[(?:PRIVATE_)?([A-Z_]+?)\d*\]", replacement)
            label = m.group(1).replace("PRIVATE_", "") if m else "PII"
            entities.append({"text": replaced, "label": label, "start": i1, "end": i2})
    return entities


def compute_metrics(original: str, anonymized: str, privacy_mask: list) -> dict:
    """
    Calcule precision, recall, F1 et leakage_rate de façon unifiée.

    - Recall   = PII du ground truth absentes du texte anonymisé / total PII
    - Precision = entités détectées (via diff) qui correspondent à une PII du GT / total détectées
    - F1       = moyenne harmonique precision/recall
    - Leakage  = 1 - recall (% de PII qui ont fui)
    """
    gt_values = [
        e["value"].strip().lower()
        for e in privacy_mask
        if e.get("value", "").strip()
    ]
    if not gt_values:
        return {"precision": None, "recall": None, "f1": None, "leakage_rate": None, "support": 0}

    text_lower = anonymized.lower()
    detected = extract_detected_entities(original, anonymized)

    # Recall : PII du GT qui ne sont plus dans le texte
    masked = sum(1 for v in gt_values if v not in text_lower)
    recall = masked / len(gt_values)

    # Precision : détections qui correspondent à une vraie PII
    if detected:
        tp = sum(
            1 for e in detected
            if any(gt in e["text"].lower() or e["text"].lower() in gt for gt in gt_values)
        )
        precision = tp / len(detected)
    else:
        precision = 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    leakage = 1 - recall

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "leakage_rate": round(leakage, 3),
        "support": len(gt_values),
    }


def compute_per_label(original: str, anonymized: str, privacy_mask: list) -> dict:
    """Métriques recall/leakage par type d'entité."""
    from collections import defaultdict

    label_map = {
        "GIVENNAME": "PERSON", "SURNAME": "PERSON",
        "EMAIL_ADDRESS": "EMAIL", "TELEPHONENUM": "PHONE", "PHONE_NUMBER": "PHONE",
        "STREET": "ADDRESS", "CITY": "ADDRESS", "LOCATION": "ADDRESS",
        "DATE_TIME": "DATE",
    }

    def norm(label):
        label = label.upper()
        return label_map.get(label, label)

    gt_by_label = defaultdict(list)
    for e in privacy_mask:
        if e.get("value", "").strip():
            gt_by_label[norm(e["label"])].append(e["value"].strip().lower())

    text_lower = anonymized.lower()
    per_label = {}
    for label, values in gt_by_label.items():
        leaks = sum(1 for v in values if v in text_lower)
        recall = (len(values) - leaks) / len(values)
        per_label[label] = {
            "recall": round(recall, 3),
            "leakage": round(leaks / len(values), 3),
            "support": len(values),
        }
    return per_label
