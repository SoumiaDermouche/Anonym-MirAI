"""
Assertion promptfoo custom — évalue la qualité d'anonymisation.
Retourne recall comme score principal + détails dans reason.
"""
import sys
import json
sys.path.insert(0, "/home/onyxia/work/Anonym-MirAI")
from assertions.metrics import compute_metrics


def get_assert(output: str, context: dict) -> dict:
    """
    output  : texte anonymisé retourné par le provider
    context : {'vars': {'text': ..., 'privacy_mask_json': ...}, ...}
    """
    original = context["vars"].get("text", "")
    mask_json = context["vars"].get("privacy_mask_json", "[]")

    try:
        privacy_mask = json.loads(mask_json)
    except Exception:
        return {"pass": False, "score": 0.0, "reason": "privacy_mask_json invalide"}

    m = compute_metrics(original, output, privacy_mask)

    if m["recall"] is None:
        return {"pass": True, "score": 1.0, "reason": "Aucune PII dans cet exemple"}

    # Seuil minimal : recall >= 0.7 (au moins 70% des PII masquées)
    passed = m["recall"] >= 0.7

    reason = (
        f"Recall={m['recall']:.3f} | Precision={m['precision']:.3f} | "
        f"F1={m['f1']:.3f} | Leakage={m['leakage_rate']:.3f} | "
        f"Support={m['support']} entités PII"
    )

    return {
        "pass": passed,
        "score": m["recall"],   # score principal = recall (privacy avant tout)
        "reason": reason,
    }
