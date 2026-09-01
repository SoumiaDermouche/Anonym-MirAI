"""Génère un rapport HTML d'analyse d'erreurs avec entités colorées et métriques corrigées."""
import ast
import time
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from providers.presidio_provider import anonymize as presidio_anonymize
from providers.privacy_filter_local_provider import anonymize as privacy_filter_anonymize
from providers.openmed_local_provider import anonymize as openmed_anonymize

SAMPLE_SIZE = 30

ENTITY_COLORS = {
    "PERSON": "#4A90D9", "GIVENNAME": "#4A90D9", "SURNAME": "#4A90D9",
    "EMAIL": "#27AE60", "EMAIL_ADDRESS": "#27AE60",
    "PHONE": "#E67E22", "TELEPHONENUM": "#E67E22", "PHONE_NUMBER": "#E67E22",
    "ADDRESS": "#8E44AD", "STREET": "#8E44AD", "CITY": "#8E44AD", "LOCATION": "#8E44AD",
    "DATE": "#E74C3C", "DATE_TIME": "#E74C3C",
    "AGE": "#16A085", "URL": "#95A5A6",
}

def get_color(label: str) -> str:
    label = label.upper().replace(" ", "_")
    for key, color in ENTITY_COLORS.items():
        if key in label:
            return color
    return "#7F8C8D"


# --- Highlighting HTML ---
def highlight_text(text: str, entities: list[dict], use_spans: bool = True) -> str:
    """Insère des balises <mark> colorées autour des entités dans le texte."""
    if not entities or not use_spans:
        return text.replace("<", "&lt;").replace(">", "&gt;")

    sorted_ents = sorted(
        [e for e in entities if e.get("start") is not None and e.get("end") is not None],
        key=lambda e: e["start"]
    )
    result = []
    pos = 0
    for ent in sorted_ents:
        start, end = ent["start"], ent["end"]
        if start < pos:
            continue
        result.append(text[pos:start].replace("<", "&lt;").replace(">", "&gt;"))
        color = get_color(ent["label"])
        snippet = text[start:end].replace("<", "&lt;").replace(">", "&gt;")
        result.append(
            f'<mark style="background:{color};color:white;border-radius:3px;'
            f'padding:1px 4px;margin:0 1px;" title="{ent["label"]}">'
            f'{snippet}<sup style="font-size:9px;margin-left:2px">{ent["label"]}</sup></mark>'
        )
        pos = end
    result.append(text[pos:].replace("<", "&lt;").replace(">", "&gt;"))
    return "".join(result)


def highlight_ground_truth(text: str, privacy_mask: list) -> str:
    entities = [
        {"text": e["value"], "label": e["label"], "start": e["start"], "end": e["end"]}
        for e in privacy_mask if "start" in e and "end" in e
    ]
    return highlight_text(text, entities)


# --- Métriques ---
def compute_metrics(original: str, anonymized: str, privacy_mask: list, entities: list) -> dict:
    gt_values = [e["value"].strip().lower() for e in privacy_mask if "value" in e and e["value"].strip()]
    gt_labels = {e["label"].upper() for e in privacy_mask}
    total = len(gt_values)
    if total == 0:
        return {"precision": None, "recall": None, "f1": None, "leakage_rate": None, "total_pii": 0}

    text_lower = anonymized.lower()
    leaks = sum(1 for v in gt_values if v in text_lower)
    masked = total - leaks
    recall = masked / total

    if entities:
        uses_placeholders = any((e.get("text") or "").startswith("[") for e in entities)
        if uses_placeholders:
            true_pos = sum(1 for e in entities if e.get("label", "").upper() in gt_labels)
        else:
            true_pos = sum(
                1 for e in entities
                if any(v in (e.get("text") or "").lower() for v in gt_values)
            )
        precision = true_pos / len(entities) if entities else 0.0
    else:
        precision = 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "leakage_rate": round(leaks / total, 3),
        "total_pii": total,
    }


# --- Génération HTML ---
HTML_HEAD = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Rapport d'analyse — Anonymisation PII</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  h2 { color: #34495e; margin-top: 40px; }
  .summary-table { width: 100%; border-collapse: collapse; margin: 20px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  .summary-table th { background: #2c3e50; color: white; padding: 12px 16px; text-align: left; }
  .summary-table td { padding: 10px 16px; border-bottom: 1px solid #eee; }
  .summary-table tr:last-child td { border-bottom: none; }
  .good { color: #27ae60; font-weight: bold; }
  .bad { color: #e74c3c; font-weight: bold; }
  .mid { color: #e67e22; font-weight: bold; }
  .test-card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
  .test-card h3 { color: #2c3e50; margin-top: 0; border-left: 4px solid #3498db; padding-left: 10px; }
  .provider-block { margin: 12px 0; padding: 12px; border-radius: 6px; background: #fafafa; border: 1px solid #e0e0e0; }
  .provider-title { font-weight: bold; color: #555; margin-bottom: 6px; }
  .text-display { line-height: 1.8; font-size: 14px; padding: 8px; background: white; border-radius: 4px; }
  .metrics-inline { font-size: 12px; color: #777; margin-top: 6px; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; }
  .legend-item { padding: 4px 10px; border-radius: 12px; color: white; font-size: 12px; }
</style>
</head>
<body>
<h1>Rapport d'analyse — Anonymisation PII</h1>
"""

def color_score(val, higher_better=True):
    if val is None:
        return "—"
    if higher_better:
        cls = "good" if val >= 0.7 else ("mid" if val >= 0.4 else "bad")
    else:
        cls = "good" if val <= 0.2 else ("mid" if val <= 0.5 else "bad")
    return f'<span class="{cls}">{val:.3f}</span>'


def main():
    print(f"Chargement de {SAMPLE_SIZE} exemples...")
    df = pd.read_csv("data/french_pii_sample.csv").head(SAMPLE_SIZE)
    df["privacy_mask"] = df["privacy_mask"].apply(ast.literal_eval)

    providers_cfg = [
        ("Presidio", presidio_anonymize, "#3498db"),
        ("Privacy Filter", privacy_filter_anonymize, "#e74c3c"),
        ("OpenMED", openmed_anonymize, "#27ae60"),
    ]

    # Collecter tous les résultats
    all_results = {name: [] for name, _, _ in providers_cfg}
    all_data = []

    for idx, row in df.iterrows():
        sample = {"source_text": row["source_text"], "privacy_mask": row["privacy_mask"], "providers": {}}
        for name, fn, color in providers_cfg:
            try:
                t0 = time.time()
                result = fn(row["source_text"])
                latency = time.time() - t0
                anonymized = result["anonymized"]
                entities = result.get("entities", [])

                metrics = compute_metrics(row["source_text"], anonymized, row["privacy_mask"], entities)
                metrics["latency_s"] = round(latency, 2)

                sample["providers"][name] = {
                    "anonymized": anonymized,
                    "entities": entities,
                    "metrics": metrics,
                    "color": color,
                }
                all_results[name].append(metrics)
            except Exception as e:
                sample["providers"][name] = {"error": str(e), "color": color}
                all_results[name].append({"error": str(e)})
            print(f"  [{idx+1}/{SAMPLE_SIZE}] {name}", end="\r")
        all_data.append(sample)
    print()

    # --- Résumé terminal ---
    print("\n" + "=" * 65)
    print(f"{'PROVIDER':<25} {'Precision':>9} {'Recall':>9} {'F1':>7} {'Leakage':>9} {'Lat(s)':>7}")
    print("=" * 65)
    for name, _, _ in providers_cfg:
        valid = [r for r in all_results[name] if r.get("error") is None and r.get("precision") is not None]
        if not valid:
            print(f"{name:<25} {'N/A':>9}")
            continue
        p  = sum(r["precision"] for r in valid) / len(valid)
        r  = sum(r["recall"] for r in valid) / len(valid)
        f1 = sum(r["f1"] for r in valid) / len(valid)
        lr = sum(r["leakage_rate"] for r in valid) / len(valid)
        lat = sum(r["latency_s"] for r in valid) / len(valid)
        print(f"{name:<25} {p:>9.3f} {r:>9.3f} {f1:>7.3f} {lr:>9.3f} {lat:>7.2f}s")
    print("=" * 65)

    # --- HTML ---
    html = [HTML_HEAD]

    # Légende couleurs
    html.append('<div class="legend">')
    for label, color in ENTITY_COLORS.items():
        html.append(f'<span class="legend-item" style="background:{color}">{label}</span>')
    html.append('</div>')

    # Tableau résumé
    html.append('<h2>Résumé des métriques</h2>')
    html.append('<table class="summary-table"><tr><th>Provider</th><th>Precision</th><th>Recall</th><th>F1</th><th>Leakage Rate ↓</th><th>Latence</th></tr>')
    for name, _, _ in providers_cfg:
        valid = [r for r in all_results[name] if r.get("error") is None and r.get("precision") is not None]
        if not valid:
            html.append(f'<tr><td><b>{name}</b></td><td colspan="5">Erreur / non configuré</td></tr>')
            continue
        p  = sum(r["precision"] for r in valid) / len(valid)
        r  = sum(r["recall"] for r in valid) / len(valid)
        f1 = sum(r["f1"] for r in valid) / len(valid)
        lr = sum(r["leakage_rate"] for r in valid) / len(valid)
        lat = sum(r["latency_s"] for r in valid) / len(valid)
        html.append(f'<tr><td><b>{name}</b></td><td>{color_score(p)}</td><td>{color_score(r)}</td>'
                    f'<td>{color_score(f1)}</td><td>{color_score(lr, higher_better=False)}</td>'
                    f'<td>{lat:.2f}s</td></tr>')
    html.append('</table>')

    # Détail par test
    html.append('<h2>Analyse par exemple</h2>')
    for i, sample in enumerate(all_data):
        html.append(f'<div class="test-card">')
        html.append(f'<h3>Test #{i+1}</h3>')

        # Texte original avec vérité terrain
        html.append('<div class="provider-block">')
        html.append('<div class="provider-title">Texte original (PII surlignées = vérité terrain)</div>')
        html.append(f'<div class="text-display">{highlight_ground_truth(sample["source_text"], sample["privacy_mask"])}</div>')
        html.append('</div>')

        # Résultats par provider
        for name, _, pcolor in providers_cfg:
            pdata = sample["providers"].get(name, {})
            html.append(f'<div class="provider-block" style="border-left: 4px solid {pcolor};">')
            html.append(f'<div class="provider-title" style="color:{pcolor};">{name}</div>')
            if "error" in pdata:
                html.append(f'<div style="color:red">Erreur : {pdata["error"][:200]}</div>')
            else:
                ents = pdata.get("entities", [])
                anon_text = pdata.get("anonymized", "")
                # Afficher texte anonymisé avec entités surlignées (via entités récupérées)
                highlighted = highlight_text(sample["source_text"], ents)
                html.append(f'<div class="text-display">{highlighted}</div>')
                m = pdata.get("metrics", {})
                p_str = f"{m['precision']:.3f}" if m.get("precision") is not None else "—"
                r_str = f"{m['recall']:.3f}" if m.get("recall") is not None else "—"
                f_str = f"{m['f1']:.3f}" if m.get("f1") is not None else "—"
                lr_str = f"{m['leakage_rate']:.3f}" if m.get("leakage_rate") is not None else "—"
                html.append(f'<div class="metrics-inline">Precision: <b>{p_str}</b> | Recall: <b>{r_str}</b> | F1: <b>{f_str}</b> | Leakage: <b>{lr_str}</b> | {m.get("latency_s","—")}s</div>')
            html.append('</div>')

        html.append('</div>')

    html.append('</body></html>')

    out_path = "data/rapport_analyse.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"\nRapport HTML sauvegardé : {out_path}")

    # CSV détaillé
    rows = []
    for i, sample in enumerate(all_data):
        for name, _, _ in providers_cfg:
            pdata = sample["providers"].get(name, {})
            m = pdata.get("metrics", {})
            rows.append({"test": i+1, "provider": name, "source_text": sample["source_text"][:80], **m})
    pd.DataFrame(rows).to_csv("data/evaluation_results.csv", index=False)
    print(f"CSV sauvegardé : data/evaluation_results.csv")


if __name__ == "__main__":
    main()
