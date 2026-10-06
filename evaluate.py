"""Évaluation locale — Precision, Recall, F1, Leakage par provider et par label."""
import argparse
import ast
import time
from collections import defaultdict

import pandas as pd
from assertions.metrics import compute_metrics, compute_per_label
from providers.presidio_provider import anonymize as presidio_anonymize
from providers.privacy_filter_local_provider import anonymize as privacy_filter_local_anonymize
from providers.openmed_local_provider import anonymize as openmed_local_anonymize
from providers.gliner_local_provider import anonymize as gliner_local_anonymize
from providers.gliner_fr_provider import anonymize as gliner_fr_anonymize
from providers.gliner2_provider import anonymize as gliner2_anonymize
from providers.ensemble_provider import DEFAULT_MEMBERS, STRATEGIES, anonymize_with

DATA_FILE = "data/french_pii_test.csv"


def load_samples(sample_size: int | None) -> pd.DataFrame:
    # french_pii_test.csv vient du split "validation" officiel d'ai4privacy/pii-masking-openpii-1m
    # (jamais téléchargé pendant l'entraînement du GLiNER fine-tuné, contrairement à
    # french_pii_sample.csv qui vient du split "train" et servait aussi de validation pendant
    # le training — cf. README, section Fine-tuning).
    df = pd.read_csv(DATA_FILE)
    if sample_size is not None:
        df = df.head(sample_size)
    df["privacy_mask"] = df["privacy_mask"].apply(ast.literal_eval)
    return df


def cached(fn):
    """Mémorise les résultats par texte : les croisements réutilisent les sorties des
    détecteurs déjà évalués au lieu de les recalculer."""
    cache = {}

    def wrapper(text: str, custom_instruction: str = "") -> dict:
        if text not in cache:
            cache[text] = fn(text)
        return cache[text]

    return wrapper


def run_provider(name: str, fn, df: pd.DataFrame):
    global_results = []
    label_accum = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        try:
            t0 = time.time()
            result = fn(row["source_text"])
            latency = time.time() - t0

            m = compute_metrics(row["source_text"], result["anonymized"], row["privacy_mask"])
            m["latency_s"] = round(latency, 2)
            global_results.append(m)

            per_label = compute_per_label(row["source_text"], result["anonymized"], row["privacy_mask"])
            for label, lm in per_label.items():
                label_accum[label]["recall"].append(lm["recall"])
                label_accum[label]["leakage"].append(lm["leakage"])
                label_accum[label]["support"].append(lm["support"])

        except Exception as e:
            global_results.append({"error": str(e)})

    return global_results, label_accum


def print_global(name: str, results: list):
    valid = [r for r in results if "error" not in r and r.get("recall") is not None]
    if not valid:
        print(f"  {name}: aucun résultat valide")
        return
    p  = sum(r["precision"] for r in valid) / len(valid)
    r  = sum(r["recall"] for r in valid) / len(valid)
    f1 = sum(r["f1"] for r in valid) / len(valid)
    lr = sum(r["leakage_rate"] for r in valid) / len(valid)
    lat = sum(r["latency_s"] for r in valid) / len(valid)
    print(f"  {name:<25} {p:>9.3f} {r:>9.3f} {f1:>7.3f} {lr:>9.3f} {lat:>7.2f}s")


def print_per_label(name: str, label_accum: dict):
    print(f"\n  {name} — par type d'entité :")
    print(f"  {'LABEL':<18} {'Recall':>9} {'Leakage':>9} {'Support':>8}")
    print(f"  {'-'*48}")
    rows = []
    for label, accum in label_accum.items():
        recall  = sum(accum["recall"]) / len(accum["recall"])
        leakage = sum(accum["leakage"]) / len(accum["leakage"])
        support = sum(accum["support"])
        rows.append((label, recall, leakage, support))
    for label, recall, leakage, support in sorted(rows, key=lambda x: -x[3]):
        print(f"  {label:<18} {recall:>9.3f} {leakage:>9.3f} {support:>8}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sample-size", "-n", type=int, default=None,
        help=f"Nombre d'exemples à évaluer depuis {DATA_FILE} (par défaut : tous, sans limite).",
    )
    p.add_argument(
        "--ensemble-members", default=",".join(DEFAULT_MEMBERS),
        help="Détecteurs combinés dans les croisements, séparés par des virgules "
             "(parmi gliner_fr, gliner2, privacy_filter, openmed, gliner, presidio).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = load_samples(args.sample_size)
    print(f"Chargement de {len(df)} exemples depuis {DATA_FILE}...\n")

    members = {
        "privacy_filter": cached(privacy_filter_local_anonymize),
        "openmed":        cached(openmed_local_anonymize),
        "gliner":         cached(gliner_local_anonymize),
        "gliner_fr":      cached(gliner_fr_anonymize),
        "gliner2":        cached(gliner2_anonymize),
        "presidio":       cached(presidio_anonymize),
    }
    ensemble_members = {n: members[n] for n in args.ensemble_members.split(",")}

    providers = [
        ("Privacy Filter", members["privacy_filter"]),
        ("OpenMed",        members["openmed"]),
        ("GLiNER",         members["gliner"]),
        ("GLiNER (fine-tuné FR)", members["gliner_fr"]),
        ("GLiNER2",        members["gliner2"]),
        ("Presidio",       members["presidio"]),
    ] + [
        (f"Croisement {strategy}", lambda text, s=strategy: anonymize_with(text, ensemble_members, s))
        for strategy in STRATEGIES
    ]

    print("=" * 65)
    print(f"  {'PROVIDER':<25} {'Precision':>9} {'Recall':>9} {'F1':>7} {'Leakage':>9} {'Lat(s)':>7}")
    print("=" * 65)

    all_label_data = {}
    csv_rows = []

    for name, fn in providers:
        results, label_accum = run_provider(name, fn, df)
        print_global(name, results)
        all_label_data[name] = label_accum

        for r in results:
            if "error" not in r:
                csv_rows.append({"provider": name, **r})

    print("=" * 65)
    print("  Recall : % PII masquées | Leakage : % PII ayant fui (0 = parfait)")
    print(f"  Croisements sur : {', '.join(ensemble_members)} — latence affichée hors détecteurs")
    print("  (sorties mises en cache) : la latence réelle ≈ somme des latences des détecteurs.")

    for name in all_label_data:
        print_per_label(name, all_label_data[name])

    pd.DataFrame(csv_rows).to_csv("data/evaluation_results.csv", index=False)
    print(f"\nCSV sauvegardé : data/evaluation_results.csv")


if __name__ == "__main__":
    main()
