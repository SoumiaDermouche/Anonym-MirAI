"""Convertit le subset français d'ai4privacy/pii-masking-openpii-1m au format
d'entraînement GLiNER : {"tokenized_text": [...], "ner": [[start_tok, end_tok, label], ...]}.

Les spans du dataset (privacy_mask) sont en offsets caractères ; on les réaligne
sur les tokens produits par le même WhitespaceTokenSplitter que celui utilisé
par GLiNER à l'inférence (gliner.data_processing.tokenizer.WordsSplitter),
pour garantir la cohérence train/inférence.

Usage :
    python training/prepare_gliner_data.py
"""
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from gliner.data_processing.tokenizer import WordsSplitter

PARQUET_DIR = Path(__file__).parent.parent / "data" / "ai4privacy_parquet"
OUT_DIR = Path(__file__).parent.parent / "data"

# Description en langage naturel de chacun des 19 labels ai4privacy — GLiNER est
# un modèle zero-shot conditionné par le texte du label, donc on privilégie des
# phrases descriptives plutôt que les codes bruts du dataset.
LABEL_TO_TEXT = {
    "TITLE": "title",
    "GIVENNAME": "given name",
    "SURNAME": "surname",
    "DATE": "date",
    "CITY": "city",
    "STREET": "street",
    "BUILDINGNUM": "building number",
    "ZIPCODE": "zip code",
    "AGE": "age",
    "SEX": "sex",
    "GENDER": "gender",
    "EMAIL": "email address",
    "TELEPHONENUM": "phone number",
    "CREDITCARDNUMBER": "credit card number",
    "IDCARDNUM": "national id card number",
    "PASSPORTNUM": "passport number",
    "DRIVERLICENSENUM": "driver license number",
    "TAXNUM": "tax identification number",
    "SOCIALNUM": "social security number",
}

_splitter = WordsSplitter(splitter_type="whitespace")


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return list(_splitter(text))


def char_span_to_token_span(tokens: list[tuple[str, int, int]], start: int, end: int) -> tuple[int, int] | None:
    """Convertit un span caractère (start, end) en span token [start_idx, end_idx] inclusif."""
    tok_start = tok_end = None
    for i, (_, t_start, t_end) in enumerate(tokens):
        if t_end > start and t_start < end:  # chevauchement
            if tok_start is None:
                tok_start = i
            tok_end = i
    if tok_start is None:
        return None
    return tok_start, tok_end


def convert_row(row) -> dict | None:
    text = row["source_text"]
    tokens = tokenize(text)
    if not tokens:
        return None

    ner = []
    for ent in row["privacy_mask"]:
        label = LABEL_TO_TEXT.get(ent["label"])
        if label is None:
            continue
        span = char_span_to_token_span(tokens, ent["start"], ent["end"])
        if span is None:
            continue
        ner.append([span[0], span[1], label])

    if not ner:
        return None

    return {
        "tokenized_text": [t[0] for t in tokens],
        "ner": ner,
    }


def load_french_rows() -> list:
    rows = []
    for path in sorted(PARQUET_DIR.glob("*.parquet")):
        try:
            table = pq.read_table(path, columns=["source_text", "privacy_mask", "language"])
        except Exception as e:
            print(f"  skip {path.name} (unreadable, still downloading?): {e}")
            continue
        df = table.to_pandas()
        fr = df[df["language"] == "fr"]
        if len(fr):
            print(f"  {path.name}: {len(fr)} exemples fr")
            rows.extend(fr.to_dict("records"))
    return rows


def main():
    print("Chargement des lignes françaises depuis les shards parquet...")
    rows = load_french_rows()
    print(f"Total lignes fr brutes : {len(rows)}")

    examples = []
    for row in rows:
        ex = convert_row(row)
        if ex is not None:
            examples.append(ex)
    print(f"Total exemples convertis (avec >=1 entité) : {len(examples)}")

    random.Random(42).shuffle(examples)

    # Split à 3 voies, tenues à l'écart les unes des autres :
    #   - train      : utilisé pour la descente de gradient
    #   - validation : utilisé PENDANT l'entraînement (eval_loss/eval_steps) — jamais pour la
    #                  backprop, mais "vu" par le pipeline d'entraînement
    #   - test       : jamais vu sous aucune forme avant l'entraînement — réservé exclusivement
    #                  au reporting final (evaluate.py / comparaison entre providers)
    # IMPORTANT : la frontière train/holdout (2%, min 200) doit rester strictement identique
    # à l'ancien split eval — sinon des exemples déjà entraînés basculeraient dans le test.
    n_holdout = max(200, int(0.02 * len(examples)))
    holdout, train_examples = examples[:n_holdout], examples[n_holdout:]

    random.Random(123).shuffle(holdout)  # mélange indépendant, seed différente
    half = len(holdout) // 2
    val_examples = holdout[:half]
    test_examples = holdout[half:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "gliner_train_fr_full.json", "w", encoding="utf-8") as f:
        json.dump(train_examples, f, ensure_ascii=False)
    with open(OUT_DIR / "gliner_eval_fr.json", "w", encoding="utf-8") as f:
        json.dump(val_examples, f, ensure_ascii=False)
    with open(OUT_DIR / "gliner_test_fr.json", "w", encoding="utf-8") as f:
        json.dump(test_examples, f, ensure_ascii=False)

    sample_train = train_examples[:300]
    sample_eval = val_examples[:50]
    with open(OUT_DIR / "gliner_train_fr_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_train, f, ensure_ascii=False)
    with open(OUT_DIR / "gliner_eval_fr_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_eval, f, ensure_ascii=False)

    print(f"train complet      : {len(train_examples)} -> gliner_train_fr_full.json")
    print(f"validation complet : {len(val_examples)} -> gliner_eval_fr.json (utilisé pendant l'entraînement)")
    print(f"test complet       : {len(test_examples)} -> gliner_test_fr.json (réservé au reporting final)")
    print(f"train sample       : {len(sample_train)} -> gliner_train_fr_sample.json")
    print(f"eval sample        : {len(sample_eval)} -> gliner_eval_fr_sample.json")
    print("Exemple :", json.dumps(train_examples[0], ensure_ascii=False, indent=2)[:600])


if __name__ == "__main__":
    main()
