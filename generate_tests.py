"""Génère tests.yaml pour promptfoo avec privacy_mask inclus dans les vars."""
import ast
import json
import pandas as pd
import yaml

df = pd.read_csv("data/french_pii_sample.csv")

tests = []
for _, row in df.head(50).iterrows():
    privacy_mask = ast.literal_eval(row["privacy_mask"]) if isinstance(row["privacy_mask"], str) else []
    tests.append({
        "vars": {
            "text": row["source_text"],
            "privacy_mask_json": json.dumps(privacy_mask, ensure_ascii=False),
        },
        "assert": [
            {
                "type": "python",
                "value": "file://assertions/anonymization_assert.py",
            }
        ],
    })

with open("tests.yaml", "w", encoding="utf-8") as f:
    yaml.dump(tests, f, allow_unicode=True, default_flow_style=False)

print(f"{len(tests)} tests générés dans tests.yaml")
