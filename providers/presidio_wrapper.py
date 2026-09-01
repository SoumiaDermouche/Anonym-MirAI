"""Wrapper promptfoo pour Presidio — lit le texte sur stdin, écrit le résultat sur stdout."""
import sys
import json
sys.path.insert(0, "/home/onyxia/work/Anonym-MirAI")
from providers.presidio_provider import anonymize

def call_api(prompt, options, context):
    result = anonymize(prompt)
    return {"output": result["anonymized"], "metadata": {"entities": result["entities"]}}
