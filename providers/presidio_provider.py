"""Provider Presidio — détection et anonymisation PII par règles/NER (sans LLM)."""
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine


def _build_analyzer() -> AnalyzerEngine:
    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "fr", "model_name": "fr_core_news_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=config)
    nlp_engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["fr"])


_analyzer = None
_anonymizer = AnonymizerEngine()


def anonymize(text: str, custom_instruction: str = "") -> dict:
    """Retourne le texte anonymisé et la liste des entités détectées."""
    global _analyzer
    if _analyzer is None:
        _analyzer = _build_analyzer()

    # Si l'instruction ressemble à une liste d'entités (ex: "PERSON, EMAIL"), on filtre
    entities_filter = None
    if custom_instruction.strip():
        tokens = [t.strip().upper() for t in custom_instruction.replace(";", ",").split(",") if t.strip()]
        if tokens and all(t.replace("_", "").isalpha() for t in tokens):
            entities_filter = tokens

    results = _analyzer.analyze(text=text, language="fr", entities=entities_filter)
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)

    entities = [
        {
            "text": text[r.start:r.end],
            "label": r.entity_type,
            "score": round(r.score, 3),
            "start": r.start,
            "end": r.end,
        }
        for r in results
    ]

    return {
        "provider": "presidio",
        "original": text,
        "anonymized": anonymized.text,
        "entities": entities,
    }


if __name__ == "__main__":
    sample = "Bonjour, je m'appelle Jean Dupont et mon email est jean.dupont@example.com."
    result = anonymize(sample)
    print("Texte anonymisé:", result["anonymized"])
    print("Entités:", result["entities"])
