"""CLI : python -m anonym_audio entree.wav sortie.wav [options]"""
import argparse
import importlib
from types import SimpleNamespace

from .pipeline import redact_audio_pii

PROVIDERS = {
    "presidio": "providers.presidio_provider",
    "gliner": "providers.gliner_local_provider",
    "gliner_fr": "providers.gliner_fr_provider",
    "gliner2": "providers.gliner2_provider",
    "openmed": "providers.openmed_local_provider",
    "privacy_filter": "providers.privacy_filter_local_provider",
    "ensemble": "providers.ensemble_provider",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Redaction de PII vocale (Whisper + détection PII texte)")
    parser.add_argument("audio_in", help="Fichier audio source")
    parser.add_argument("audio_out", help="Fichier audio rédigé en sortie")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="presidio")
    parser.add_argument("--mode", choices=["beep", "silence"], default="beep")
    parser.add_argument("--padding", type=float, default=0.15, help="Marge de sécurité en secondes autour de chaque plage PII")
    parser.add_argument("--language", default="fr")
    parser.add_argument("--entities", default="", help="Filtre d'entités, ex: PERSON,PHONE_NUMBER")
    parser.add_argument("--members", default="gliner_fr,gliner2,privacy_filter",
                        help="--provider ensemble : détecteurs combinés, séparés par des virgules")
    parser.add_argument("--strategy", choices=["union", "vote", "pondere"], default="vote",
                        help="--provider ensemble : méthode de combinaison")
    args = parser.parse_args()

    provider = importlib.import_module(PROVIDERS[args.provider])
    if args.provider == "ensemble":
        provider = SimpleNamespace(anonymize=provider.make_anonymize(args.members.split(","), args.strategy))

    report = redact_audio_pii(
        args.audio_in,
        args.audio_out,
        provider,
        mode=args.mode,
        padding_s=args.padding,
        custom_instruction=args.entities,
        language=args.language,
    )

    print(f"Audio rédigé -> {report['output_path']}")
    print(f"{len(report['audio_spans'])} passage(s) masqué(s)")
    for span in report["audio_spans"]:
        print(f"  [{span['start']:.2f}s - {span['end']:.2f}s] {span['label']} ({span['text']!r})")


if __name__ == "__main__":
    main()
