"""Pipeline complet : audio -> transcription Whisper -> détection PII -> audio rédigé.

Option A de l'état de l'art (voir readme du projet) : réutilise les providers
PII texte déjà en place (providers/presidio_provider.py, gliner_*) et n'ajoute
que l'étage de transcription/alignement temporel et l'étage d'édition audio.
"""
from pathlib import Path
from types import ModuleType

from dotenv import load_dotenv

from .align import align_words
from .pii_timestamps import entities_to_spans
from .redact_audio import redact
from .transcribe import transcribe

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def redact_audio_pii(
    audio_path: str,
    out_path: str,
    provider: ModuleType,
    mode: str = "beep",
    padding_s: float = 0.15,
    custom_instruction: str = "",
    language: str = "fr",
) -> dict:
    """Transcrit `audio_path`, détecte les PII dans la transcription via `provider`
    (un module exposant `anonymize(text, custom_instruction="") -> dict`, ex.
    providers.presidio_provider), puis rédige l'audio aux timestamps correspondants.

    Retourne un rapport {"transcript", "anonymized_transcript", "entities",
    "audio_spans", "output_path"} utile pour l'audit/le debug.
    """
    transcript = transcribe(audio_path, language=language)
    aligned_words = align_words(transcript["text"], transcript["words"])

    detection = provider.anonymize(transcript["text"], custom_instruction)
    spans = entities_to_spans(detection["entities"], aligned_words, padding_s=padding_s)

    redact(audio_path, spans, out_path, mode=mode)

    return {
        "transcript": transcript["text"],
        "anonymized_transcript": detection["anonymized"],
        "entities": detection["entities"],
        "audio_spans": spans,
        "output_path": out_path,
    }
