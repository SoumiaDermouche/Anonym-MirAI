"""Transcription audio via la gateway interne (compatible API Whisper), avec
timestamps mot-à-mot. Auth par header `Authorization: Bearer <token>`.
"""
import os

import requests

WHISPER_ENDPOINT = os.environ.get("WHISPER_ENDPOINT", "").rstrip("/")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper")


def transcribe(audio_path: str, language: str = "fr") -> dict:
    """Transcrit `audio_path` et retourne {"text": str, "words": [{"word","start","end"}]}.

    Les timestamps mot-à-mot restent approximatifs (cf. état de l'art) —
    anonym_audio.pii_timestamps applique une marge de sécurité (padding)
    autour de chaque span détecté pour compenser.
    """
    api_key = os.environ.get("WHISPER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Aucune clé trouvée : définis WHISPER_API_KEY dans .env")
    if not WHISPER_ENDPOINT:
        raise RuntimeError("WHISPER_ENDPOINT manquant dans .env")

    url = f"{WHISPER_ENDPOINT}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": WHISPER_MODEL,
        "language": language,
        "response_format": "verbose_json",
        # La gateway interne ne renvoie les timestamps mot-à-mot qu'avec la forme
        # scalaire `timestamp_granularities=word` — la forme tableau d'OpenAI
        # (`timestamp_granularities[]=word`) est acceptée mais ignorée.
        "timestamp_granularities": "word",
    }

    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f)}
        response = requests.post(url, headers=headers, data=data, files=files, timeout=300)
    response.raise_for_status()
    payload = response.json()

    # OpenAI expose `words` au niveau racine ; la gateway interne les imbrique
    # dans chaque segment (`segments[].words`). On gère les deux.
    words_raw = list(payload.get("words") or [])
    if not words_raw:
        for seg in payload.get("segments") or []:
            words_raw.extend(seg.get("words") or [])
    words = [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in words_raw]

    if not words:
        raise RuntimeError(
            "La réponse de la gateway ne contient pas de timestamps mot-à-mot "
            "('words' vide ou absent) — vérifie que le modèle/la gateway "
            "supportent bien timestamp_granularities=word."
        )

    return {"text": payload.get("text", ""), "words": words}
