"""Édition de l'audio source : remplace chaque plage PII détectée par un bip ou un silence."""
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine


def redact(audio_path: str, spans: list[dict], out_path: str, mode: str = "beep", beep_freq: int = 1000) -> str:
    """Remplace les plages `spans` ({"start","end"} en secondes) de `audio_path`
    par un bip (`mode="beep"`) ou un silence (`mode="silence"`), et exporte
    le résultat vers `out_path`. Retourne `out_path`.
    """
    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)

    for span in spans:
        start_ms = max(0, int(span["start"] * 1000))
        end_ms = min(duration_ms, int(span["end"] * 1000))
        if end_ms <= start_ms:
            continue

        seg_len = end_ms - start_ms
        if mode == "beep":
            replacement = Sine(beep_freq).to_audio_segment(duration=seg_len).apply_gain(-3)
        elif mode == "silence":
            replacement = AudioSegment.silent(duration=seg_len)
        else:
            raise ValueError(f"mode inconnu: {mode!r} (attendu: 'beep' ou 'silence')")

        replacement = replacement.set_frame_rate(audio.frame_rate).set_channels(audio.channels)
        audio = audio[:start_ms] + replacement + audio[end_ms:]

    out_format = Path(out_path).suffix.lstrip(".") or "wav"
    audio.export(out_path, format=out_format)
    return out_path
