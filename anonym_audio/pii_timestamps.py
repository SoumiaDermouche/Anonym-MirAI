"""Convertit les entités PII détectées dans le texte (offsets caractère) en
plages temporelles sur l'audio, à partir des mots alignés par anonym_audio.align.
"""


def entities_to_spans(entities: list[dict], aligned_words: list[dict], padding_s: float = 0.15) -> list[dict]:
    """Pour chaque entité PII (start/end en caractères), trouve les mots qui la
    recouvrent et déduit une plage temporelle [start, end] (secondes), avec
    une marge de sécurité `padding_s` de chaque côté. Les plages qui se
    chevauchent ou se touchent après padding sont fusionnées.
    """
    spans = []
    for ent in entities:
        e_start, e_end = ent["start"], ent["end"]
        matching = [w for w in aligned_words if w["end_char"] > e_start and w["start_char"] < e_end]
        if not matching:
            continue
        t_start = min(w["start_time"] for w in matching) - padding_s
        t_end = max(w["end_time"] for w in matching) + padding_s
        spans.append({"start": max(0.0, t_start), "end": t_end, "label": ent["label"], "text": ent["text"]})

    return _merge_spans(spans)


def _merge_spans(spans: list[dict]) -> list[dict]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: s["start"])
    merged = [dict(spans[0])]
    for s in spans[1:]:
        last = merged[-1]
        if s["start"] <= last["end"]:
            last["end"] = max(last["end"], s["end"])
            if s["label"] != last["label"]:
                last["label"] = "PII"
        else:
            merged.append(dict(s))
    return merged
