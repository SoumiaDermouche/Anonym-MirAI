"""Alignement des mots retournés par l'API Whisper sur les offsets caractère du texte transcrit."""


def align_words(text: str, words: list[dict]) -> list[dict]:
    """Associe à chaque mot ses offsets caractère dans `text` (recherche séquentielle).

    `words` : liste de {"word": str, "start": float, "end": float} (secondes),
    telle que renvoyée par l'API Whisper avec timestamp_granularities=["word"].
    """
    aligned = []
    cursor = 0
    for w in words:
        token = w["word"].strip()
        if not token:
            continue
        idx = text.find(token, cursor)
        if idx == -1:
            # Le mot n'a pas été retrouvé à partir du curseur (normalisation
            # différente entre `text` et `words`) : on garde son timing sans
            # offset fiable plutôt que de désaligner tous les mots suivants.
            idx = cursor
        start_char = idx
        end_char = idx + len(token)
        aligned.append(
            {
                "text": token,
                "start_char": start_char,
                "end_char": end_char,
                "start_time": w["start"],
                "end_time": w["end"],
            }
        )
        cursor = end_char
    return aligned
