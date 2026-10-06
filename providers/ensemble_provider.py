"""Provider « croisement » — ensemble de plusieurs détecteurs PII combinés par vote.

Méthodes inspirées de Kim, Heider & Meystre (AMIA 2018, « Ensemble-based Methods to
Improve De-identification of Electronic Health Record Narratives ») :
- union    : un caractère est masqué si au moins un détecteur le masque (recall max) ;
- vote     : masqué si au moins la moitié des détecteurs le masquent (vote majoritaire) ;
- pondere  : vote pondéré par le F1 de chaque détecteur, seuil à la moitié du poids total.

Le vote se fait au niveau caractère : les détecteurs n'ont pas la même taxonomie de
labels ni les mêmes frontières d'entités, mais tous renvoient des offsets start/end.
Le label d'une plage retenue est celui du détecteur le plus fiable (poids max) qui la couvre.
"""
import importlib
from typing import Callable

MEMBERS = {
    "gliner_fr": "providers.gliner_fr_provider",
    "gliner2": "providers.gliner2_provider",
    "privacy_filter": "providers.privacy_filter_local_provider",
    "openmed": "providers.openmed_local_provider",
    "gliner": "providers.gliner_local_provider",
    "presidio": "providers.presidio_provider",
}

# F1 mesurés par evaluate.py sur data/french_pii_test.csv (200 exemples, cf. readme).
WEIGHTS = {
    "gliner_fr": 0.977,
    "openmed": 0.894,
    "privacy_filter": 0.870,
    "gliner2": 0.832,
    "gliner": 0.801,
    "presidio": 0.642,
}

STRATEGIES = ("union", "vote", "pondere")
DEFAULT_MEMBERS = ("gliner_fr", "gliner2", "privacy_filter")
DEFAULT_STRATEGY = "vote"


def _load(name: str) -> Callable:
    return importlib.import_module(MEMBERS[name]).anonymize


def combine(text: str, member_results: dict[str, dict], strategy: str = DEFAULT_STRATEGY) -> dict:
    """Combine les sorties {nom_détecteur: résultat anonymize()} en un seul résultat."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Stratégie inconnue : {strategy!r} (attendu : {', '.join(STRATEGIES)})")
    if not member_results:
        raise ValueError("Le croisement nécessite au moins un détecteur.")

    names = list(member_results)
    weights = {n: WEIGHTS.get(n, 1.0) if strategy == "pondere" else 1.0 for n in names}
    total = sum(weights.values())
    threshold = 1e-9 if strategy == "union" else total / 2

    # Score de vote par caractère + entités de chaque détecteur pour l'étiquetage
    votes = [0.0] * len(text)
    for n in names:
        covered = [False] * len(text)
        for e in member_results[n]["entities"]:
            for i in range(max(0, e["start"]), min(len(text), e["end"])):
                covered[i] = True
        for i, c in enumerate(covered):
            if c:
                votes[i] += weights[n]

    # Plages contiguës de caractères retenus, débarrassées des espaces/ponctuation en bordure
    spans, i = [], 0
    while i < len(text):
        if votes[i] >= threshold:
            j = i
            while j < len(text) and votes[j] >= threshold:
                j += 1
            s, e = i, j
            while s < e and not text[s].isalnum():
                s += 1
            while e > s and not text[e - 1].isalnum():
                e -= 1
            if s < e:
                spans.append((s, e, max(votes[s:e]) / total))
            i = j
        else:
            i += 1

    by_priority = sorted(names, key=lambda n: -WEIGHTS.get(n, 0))
    entities = []
    for s, e, score in spans:
        sources = [n for n in names if any(x["start"] < e and x["end"] > s for x in member_results[n]["entities"])]
        for n in by_priority:
            hits = sorted(
                (max(x["start"], s), min(x["end"], e), str(x["label"]).upper().replace(" ", "_"))
                for x in member_results[n]["entities"] if x["start"] < e and x["end"] > s
            )
            if hits:
                break
        # Si le détecteur le plus fiable couvre toute la plage, on garde son découpage
        # (ex. GIVEN_NAME + SURNAME plutôt qu'un seul bloc) ; sinon une plage unique.
        covered = {i for hs, he, _ in hits for i in range(hs, he)}
        if hits and all(i in covered for i in range(s, e) if text[i].isalnum()):
            parts = hits
        else:
            parts = [(s, e, hits[0][2] if hits else "PII")]
        for ps, pe, label in parts:
            entities.append({"text": text[ps:pe], "label": label, "score": round(score, 3),
                             "start": ps, "end": pe, "sources": sources})

    anonymized = text
    for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
        anonymized = anonymized[: ent["start"]] + f"[{ent['label']}]" + anonymized[ent["end"]:]

    return {
        "provider": f"ensemble_{strategy}",
        "original": text,
        "anonymized": anonymized,
        "entities": entities,
    }


def anonymize_with(text: str, member_fns: dict[str, Callable], strategy: str = DEFAULT_STRATEGY,
                   custom_instruction: str = "") -> dict:
    """Exécute chaque détecteur puis combine. `member_fns` permet d'injecter des fonctions
    mises en cache (cf. evaluate.py) pour ne pas recalculer les détecteurs déjà évalués.

    `custom_instruction` est ignoré : chaque détecteur a son propre format de labels
    (Presidio : PERSON, GLiNER : texte libre...), les membres tournent donc avec leur
    liste par défaut."""
    results = {n: fn(text) for n, fn in member_fns.items()}
    return combine(text, results, strategy)


def make_anonymize(members=DEFAULT_MEMBERS, strategy: str = DEFAULT_STRATEGY) -> Callable:
    """Fonction anonymize(text, custom_instruction="") pour une combinaison donnée."""
    member_fns = {n: _load(n) for n in members}

    def anonymize(text: str, custom_instruction: str = "") -> dict:
        return anonymize_with(text, member_fns, strategy, custom_instruction)

    return anonymize


def anonymize(text: str, custom_instruction: str = "") -> dict:
    return make_anonymize()(text, custom_instruction)


if __name__ == "__main__":
    sample = "Bonjour, je m'appelle Jean Dupont, né le 12/03/1985. Mon email est jean.dupont@example.com et mon téléphone est 06 12 34 56 78."
    result = anonymize(sample)
    print("Texte anonymisé:", result["anonymized"])
    print("Entités:", result["entities"])
