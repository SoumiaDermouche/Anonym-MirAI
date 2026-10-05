# Anonym-MirAI

Comparaison de 4 outils d'anonymisation de données PII sur textes français, avec déploiement local via Streamlit.

## Outils comparés

Résultats de `evaluate.py` (200 exemples, `data/french_pii_test.csv` — tiré du split **`validation`
officiel** d'ai4privacy/pii-masking-openpii-1m, jamais téléchargé ni vu sous aucune forme par le
pipeline de fine-tuning de GLiNER, qui n'utilise que le split `train`) :

| Modèle | Type | Precision | Recall | F1 | Leakage | Vitesse |
|---|---|---|---|---|---|---|
| **GLiNER (fine-tuné FR)** | Local (fine-tuné sur ai4privacy) | 0.975 | 0.983 | 0.977 | 1.7% | 0.047s |
| **OpenMed** | Local | 0.957 | 0.860 | 0.894 | 14.0% | 0.917s |
| **Privacy Filter** | Local | 0.938 | 0.840 | 0.870 | 16.0% | 0.338s |
| **GLiNER** | Local (zero-shot) | 0.881 | 0.764 | 0.801 | 23.6% | 0.054s |
| **Presidio** | Règles/NER local | 0.866 | 0.544 | 0.642 | 45.6% | 0.042s |

Tous les modèles tournent en local (GPU si disponible) — aucun appel réseau, aucune clé API requise.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_lg
```

## Lancer l'application Streamlit

```bash
cd /home/onyxia/work/Anonym-MirAI
streamlit run app.py
```

L'utilisateur dépose un PDF → choisit le modèle dans la sidebar → télécharge le PDF anonymisé.

## Anonymisation audio (PII vocale)

Le package `anonym_audio/` masque les données sensibles **parlées** dans un fichier audio :
Whisper transcrit l'audio avec timestamps mot-à-mot → un détecteur PII texte (les mêmes
providers que ci-dessus) repère les entités → les passages correspondants sont remplacés
par un bip ou un silence dans l'audio d'origine.

### Prérequis

```bash
pip install -r requirements.txt        # inclut pydub + audioop-lts (Python ≥ 3.13)
sudo apt-get install -y ffmpeg          # binaire système requis par pydub
```

La transcription passe par la gateway interne (pas l'API OpenAI publique). Les variables
suivantes doivent être présentes dans `.env` (déjà configurées ici) :

```
WHISPER_ENDPOINT=https://gateway.api-hp.ai.numerique-interieur.com/v1
WHISPER_API_KEY=...          # jeton Bearer OAuth de la gateway (expire — à renouveler)
WHISPER_MODEL=whisper        # modèles dispo : whisper, whisper-diarization
```

### Interface Streamlit

```bash
cd /home/onyxia/work/Anonym-MirAI
streamlit run audio_app.py
```

L'utilisateur dépose un audio (mp3, wav, m4a, ogg, flac, webm) → choisit le détecteur, le
mode (bip / silence) et la marge de sécurité dans la sidebar → écoute et télécharge l'audio
anonymisé, avec la liste des passages masqués et la transcription.

### Ligne de commande

```bash
python -m anonym_audio entree.wav sortie.wav --provider presidio --mode beep
```

Options : `--provider {presidio,gliner,gliner_fr,gliner2,openmed,privacy_filter}`, `--mode {beep,silence}`,
`--padding` (marge en secondes autour de chaque plage, défaut `0.15`),
`--language` (défaut `fr`), `--entities` (filtre, ex. `PERSON,PHONE_NUMBER`).

## Évaluation

### 1. Charger le dataset français (à faire une seule fois)

```bash
python data/load_data.py
```

Charge 200 exemples francophones depuis `ai4privacy/pii-masking-openpii-1m` (HuggingFace, split `train`)
dans `data/french_pii_sample.csv` — utilisé par `generate_report.py` et `generate_tests.py`.

`evaluate.py` (comparaison des modèles) utilise en revanche `data/french_pii_test.csv`, généré à part
depuis le split **`validation`** officiel du dataset — jamais touché par le fine-tuning de GLiNER
(qui n'utilise que le split `train`), pour un comparatif rigoureux entre modèles. Voir la section
Fine-tuning ci-dessous pour la logique complète des splits train/validation/test.

### 2. Évaluation locale — métriques par provider et par label

```bash
python evaluate.py
```

Affiche dans le terminal : Precision, Recall, F1, Leakage Rate global et par type d'entité (PERSON, DATE, EMAIL, PHONE…).
Sauvegarde les résultats dans `data/evaluation_results.csv`.

### 3. Rapport HTML — visualisation des entités colorées

```bash
python generate_report.py
```

Génère `data/rapport_analyse.html` — pour chaque exemple : texte original avec PII surlignées, résultats de chaque provider avec entités colorées par type.

### 4. Évaluation via promptfoo — dashboard interactif

```bash
# Activer Node.js (une fois par session)
export PATH="$HOME/.local/share/fnm:$PATH"
eval "$(fnm env)"

# Charger les variables d'environnement
export $(grep -v '^#' .env | xargs)

# Générer les tests
python generate_tests.py

# Lancer l'évaluation
promptfoo eval --config promptfoo.yaml

# Voir le dashboard (localhost)
promptfoo view
```

Le score promptfoo = Recall par test (seuil passage : 0.70). Les assertions utilisent `assertions/anonymization_assert.py` — même méthode diff pour tous les providers.

## Structure du projet

```
Anonym-MirAI/
├── app.py                        # Interface Streamlit (PDF)
├── audio_app.py                  # Interface Streamlit (audio / PII vocale)
├── anonym_audio/                 # Pipeline audio : transcription + détection + masquage
├── pdf_processor.py              # Extraction et reconstruction PDF
├── evaluate.py                   # Évaluation locale par provider et label
├── generate_report.py            # Rapport HTML avec entités colorées
├── generate_tests.py             # Génère tests.yaml pour promptfoo
├── promptfoo.yaml                # Config comparaison promptfoo
├── requirements.txt
├── .env                          # Clés API (ne pas committer)
├── assertions/
│   ├── metrics.py                # Méthode diff unifiée (precision/recall/F1/leakage)
│   └── anonymization_assert.py  # Assertion custom promptfoo
├── providers/
│   ├── presidio_provider.py            # Microsoft Presidio (local)
│   ├── openmed_local_provider.py       # OpenMed (local)
│   ├── gliner_local_provider.py        # GLiNER zero-shot (local)
│   ├── gliner_fr_provider.py           # GLiNER fine-tuné FR (local)
│   ├── gliner2_provider.py             # GLiNER2 PII multilingue (local, package gliner2)
│   ├── privacy_filter_local_provider.py # Privacy Filter (local)
│   └── presidio_wrapper.py             # Wrapper pour promptfoo
└── data/
    ├── load_data.py              # Chargement dataset HuggingFace
    ├── french_pii_sample.csv     # 200 exemples FR (généré)
    ├── evaluation_results.csv    # Résultats évaluation (généré)
    └── rapport_analyse.html      # Rapport HTML (généré)
```

## Métriques

- **Recall** : % de PII du ground truth absentes du texte anonymisé *(le plus important pour la confidentialité)*
- **Precision** : % des entités masquées qui correspondent à de vraies PII
- **F1** : moyenne harmonique precision/recall
- **Leakage Rate** : % de PII ayant fui (0 = parfait, 1 = aucune anonymisation)

Toutes les métriques utilisent la même méthode : diff caractère par caractère entre texte original et anonymisé, indépendamment de ce que chaque provider rapporte.

## Visualisation des entités

L'affichage coloré des entités dans l'interface Streamlit s'inspire de deux références :

- **spaCy displacy** — visualiseur NER standard avec surlignage coloré par type d'entité et étiquettes superposées.
  Référence : https://spacy.io/usage/visualizers

- **Whisper word-level confidence** (Xenova, HuggingFace Spaces) — coloration des mots transcrits selon leur score de confiance (probabilité log).
  Référence : https://huggingface.co/spaces/Xenova/whisper-word-level-timestamps

**Code couleur appliqué :**
| Couleur | Seuil de confiance | Signification |
|---|---|---|
| 🟢 Vert | score ≥ 0.80 | Prédiction fiable |
| 🟡 Jaune | 0.60 ≤ score < 0.80 | Confiance moyenne |
| 🔴 Rouge | score < 0.60 | Prédiction incertaine |

Tous les modèles intégrés tournent en local et exposent position + score par entité :
toutes les entités détectées sont donc surlignées avec une des 3 couleurs ci-dessus.

## Fine-tuning de GLiNER sur ai4privacy (français)

### 1. Préparer les données d'entraînement

```bash
python training/prepare_gliner_data.py
```

Télécharge/filtre le subset français du split **`train`** d'`ai4privacy/pii-masking-openpii-1m`
(le dataset n'expose que 2 splits : `train` et `validation`, pas de `test` — parquet dans
`data/ai4privacy_parquet/`) et génère un split à 3 voies, toutes tenues à l'écart les unes des
autres (seed=42 puis seed=123 pour le sous-partage) :
- `data/gliner_train_fr_full.json` — **train**, utilisé pour la descente de gradient
- `data/gliner_eval_fr.json` — **validation**, utilisé pendant l'entraînement (`eval_loss`/`eval_steps`)
- `data/gliner_test_fr.json` — **test**, jamais vu pendant l'entraînement, réservé au reporting futur
- `data/gliner_train_fr_sample.json` / `data/gliner_eval_fr_sample.json` — petit échantillon (300/50 exemples) pour un test rapide

Pour la comparaison entre modèles (`evaluate.py`), on utilise plutôt `data/french_pii_test.csv`,
tiré du split `validation` **officiel** du dataset (jamais téléchargé par ce pipeline) — un test set
encore plus rigoureux qu'un simple holdout interne au split `train`. Voir section Évaluation.

### 2. Fine-tuning

Test rapide (sanity check du pipeline, ~20s sur GPU) :

```bash
python training/finetune_gliner.py \
  --train data/gliner_train_fr_sample.json \
  --eval data/gliner_eval_fr_sample.json \
  --output-dir models/gliner-fr-poc \
  --max-steps 60 --batch-size 4 --logging-steps 5 --save-steps 30 --eval-steps 30
```

Entraînement complet (103k exemples, ~2h30–4h sur GPU) :

```bash
python training/finetune_gliner.py \
  --train data/gliner_train_fr_full.json \
  --eval data/gliner_eval_fr.json \
  --output-dir models/gliner-fr-full \
  --num-epochs 3 --batch-size 8 --logging-steps 50 --save-steps 2000 --eval-steps 2000
```

Le modèle de base est `urchade/gliner_multi-v2.1` (même modèle que `providers/gliner_local_provider.py`).
Le script sauvegarde uniquement les poids (`save_only_model`, pas l'état de l'optimizer) et ne garde
qu'un seul checkpoint (`--save-total-limit 1`) pour économiser l'espace disque.
