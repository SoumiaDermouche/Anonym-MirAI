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
| **GLiNER2** | Local (`fastino/gliner2-privacy-filter-PII-multi`) | 0.907 | 0.792 | 0.832 | 20.8% | 0.05s |
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

Options : `--provider {presidio,gliner,gliner_fr,gliner2,openmed,privacy_filter,ensemble}`, `--mode {beep,silence}`,
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

Évalue aussi les **croisements** (cf. section dédiée) ; `--ensemble-members` choisit les détecteurs combinés :

```bash
python evaluate.py --ensemble-members gliner_fr,openmed,privacy_filter
```

## Croisement de détecteurs (ensemble)

`providers/ensemble_provider.py` combine plusieurs détecteurs, d'après Kim, Heider & Meystre,
*Ensemble-based Methods to Improve De-identification of Electronic Health Record Narratives*,
AMIA Annu Symp Proc 2018:663-672 (PMID 30815108). Le vote se fait **au niveau caractère** (les
détecteurs n'ont ni les mêmes labels ni les mêmes frontières d'entités) :

- `union` : masqué si au moins un détecteur le trouve (recall maximal) ;
- `vote` : masqué si au moins la moitié des détecteurs le trouvent ;
- `pondere` : vote pondéré par le F1 de chaque détecteur (`WEIGHTS`).

Disponible dans `app.py` et `audio_app.py` (« Croisement (ensemble) ») et en CLI :
`python -m anonym_audio in.wav out.wav --provider ensemble --members gliner_fr,gliner2,privacy_filter --strategy vote`.

Résultats (`evaluate.py`, 200 exemples `french_pii_test.csv`) — F1 :

| Détecteurs combinés | union | vote | pondere |
|---|---|---|---|
| gliner_fr, gliner2, privacy_filter (défaut) | 0.939 | 0.951 | 0.951 |
| gliner_fr, openmed, privacy_filter | 0.962 | 0.939 | 0.939 |
| les 6 détecteurs | 0.905 | 0.952 | 0.950 |
| *rappel : GLiNER (fine-tuné FR) seul* | | **0.977** | |

**Sur ce jeu de test, aucun croisement ne dépasse GLiNER fine-tuné FR seul.** Un détecteur domine
nettement les autres : l'union ajoute leurs faux positifs (precision ↓) sans gain de recall (0.983
déjà atteint), le vote fait perdre des PII que seul GLiNER FR trouve (recall ↓). Ce jeu de test vient
du même dataset (ai4privacy) que le fine-tuning, ce qui avantage GLiNER FR ; l'intérêt du croisement
(meilleure généralisation entre corpus, cf. Kim et al. AMIA 2020, PMID 33936439) reste à mesurer sur
des données d'une autre source (transcriptions audio réelles, documents internes).

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
│   ├── ensemble_provider.py            # Croisement de détecteurs (union / vote / vote pondéré)
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
