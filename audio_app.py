"""Application Streamlit — anonymisation de PII vocale (audio)."""
import importlib
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

# Empêche transformers de charger TensorFlow (cf. app.py — même contrainte
# pour les providers PII texte locaux utilisés ici : GLiNER, Presidio...).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

import pandas as pd
import streamlit as st

from anonym_audio.pipeline import redact_audio_pii

st.set_page_config(page_title="Anonymisation audio", page_icon="🎙️", layout="centered")
st.title("Anonymisation de PII vocale")
st.caption(
    "Déposez un audio — Whisper le transcrit, un détecteur repère les données "
    "sensibles dans le texte, puis ces passages sont masqués (bip ou silence) "
    "dans l'audio d'origine, aux mêmes timestamps."
)

# --- Détecteurs PII texte disponibles (réutilise providers/) ---
PROVIDERS = {
    "Presidio": "providers.presidio_provider",
    "GLiNER": "providers.gliner_local_provider",
    "GLiNER (fine-tuné FR)": "providers.gliner_fr_provider",
    "GLiNER2 (PII multilingue)": "providers.gliner2_provider",
    "OpenMed": "providers.openmed_local_provider",
    "Privacy Filter": "providers.privacy_filter_local_provider",
}

with st.sidebar:
    st.header("Configuration")
    selected_provider_name = st.selectbox("Détecteur PII (texte)", options=list(PROVIDERS.keys()))
    mode = st.radio(
        "Méthode de masquage",
        options=["beep", "silence"],
        format_func=lambda m: "Bip" if m == "beep" else "Silence",
        horizontal=True,
    )
    padding_ms = st.slider(
        "Marge de sécurité",
        min_value=0, max_value=500, value=150, step=25,
        help="Élargit chaque plage masquée (en ms) pour absorber l'imprécision des timestamps Whisper.",
    )
    language = st.text_input("Langue", value="fr")

    st.divider()
    if selected_provider_name in ("GLiNER", "GLiNER (fine-tuné FR)", "GLiNER2 (PII multilingue)"):
        custom_instruction = st.text_area(
            "Labels à détecter",
            placeholder="prénom\nnom de famille\nnuméro de téléphone",
            height=120,
            help="Vide = utilise la liste PII complète par défaut.",
        )
    elif selected_provider_name == "Presidio":
        custom_instruction = st.text_area(
            "Types d'entités Presidio",
            placeholder="PERSON, PHONE_NUMBER, EMAIL_ADDRESS",
            height=80,
            help="Vide = détecter toutes les entités.",
        )
    else:
        custom_instruction = ""

# --- Initialisation session state ---
for k in ("report", "audio_out_bytes", "filename", "last_uploaded_name"):
    if k not in st.session_state:
        st.session_state[k] = None

# --- Zone de dépôt de l'audio ---
uploaded = st.file_uploader("Déposez votre fichier audio ici", type=["mp3", "wav", "m4a", "ogg", "flac", "webm"])

# Réinitialiser si le fichier est supprimé ou remplacé
current_name = uploaded.name if uploaded else None
if current_name != st.session_state.last_uploaded_name:
    for k in ("report", "audio_out_bytes", "filename"):
        st.session_state[k] = None
    st.session_state.last_uploaded_name = current_name

if uploaded:
    audio_bytes = uploaded.read()
    st.success(f"Fichier chargé : **{uploaded.name}** ({len(audio_bytes) // 1024} Ko)")
    st.audio(audio_bytes)

    if st.button("Anonymiser l'audio", type="primary", use_container_width=True):
        provider = importlib.import_module(PROVIDERS[selected_provider_name])
        suffix = Path(uploaded.name).suffix or ".wav"

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / f"in{suffix}"
            out_path = Path(tmpdir) / f"out{suffix}"
            in_path.write_bytes(audio_bytes)

            with st.spinner("Transcription et détection en cours…"):
                t_start = time.time()
                try:
                    report = redact_audio_pii(
                        str(in_path), str(out_path), provider,
                        mode=mode, padding_s=padding_ms / 1000,
                        custom_instruction=custom_instruction, language=language,
                    )
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    st.stop()
                report["elapsed"] = time.time() - t_start

            st.session_state.report = report
            st.session_state.audio_out_bytes = out_path.read_bytes()
            st.session_state.filename = f"anonymise_{uploaded.name}"

# --- Résultats persistés ---
if st.session_state.report is not None:
    st.divider()
    report = st.session_state.report

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Passages masqués", len(report["audio_spans"]))
    with col2:
        st.metric("Temps de traitement", f"{report['elapsed']:.1f} s")

    if report["entities"]:
        st.subheader("Entités détectées")
        counts = Counter(e["label"] for e in report["entities"])
        cols = st.columns(min(len(counts), 4))
        for idx, (label, count) in enumerate(counts.most_common()):
            cols[idx % len(cols)].metric(label, count)

        with st.expander("Détail des plages masquées"):
            df = pd.DataFrame(report["audio_spans"]).rename(
                columns={"start": "Début (s)", "end": "Fin (s)", "label": "Type", "text": "Texte détecté"}
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune donnée sensible détectée dans la transcription.")

    with st.expander("Transcription"):
        st.text(report["transcript"])

    st.subheader("Audio anonymisé")
    st.audio(st.session_state.audio_out_bytes)
    mime = "audio/mpeg" if Path(st.session_state.filename).suffix.lower() == ".mp3" else "audio/wav"
    st.download_button(
        label="Télécharger l'audio anonymisé",
        data=st.session_state.audio_out_bytes,
        file_name=st.session_state.filename,
        mime=mime,
        use_container_width=True,
        type="primary",
    )
