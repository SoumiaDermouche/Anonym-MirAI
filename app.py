"""Application Streamlit — anonymisation de PDF avec choix du modèle."""
import html as _html
import os
import time
from collections import Counter
from pathlib import Path

# Empêche transformers de charger TensorFlow : la coexistence avec
# torch/triton provoque un segfault au chargement des modèles locaux
# (GLiNER, OpenMed, Privacy Filter) et fait crasher le serveur Streamlit
# (vu côté navigateur comme "Connection error / status 500").
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

import pandas as pd
import streamlit as st
from pdf_processor import extract_pages, build_pdf


# ---------------------------------------------------------------------------
# Visualisation des entités colorées (inspiré de spaCy displacy +
# Whisper word-level confidence — cf. README section Visualisation)
# ---------------------------------------------------------------------------

def _conf_style(score: float) -> tuple[str, str]:
    """Retourne (background, couleur_texte) selon le score de confiance."""
    if score >= 0.8:
        return "#d4edda", "#155724"   # vert  — haute confiance
    if score >= 0.6:
        return "#fff3cd", "#856404"   # jaune — confiance moyenne
    return "#f8d7da", "#721c24"       # rouge — faible confiance


def render_annotated_text(original: str, anonymized: str, entities: list) -> str:
    """
    Génère du HTML avec les entités surlignées dans le texte original,
    à partir de leurs positions start/end et de leur score de confiance.
    Les modèles proxy n'exposent ni l'un ni l'autre : leur texte est
    affiché tel quel, sans annotation (on ne peut pas afficher une
    entité détectée sans score de confiance associé).
    """
    has_positions = bool(entities) and "start" in entities[0] and "score" in entities[0]

    if not has_positions:
        return _html.escape(original).replace("\n", "<br>")

    spans = [
        (e["start"], e["end"], e.get("label", ""), e["score"])
        for e in sorted(entities, key=lambda e: e["start"])
    ]

    parts = []
    last_end = 0
    for start, end, label, score in spans:
        if start < last_end:
            continue
        # Texte brut avant l'entité
        parts.append(_html.escape(original[last_end:start]).replace("\n", "<br>"))
        # Span coloré
        bg, fg = _conf_style(score)
        label_tag = (
            f'<sup style="font-size:0.72em;font-weight:bold;color:{fg};margin-left:3px">'
            f'{_html.escape(label)}&nbsp;{score:.0%}</sup>'
        ) if label else ""
        parts.append(
            f'<mark style="background:{bg};padding:1px 5px;border-radius:4px;margin:0 1px">'
            f'{_html.escape(original[start:end])}{label_tag}</mark>'
        )
        last_end = end

    parts.append(_html.escape(original[last_end:]).replace("\n", "<br>"))
    return "".join(parts)


_LEGEND_HTML = """
<div style="font-size:0.82em;margin-bottom:10px;display:flex;gap:10px;flex-wrap:wrap">
  <span><mark style="background:#d4edda;padding:1px 6px;border-radius:4px">&#9679;</mark>&nbsp;≥&nbsp;80% — fiable</span>
  <span><mark style="background:#fff3cd;padding:1px 6px;border-radius:4px">&#9679;</mark>&nbsp;60–80% — moyen</span>
  <span><mark style="background:#f8d7da;padding:1px 6px;border-radius:4px">&#9679;</mark>&nbsp;&lt;&nbsp;60% — incertain</span>
</div>
"""


def build_annotated_html(pages_raw: list[str], pages_anonymized: list[str], pages_entities: list[list]) -> bytes:
    """Document HTML autonome — toutes les pages, même rendu que 'Visualisation des entités'."""
    parts = [
        "<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'>"
        "<title>Entités détectées</title></head>"
        "<body style='font-family:sans-serif;max-width:900px;margin:24px auto;padding:0 16px'>"
        "<h1 style='font-size:1.3em'>Entités détectées — texte original</h1>",
        _LEGEND_HTML,
    ]
    for i, (raw, anon, ents) in enumerate(zip(pages_raw, pages_anonymized, pages_entities)):
        annotated = render_annotated_text(raw, anon, ents)
        parts.append(
            f'<h2 style="font-size:0.95em;color:#555;border-top:1px solid #e0e0e0;'
            f'padding-top:14px;margin-top:20px">Page {i + 1}</h2>'
        )
        parts.append(
            f'<div style="line-height:1.9;font-family:monospace;font-size:0.9em;'
            f'white-space:pre-wrap;border:1px solid #e0e0e0;border-radius:6px;padding:12px">'
            f'{annotated}</div>'
        )
    parts.append("</body></html>")
    return "".join(parts).encode("utf-8")

st.set_page_config(
    page_title="Anonymisation de PDF",
    page_icon="🔒",
    layout="centered",
)

st.title("Anonymisation de documents PDF")
st.caption("Chargez un PDF contenant des données sensibles — toutes les PII seront masquées.")

# --- Définition des modèles disponibles ---
MODELS = {
    "Privacy Filter": {
        "provider": "privacy_filter_local",
        "env_endpoint": None,
        "env_key": None,
    },
    "OpenMed": {
        "provider": "openmed_local",
        "env_endpoint": None,
        "env_key": None,
    },
    "GLiNER": {
        "provider": "gliner_local",
        "env_endpoint": None,
        "env_key": None,
    },
    "GLiNER (fine-tuné FR)": {
        "provider": "gliner_fr_local",
        "env_endpoint": None,
        "env_key": None,
    },
    "Presidio": {
        "provider": "presidio",
        "env_endpoint": None,
        "env_key": None,
    },
}

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    selected_model_name = st.selectbox(
        "Modèle d'anonymisation",
        options=list(MODELS.keys()),
        key="selected_model",
    )

    st.divider()
    if selected_model_name in ("GLiNER", "GLiNER (fine-tuné FR)"):
        st.subheader("Entités à détecter")
        default_help = (
            "GLiNER accepte n'importe quel label en français ou anglais.\n\n"
            "Exemples : prénom, nom de famille, adresse email, numéro de téléphone, "
            "date de naissance, numéro de sécurité sociale, organisation, lieu\n\n"
            "Vide = utilise la liste PII complète par défaut."
        )
        if selected_model_name == "GLiNER (fine-tuné FR)":
            default_help = (
                "Modèle fine-tuné sur ai4privacy (FR) : title, given name, surname, date, city, "
                "street, building number, zip code, age, sex, gender, email address, phone number, "
                "credit card number, national id card number, passport number, driver license number, "
                "tax identification number, social security number.\n\n"
                "Vide = utilise cette liste par défaut (précision optimale sur ces labels)."
            )
        custom_instruction = st.text_area(
            "Labels (un par ligne ou séparés par des virgules)",
            placeholder="prénom\nnom de famille\nadresse email\nnuméro de téléphone",
            height=160,
            help=default_help,
        )
    elif selected_model_name == "Presidio":
        st.subheader("Filtrer les entités")
        custom_instruction = st.text_area(
            "Types d'entités Presidio",
            placeholder="PERSON, EMAIL_ADDRESS, PHONE_NUMBER",
            height=100,
            help=(
                "Types courants : PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, "
                "DATE_TIME, IBAN_CODE, CREDIT_CARD\n\n"
                "Vide = détecter toutes les entités."
            ),
        )
    else:
        custom_instruction = ""

# Effacer les résultats si le modèle a changé
if st.session_state.get("last_model") != selected_model_name:
    for k in ("pages_anonymized", "pages_raw", "entities_all", "pages_entities", "pdf_out", "filename", "response_time"):
        st.session_state[k] = None

model_cfg = MODELS[selected_model_name]


def run_anonymize(text: str, provider: str, endpoint: str, api_key: str, custom_instruction: str = "") -> dict:
    if provider == "presidio":
        from providers.presidio_provider import anonymize
        return anonymize(text, custom_instruction=custom_instruction)
    elif provider == "gliner_local":
        from providers.gliner_local_provider import anonymize
        return anonymize(text, custom_instruction=custom_instruction)
    elif provider == "gliner_fr_local":
        from providers.gliner_fr_provider import anonymize
        return anonymize(text, custom_instruction=custom_instruction)
    elif provider == "openmed_local":
        from providers.openmed_local_provider import anonymize
        return anonymize(text, custom_instruction=custom_instruction)
    elif provider == "privacy_filter_local":
        from providers.privacy_filter_local_provider import anonymize
        return anonymize(text, custom_instruction=custom_instruction)


# --- Initialisation session state ---
for k in ("pages_anonymized", "pages_raw", "entities_all", "pages_entities", "pdf_out", "filename", "last_model", "response_time", "last_uploaded_name"):
    if k not in st.session_state:
        st.session_state[k] = None

# --- Zone de dépôt du PDF ---
uploaded = st.file_uploader("Déposez votre PDF ici", type=["pdf"])

# Réinitialiser si le fichier est supprimé ou remplacé
current_name = uploaded.name if uploaded else None
if current_name != st.session_state.last_uploaded_name:
    for k in ("pages_anonymized", "pages_raw", "entities_all", "pages_entities", "pdf_out", "filename", "response_time"):
        st.session_state[k] = None
    st.session_state.last_uploaded_name = current_name

if uploaded:
    pdf_bytes = uploaded.read()
    st.success(f"Fichier chargé : **{uploaded.name}** ({len(pdf_bytes) // 1024} Ko)")

    if st.button("Anonymiser", type="primary", use_container_width=True):
        pages_raw = extract_pages(pdf_bytes)
        pages_anonymized = []
        entities_all = []
        pages_entities = []   # entités par page pour la visualisation

        progress = st.progress(0, text="Anonymisation en cours…")
        error_occurred = False
        t_start = time.time()
        for i, page_text in enumerate(pages_raw):
            if page_text.strip():
                try:
                    result = run_anonymize(page_text, model_cfg["provider"], None, None, custom_instruction)
                    page_ents = result.get("entities", [])
                    pages_anonymized.append(result["anonymized"])
                    entities_all.extend(page_ents)
                    pages_entities.append(page_ents)
                except Exception as e:
                    st.error(f"Erreur page {i+1} : {e}")
                    error_occurred = True
                    break
            else:
                pages_anonymized.append("")
                pages_entities.append([])
            progress.progress((i + 1) / len(pages_raw), text=f"Page {i + 1}/{len(pages_raw)}")

        if error_occurred:
            progress.empty()
            st.stop()
        progress.empty()

        st.session_state.pages_raw = pages_raw
        st.session_state.pages_anonymized = pages_anonymized
        st.session_state.entities_all = entities_all
        st.session_state.pages_entities = pages_entities
        st.session_state.pdf_out = bytes(build_pdf(pages_anonymized))
        st.session_state.filename = f"anonymise_{uploaded.name}"
        st.session_state.last_model = selected_model_name
        st.session_state.response_time = time.time() - t_start

# --- Résultats persistés ---
if st.session_state.pdf_out is not None:
    st.divider()

    col_model, col_time = st.columns([2, 1])
    with col_model:
        if st.session_state.last_model:
            st.caption(f"Modèle utilisé : **{st.session_state.last_model}**")
    with col_time:
        if st.session_state.response_time is not None:
            st.metric("Temps de réponse", f"{st.session_state.response_time:.2f} s")

    st.subheader("Entités détectées")
    entities_all = st.session_state.entities_all
    if entities_all:
        counts = Counter(e["label"] for e in entities_all)
        cols = st.columns(min(len(counts), 4))
        for idx, (label, count) in enumerate(counts.most_common()):
            cols[idx % len(cols)].metric(label, count)

        with st.expander("Détail des entités trouvées"):
            df = pd.DataFrame(entities_all).rename(columns={"text": "Placeholder", "label": "Type"})
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune entité PII détectée.")

    with st.expander("Visualisation des entités — texte original (page 1)"):
        raw0 = (st.session_state.pages_raw or [""])[0]
        anon0 = (st.session_state.pages_anonymized or [""])[0]
        ents0 = (st.session_state.pages_entities or [[]])[0]
        st.markdown(_LEGEND_HTML, unsafe_allow_html=True)
        annotated = render_annotated_text(raw0[:3000], anon0[:3000], ents0)
        st.markdown(
            f'<div style="line-height:1.9;font-family:monospace;font-size:0.9em;'
            f'white-space:pre-wrap;border:1px solid #e0e0e0;border-radius:6px;padding:12px">'
            f'{annotated}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Texte anonymisé brut (page 1)"):
        st.text(anon0[:2000])

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="Télécharger le PDF anonymisé",
            data=st.session_state.pdf_out,
            file_name=st.session_state.filename,
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )
    with col_dl2:
        annotated_html = build_annotated_html(
            st.session_state.pages_raw,
            st.session_state.pages_anonymized,
            st.session_state.pages_entities,
        )
        html_filename = Path(st.session_state.filename).stem + ".html"
        st.download_button(
            label="Télécharger les entités annotées (HTML)",
            data=annotated_html,
            file_name="annote_" + html_filename,
            mime="text/html",
            use_container_width=True,
        )
