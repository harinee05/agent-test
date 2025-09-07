import streamlit as st
import pandas as pd
from datetime import datetime
import os
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

SERVICE_LIST_URL = os.getenv(
    "SERVICE_LIST_URL",
    "https://selfservicelp.las-cruces.org/prod/tyler311/public/v2/services.xml?env=live"
)
DATASET_PATH = "311_training_dataset.csv"
N_BEST = 3

STYLES = """
<style>
body {
    background: #f4f6f8 !important;
}
h1.title {
    font-family: 'Roboto', sans-serif;
    font-size: 2.3em;
    color: #263238;
    letter-spacing: 1px;
    margin-bottom:0.4em;
}
div.field-label {
    font-weight: 550; color:#37474f; margin-top:1.1em;margin-bottom:-10px;
}
span.service-chip {
    background: #e3f2fd;
    color: #1565c0;
    font-size: 1em;
    border-radius: 12px;
    padding: 3px 9px;
    margin-right:8px;
    border:1px solid #90caf9;
}
div.service-match-box {
    background: #f5f7fa;
    border-left:4px solid #1976d2;
    padding: 12px 18px 8px 18px;
    border-radius: 7px;
    margin-bottom: 13px;
    box-shadow:0 2px 6px #eceff100;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

# 1. Performance-optimized cache
@st.cache_data(show_spinner="Loading service catalog & initializing semantic search...")
def load_services_and_embeddings():
    response = requests.get(SERVICE_LIST_URL, timeout=10)
    data = response.json()
    svclist = data.get("services", {}).get("service", [])
    services = [
        {
            "code": svc.get("service_code", ""),
            "name": svc.get("service_name", ""),
            "description": svc.get("description", "")
        }
        for svc in svclist
    ]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [f"{svc['name']}: {svc['description']}" for svc in services]
    embs = model.encode(texts)
    return services, embs, model

# Injects custom professional-styled CSS
st.markdown(STYLES, unsafe_allow_html=True)
st.set_page_config(page_title="311 Service Request Matcher", layout="centered", page_icon="📑")

# HEADER: Subtle color with a clean, serious headline.
st.markdown("<h1 class='title'>311 Service Request Type Matcher</h1>", unsafe_allow_html=True)
st.markdown(
    "<div style='color:#789; font-size:1.15em;margin-bottom:25px;max-width:550px;line-height:1.5;'>"
    "<b>Purpose:</b> Map user-reported city issues to the most fitting service request type, with clear audit logging for model training and future improvement.</div>",
    unsafe_allow_html=True,
)

services, embs, model = load_services_and_embeddings()

# --- User input
st.markdown("<div class='field-label'>Describe your city issue or request <span style='color:#aaa'>(required)</span></div>", unsafe_allow_html=True)
user_input = st.text_input("", placeholder="E.g., There's a large pothole on Main St. near 12th Ave.")

# Main matching + logging workflow
if user_input and (len(services) > 0) and (embs is not None) and (model is not None):
    user_emb = model.encode([user_input])[0]
    sims = np.inner(embs, user_emb)
    top_idx = np.argsort(-sims)
    matches = []
    for rank in range(min(N_BEST, len(services))):
        i = top_idx[rank]
        matches.append({
            "code": services[i]["code"],
            "name": services[i]["name"],
            "description": services[i]["description"],
            "score": float(sims[i]),
        })
    # Result display: professional, succinct chips and cards
    st.markdown("<div class='field-label'>Automated Top Service Matches</div>", unsafe_allow_html=True)
    for m in matches:
        st.markdown(
            f"<div class='service-match-box'>"
            f"<span class='service-chip'>Code: {m['code']}</span>"
            f"<b>{m['name']}</b><br>"
            f"<span style='color:#455a64;'>{m['description']}</span><br>"
            f"<span style='color:#0079c4; font-size:0.97em;'>Match Score:</span> "
            f"<b style='font-family:monospace; color:#114477'>{m['score']:.3f}</b></div>",
            unsafe_allow_html=True
        )
    st.markdown("<div class='field-label'>Select best match for this description <span style='color:#aaa'>(required)</span></div>", unsafe_allow_html=True)
    names = [m['name'] for m in matches]
    chosen = st.selectbox("", names, key='select_service')
    chosen_idx = names.index(chosen)
    final_match = matches[chosen_idx]
    if st.button("Log This Request for Model Training", use_container_width=True):
        log_row = {
            "timestamp": datetime.now().isoformat(),
            "user_description": user_input,
            "bot_top_service_codes": "|".join([m["code"] for m in matches]),
            "bot_top_service_names": "|".join(names),
            "bot_top_scores": "|".join(str(round(m["score"],3)) for m in matches),
            "bot_top_descriptions": "|".join([m["description"] for m in matches]),
            "user_selected_code": final_match["code"],
            "user_selected_name": final_match["name"],
            "user_selected_description": final_match["description"]
        }
        write_header = not os.path.exists(DATASET_PATH)
        df = pd.DataFrame([log_row])
        df.to_csv(DATASET_PATH, mode='a', header=write_header, index=False, encoding="utf-8")
        st.success("✔️ Request and mapping were logged for model review.")
        with st.expander("See Log Entry Details"):
            st.json(log_row)

# --- Side panels for context/reference ---
with st.expander("🔎 See official 311 service catalog"):
    st.dataframe(pd.DataFrame(services), use_container_width=True)

if os.path.exists(DATASET_PATH):
    with st.expander("📝 Review logged mappings / annotations"):
        st.dataframe(pd.read_csv(DATASET_PATH), use_container_width=True)
    with open(DATASET_PATH, "rb") as f:
        st.download_button("Download Full Log (CSV)", f, file_name=DATASET_PATH, use_container_width=True)
