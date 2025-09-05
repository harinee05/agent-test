import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
from datetime import datetime
import os

SERVICE_LIST_URL = "https://selfservicelp.las-cruces.org/prod/tyler311/public/v2/services.xml?env=live"

DATASET_PATH = "311_training_dataset.csv"
N_BEST = 3

@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(ttl=3600)
def fetch_services_live():
    j = requests.get(SERVICE_LIST_URL).json()
    svclist = j.get("services", {}).get("service", [])
    return [
        {
            'code': svc.get('service_code', ''),
            'name': svc.get('service_name', ''),
            'description': svc.get('description', '')
        }
        for svc in svclist
    ]

@st.cache_data(ttl=3600)
def build_service_embeddings(services, _model):
    texts = [f"{svc['name']}: {svc['description']}" for svc in services]
    embs = _model.encode(texts)
    return embs

def smart_match_service(user_desc, services, embs, model, n_best=3, score_threshold=0.3):
    user_emb = model.encode([user_desc])[0]
    sims = np.inner(embs, user_emb)
    top_idx = np.argsort(-sims)
    results = []
    for rank in range(n_best):
        i = top_idx[rank]
        score = sims[i]
        if score < score_threshold:
            continue
        results.append({
            "service": services[i],
            "score": float(score)
        })
    return results

def log_to_dataset(row):
    write_header = not os.path.exists(DATASET_PATH)
    df = pd.DataFrame([row])
    df.to_csv(DATASET_PATH, mode='a', header=write_header, index=False, encoding="utf-8")

# ----------- Streamlit App -----------
st.set_page_config(page_title="311 Service Matcher Logger", layout="centered")
st.title("📋 311 Service Matcher & Trainer (with Logging)")

model = load_model()
services = fetch_services_live()
service_embs = build_service_embeddings(services, model)

user_input = st.text_input("Describe your city issue (in your own words)")

if user_input:
    matches = smart_match_service(user_input, services, service_embs, model, n_best=N_BEST)
    top_services = [m["service"] for m in matches]
    codes = [svc["code"] for svc in top_services]
    names = [svc["name"] for svc in top_services]
    descs = [svc["description"] for svc in top_services]
    scores = [m["score"] for m in matches]
    st.markdown("#### 🤖 Bot's Top Service Matches:")
    for svc, score in zip(top_services, scores):
        st.write(
            f"- **{svc['name']}** (code: {svc['code']}): {svc['description']}  "
            f"_Score: {score:.3f}_"
        )

    if matches:
        chosen = st.selectbox(
            "Select which service is the best match:", [svc["name"] for svc in top_services]
        )
        chosen_idx = names.index(chosen)
        final_svc = top_services[chosen_idx]
        if st.button("Log Interaction to Dataset"):
            log_row = {
                "timestamp": datetime.now().isoformat(),
                "user_description": user_input,
                "bot_top_service_codes": "|".join(codes),
                "bot_top_service_names": "|".join(names),
                "bot_top_scores": "|".join(str(round(s,3)) for s in scores),
                "bot_top_descriptions": "|".join(descs),
                "user_selected_code": final_svc["code"],
                "user_selected_name": final_svc["name"],
                "user_selected_description": final_svc["description"]
            }
            log_to_dataset(log_row)
            st.success("Logged! This turn was saved to 311_training_dataset.csv")
            st.write(log_row)
    else:
        st.warning("No confident match found. Please rephrase or check the full service list.")

if os.path.exists(DATASET_PATH):
    with st.expander("See the current training dataset"):
        st.dataframe(pd.read_csv(DATASET_PATH))
