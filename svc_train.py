import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os

API_URL = "http://localhost:9000/match"   # Change to FastAPI host

DATASET_PATH = "311_training_dataset.csv"
N_BEST = 3

def log_to_dataset(row):
    write_header = not os.path.exists(DATASET_PATH)
    df = pd.DataFrame([row])
    df.to_csv(DATASET_PATH, mode='a', header=write_header, index=False, encoding="utf-8")

st.set_page_config(page_title="311 Service Matcher Logger", layout="centered")
st.title("📋 311 Service Matcher & Trainer (with Logging, FastAPI Backend)")

user_input = st.text_input("Describe your city issue (in your own words)")

if user_input:
    resp = requests.post(API_URL, json={"user_desc": user_input, "n_best": N_BEST})
    matches = resp.json()
    codes = [m["code"] for m in matches]
    names = [m["name"] for m in matches]
    descs = [m["description"] for m in matches]
    scores = [m["score"] for m in matches]
    st.markdown("#### 🤖 Bot's Top Service Matches:")
    for m in matches:
        st.write(
            f"- **{m['name']}** (code: {m['code']}): {m['description']}  "
            f"_Score: {m['score']:.3f}_"
        )
    if matches:
        chosen = st.selectbox(
            "Select which service is the best match:", names
        )
        chosen_idx = names.index(chosen)
        final_match = matches[chosen_idx]
        if st.button("Log Interaction to Dataset"):
            log_row = {
                "timestamp": datetime.now().isoformat(),
                "user_description": user_input,
                "bot_top_service_codes": "|".join(codes),
                "bot_top_service_names": "|".join(names),
                "bot_top_scores": "|".join(str(round(s,3)) for s in scores),
                "bot_top_descriptions": "|".join(descs),
                "user_selected_code": final_match["code"],
                "user_selected_name": final_match["name"],
                "user_selected_description": final_match["description"]
            }
            log_to_dataset(log_row)
            st.success("Logged! This turn was saved to 311_training_dataset.csv")
            st.write(log_row)
    else:
        st.warning("No confident match found. Please rephrase or check the full service list.")

if os.path.exists(DATASET_PATH):
    with st.expander("See the current training dataset"):
        st.dataframe(pd.read_csv(DATASET_PATH))
