#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.getcwd())
from mdrs_net_lib import sigmoid, stratify_risk, RISK_TIERS

MODEL_PATH = "models/mdrs_net_final_model.pkl"
META_PATH = "models/mdrs_net_metadata.json"

st.set_page_config(page_title="MDRS-Net++ Multi-Disease Risk Stratification", layout="wide")

@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return bundle, meta

bundle, meta = load_bundle()
model = bundle["model"]
scalers = bundle["scalers"]
feats = bundle["feats"]
encoders = bundle["encoders"]
imputers = bundle["imputer"]
labels = bundle["labels"]

st.title("MDRS-Net++ Multi-Disease Risk Stratification")

disease_keys = list(labels.keys())
disease_display = [labels[k] for k in disease_keys]
selected_display = st.sidebar.selectbox("Select Disease", disease_display)
selected_key = disease_keys[disease_display.index(selected_display)]

st.sidebar.markdown("---")
st.sidebar.write(f"Number of features: {len(feats[selected_key])}")

st.subheader(f"Patient Input — {labels[selected_key]}")

feature_list = feats[selected_key]
encoder_dict = encoders[selected_key]
imputer = imputers[selected_key]

col1, col2 = st.columns(2)
input_values = {}

for i, feat in enumerate(feature_list):
    target_col = col1 if i % 2 == 0 else col2
    if feat in encoder_dict:
        le = encoder_dict[feat]
        options = list(le.classes_)
        val = target_col.selectbox(feat, options, key=f"{selected_key}_{feat}")
        input_values[feat] = le.transform([val])[0]
    else:
        try:
            idx = feature_list.index(feat)
            default_val = float(imputer.statistics_[idx])
        except Exception:
            default_val = 0.0
        val = target_col.number_input(feat, value=default_val, key=f"{selected_key}_{feat}")
        input_values[feat] = val

if st.button("Predict Risk"):
    row = pd.DataFrame([input_values])[feature_list]
    X_imputed = imputer.transform(row)
    X_scaled = scalers[selected_key].transform(X_imputed)
    logits = model.forward(X_scaled, selected_key, training=False)
    prob = float(sigmoid(logits)[0])
    tier = stratify_risk(np.array([prob]))[0]
    attributions = model.explain(X_scaled, selected_key)
    order = np.argsort(-np.abs(attributions[0]))[:5]
    top_feats = [(feature_list[j], float(attributions[0, j])) for j in order]

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Risk Probability", f"{prob*100:.2f}%")
    with c2:
        tier_color = {"Low": "green", "Moderate": "orange", "High": "red", "Critical": "darkred"}
        st.markdown(f"### Risk Tier: :{tier_color.get(tier,'gray')}[{tier}]")

    st.subheader("Top Contributing Features")
    contrib_df = pd.DataFrame(top_feats, columns=["Feature", "Attribution"])
    st.bar_chart(contrib_df.set_index("Feature"))
    st.dataframe(contrib_df)

st.markdown("---")
st.subheader("Model Performance Summary")
final_metrics = bundle.get("final_metrics", {})
if selected_key in final_metrics:
    m = final_metrics[selected_key]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("AUC", f"{m.get('AUC', float('nan')):.3f}")
    mc2.metric("F1", f"{m.get('F1', float('nan')):.3f}")
    mc3.metric("Precision", f"{m.get('Precision', float('nan')):.3f}")
    mc4.metric("Recall", f"{m.get('Recall', float('nan')):.3f}")

