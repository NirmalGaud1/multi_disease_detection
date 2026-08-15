import os
import sys
import json
import pickle
import shutil
import subprocess
import importlib.util
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="MDRS-Net++ Multi-Disease Risk Stratification",
    layout="wide"
)


DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1ePN-Dvt5p54aEAVdkU5qsEEbcNtVnnhI?usp=sharing"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mdrs_net_final_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "mdrs_net_metadata.json")
LIB_PATH = os.path.join(MODEL_DIR, "mdrs_net_lib.py")


def install_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "gdown"]
        )
        import gdown
        return gdown


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    required_files = [
        MODEL_PATH,
        META_PATH
    ]

    if all(os.path.exists(f) for f in required_files):
        return

    gdown = install_gdown()

    try:
        gdown.download_folder(
            DRIVE_FOLDER_URL,
            output=MODEL_DIR,
            quiet=False,
            use_cookies=False
        )
    except Exception as e:
        st.error(
            "Unable to download model files from Google Drive.\n\n"
            "Make sure the Google Drive folder is shared as "
            "'Anyone with the link - Viewer'."
        )
        st.exception(e)
        st.stop()


def locate_downloaded_files():
    model_found = None
    meta_found = None
    lib_found = None

    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            full_path = os.path.join(root, file)

            if file == "mdrs_net_final_model.pkl":
                model_found = full_path

            elif file == "mdrs_net_metadata.json":
                meta_found = full_path

            elif file == "mdrs_net_lib.py":
                lib_found = full_path

    if model_found and model_found != MODEL_PATH:
        shutil.copy2(model_found, MODEL_PATH)

    if meta_found and meta_found != META_PATH:
        shutil.copy2(meta_found, META_PATH)

    if lib_found and lib_found != LIB_PATH:
        shutil.copy2(lib_found, LIB_PATH)

    return model_found, meta_found, lib_found


def load_library():
    if os.path.exists(LIB_PATH):

        module_dir = os.path.abspath(MODEL_DIR)

        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        try:
            import mdrs_net_lib
            return mdrs_net_lib
        except Exception:
            pass

        try:
            spec = importlib.util.spec_from_file_location(
                "mdrs_net_lib",
                LIB_PATH
            )

            module = importlib.util.module_from_spec(spec)
            sys.modules["mdrs_net_lib"] = module
            spec.loader.exec_module(module)

            return module

        except Exception as e:
            st.error("Unable to load mdrs_net_lib.py")
            st.exception(e)
            st.stop()

    try:
        import mdrs_net_lib
        return mdrs_net_lib
    except Exception as e:
        st.error(
            "mdrs_net_lib.py is required by the saved model but "
            "was not found."
        )

        st.warning(
            "Please upload mdrs_net_lib.py into the same Google Drive "
            "folder as mdrs_net_final_model.pkl and "
            "mdrs_net_metadata.json."
        )

        st.exception(e)
        st.stop()


download_models()

model_file, meta_file, lib_file = locate_downloaded_files()

if not os.path.exists(MODEL_PATH):
    st.error(
        "mdrs_net_final_model.pkl was not found in the Google Drive folder."
    )
    st.stop()

if not os.path.exists(META_PATH):
    st.error(
        "mdrs_net_metadata.json was not found in the Google Drive folder."
    )
    st.stop()


mdrs_net_lib = load_library()

sigmoid = mdrs_net_lib.sigmoid
stratify_risk = mdrs_net_lib.stratify_risk
RISK_TIERS = mdrs_net_lib.RISK_TIERS


@st.cache_resource
def load_bundle():

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    return bundle, meta


try:
    bundle, meta = load_bundle()

except Exception as e:
    st.error("Unable to load the MDRS-Net++ model.")
    st.exception(e)

    st.info(
        "If the error mentions 'mdrs_net_lib', make sure "
        "mdrs_net_lib.py is uploaded to the Google Drive folder."
    )

    st.stop()


model = bundle["model"]
scalers = bundle["scalers"]
feats = bundle["feats"]
encoders = bundle["encoders"]
imputers = bundle["imputer"]
labels = bundle["labels"]


st.title("MDRS-Net++ Multi-Disease Risk Stratification")

st.markdown(
    """
    **Multi-Disease AI Risk Prediction and Stratification System**
    
    Select a disease, enter the patient information, and obtain the
    predicted risk probability, risk tier, and the most influential
    features contributing to the prediction.
    """
)


disease_keys = list(labels.keys())

disease_display = [
    labels[k]
    for k in disease_keys
]


selected_display = st.sidebar.selectbox(
    "Select Disease",
    disease_display
)


selected_key = disease_keys[
    disease_display.index(selected_display)
]


st.sidebar.markdown("---")

st.sidebar.write(
    f"**Number of features:** {len(feats[selected_key])}"
)


st.sidebar.markdown("---")

st.sidebar.info(
    "Model: MDRS-Net++"
)


st.subheader(
    f"Patient Input — {labels[selected_key]}"
)


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

        val = target_col.selectbox(
            feat,
            options,
            key=f"{selected_key}_{feat}"
        )

        input_values[feat] = le.transform([val])[0]

    else:

        try:
            idx = feature_list.index(feat)

            default_val = float(
                imputer.statistics_[idx]
            )

        except Exception:
            default_val = 0.0

        val = target_col.number_input(
            feat,
            value=default_val,
            key=f"{selected_key}_{feat}"
        )

        input_values[feat] = val


st.markdown("---")


if st.button(
    "Predict Risk",
    type="primary",
    use_container_width=True
):

    try:

        row = pd.DataFrame(
            [input_values]
        )[feature_list]


        X_imputed = imputer.transform(row)


        X_scaled = scalers[
            selected_key
        ].transform(X_imputed)


        logits = model.forward(
            X_scaled,
            selected_key,
            training=False
        )


        prob = float(
            sigmoid(logits)[0]
        )


        tier = stratify_risk(
            np.array([prob])
        )[0]


        attributions = model.explain(
            X_scaled,
            selected_key
        )


        order = np.argsort(
            -np.abs(attributions[0])
        )[:5]


        top_feats = [
            (
                feature_list[j],
                float(attributions[0, j])
            )
            for j in order
        ]


        st.markdown("---")

        st.subheader("Prediction Result")


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Risk Probability",
                f"{prob * 100:.2f}%"
            )


        with c2:

            st.metric(
                "Risk Tier",
                tier
            )


        with c3:

            if prob < 0.25:
                status = "Low Risk"

            elif prob < 0.50:
                status = "Moderate Risk"

            elif prob < 0.75:
                status = "High Risk"

            else:
                status = "Critical Risk"

            st.metric(
                "Risk Status",
                status
            )


        st.markdown("---")

        st.subheader(
            "Top Contributing Features"
        )


        contrib_df = pd.DataFrame(
            top_feats,
            columns=[
                "Feature",
                "Attribution"
            ]
        )


        chart_df = contrib_df.set_index(
            "Feature"
        )


        st.bar_chart(
            chart_df
        )


        st.dataframe(
            contrib_df,
            use_container_width=True,
            hide_index=True
        )


        st.markdown("---")

        st.subheader(
            "Interpretation"
        )


        if tier == "Low":

            st.success(
                "The model indicates a relatively low predicted risk "
                "for the selected disease."
            )

        elif tier == "Moderate":

            st.warning(
                "The model indicates a moderate predicted risk. "
                "Further assessment may be appropriate."
            )

        elif tier == "High":

            st.warning(
                "The model indicates a high predicted risk. "
                "Additional clinical evaluation is recommended."
            )

        else:

            st.error(
                "The model indicates a critical predicted risk. "
                "Professional clinical assessment should be considered."
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


st.markdown("---")

st.subheader(
    "Model Performance Summary"
)


final_metrics = bundle.get(
    "final_metrics",
    {}
)


if selected_key in final_metrics:

    m = final_metrics[selected_key]


    mc1, mc2, mc3, mc4 = st.columns(4)


    mc1.metric(
        "AUC",
        f"{m.get('AUC', float('nan')):.3f}"
    )


    mc2.metric(
        "F1",
        f"{m.get('F1', float('nan')):.3f}"
    )


    mc3.metric(
        "Precision",
        f"{m.get('Precision', float('nan')):.3f}"
    )


    mc4.metric(
        "Recall",
        f"{m.get('Recall', float('nan')):.3f}"
    )


else:

    st.info(
        "Performance metrics are not available for the selected disease."
    )


st.markdown("---")

st.caption(
    "MDRS-Net++ Multi-Disease Risk Stratification System"
)
