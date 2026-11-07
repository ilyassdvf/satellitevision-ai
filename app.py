
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import pandas as pd
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="SatelliteVision AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #e5eaf3;
}
[data-testid="stSidebar"] {
    background-color: #111c2e;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}
.hero {
    background: linear-gradient(120deg, #142b43, #123b39);
    padding: 32px;
    border-radius: 18px;
    border: 1px solid #28516a;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #c4d5e4;
    font-size: 16px;
}
.section-title {
    color: #75d6b2;
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
}
.info-card {
    background-color: #142238;
    border: 1px solid #263b55;
    border-radius: 14px;
    padding: 20px;
    min-height: 120px;
}
.info-card h3 {
    color: #75d6b2;
    margin-bottom: 8px;
}
.info-card p {
    color: #c4d5e4;
}
div[data-testid="stMetric"] {
    background-color: #142238;
    border: 1px solid #263b55;
    padding: 15px;
    border-radius: 12px;
}
.stButton button {
    background-color: #168b70;
    color: white;
    border-radius: 9px;
    border: none;
}
.stButton button:hover {
    background-color: #20a887;
    color: white;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# MODEL CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "final_resnet18_35perclass_seed0.pth"
)

CLASS_NAMES = ["Forest", "Residential", "SeaLake"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================================================
# PREPROCESSING PARAMETERS
# Keep identical to Experiment 4
# ==================================================

WUHAN_MEAN = np.array(
    [666.6948, 828.15875, 555.87036],
    dtype=np.float32
).reshape(1, 3, 1, 1)

WUHAN_STD = np.array(
    [408.7104, 286.7424, 275.9673],
    dtype=np.float32
).reshape(1, 3, 1, 1)

EUROSAT_MEAN = np.array(
    [0.34376082, 0.3801997, 0.40843064],
    dtype=np.float32
).reshape(1, 3, 1, 1)

EUROSAT_STD = np.array(
    [0.20128217, 0.1371562, 0.11533085],
    dtype=np.float32
).reshape(1, 3, 1, 1)

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(1, 3, 1, 1)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(1, 3, 1, 1)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model

# ==================================================
# PREPROCESSING
# ==================================================

def preprocess(raw_data):

    x = raw_data.astype(np.float32)

    x = (x - WUHAN_MEAN) / WUHAN_STD

    x = np.clip(
        x * EUROSAT_STD + EUROSAT_MEAN,
        0,
        1
    )

    x = (x - IMAGENET_MEAN) / IMAGENET_STD

    return torch.tensor(x, dtype=torch.float32)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 🛰️ SatelliteVision AI")
    st.caption("Satellite Image Classification")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Image Classification",
            "About the Model",
            "Research Results"
        ]
    )

    st.markdown("---")

    st.caption("Research Demonstration")
    st.caption("ResNet18 Deep Learning Model")
    st.caption(f"Device: {DEVICE.type.upper()}")

# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>🛰️ SatelliteVision AI</h1>
        <p>
        Intelligent satellite image classification
        using deep learning and ResNet18.
        </p>
        <p>
        Research demonstration: efficient fine-tuning
        for geographically unseen satellite imagery.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>🌳 Forest</h3>
            <p>Tree cover and forest land.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>🏘️ Residential</h3>
            <p>Built-up and urban land.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
            <h3>🌊 SeaLake</h3>
            <p>Permanent water bodies.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Research Model")

    m1, m2, m3 = st.columns(3)

    m1.metric("Architecture", "ResNet18")
    m2.metric("Training Patches", "105")
    m3.metric("Target Classes", "3")

    st.info(
        "This system classifies individual 64 × 64 satellite "
        "patches. It does not classify an entire satellite scene."
    )

    if st.button("Open Image Classification", use_container_width=True):
        st.session_state["navigation_target"] = "Image Classification"
        st.rerun()

# ==================================================
# IMAGE CLASSIFICATION PAGE
# ==================================================

elif page == "Image Classification":

    st.title("Image Classification")
    st.write(
        "Upload raw satellite image patches to classify "
        "their land-cover categories."
    )

    st.info(
        "Supported input: NumPy (.npy) arrays containing "
        "three satellite bands in B04, B03, B02 order."
    )

    uploaded_file = st.file_uploader(
        "Upload satellite data",
        type=["npy"]
    )

    st.caption(
        "Expected shape: (3, 64, 64) for one patch, "
        "or (N, 3, 64, 64) for multiple patches."
    )

    if uploaded_file is not None:

        try:

            raw_data = np.load(
                uploaded_file,
                allow_pickle=False
            )

            if raw_data.ndim == 3:
                raw_data = raw_data[np.newaxis, ...]

            if raw_data.ndim != 4 or raw_data.shape[1:] != (3, 64, 64):
                st.error(
                    "Invalid data shape. Expected "
                    "(3, 64, 64) or (N, 3, 64, 64)."
                )

            elif not np.issubdtype(raw_data.dtype, np.number):
                st.error("Input must contain numeric values.")

            elif not np.isfinite(raw_data).all():
                st.error("Input contains invalid values.")

            elif np.any(raw_data < 0) or np.any(raw_data > 65535):
                st.error("Values are outside the expected raw measurement range.")

            elif raw_data.size > 100_000_000:
                st.error("Uploaded dataset is too large.")

            else:

                st.success(
                    f"Dataset loaded successfully: {len(raw_data)} patch(es)"
                )

                if st.button(
                    "🚀 Classify Satellite Images",
                    use_container_width=True
                ):

                    try:

                        model = load_model()

                        all_results = []

                        progress = st.progress(0)

                        batch_size = 32

                        for start in range(
                            0, len(raw_data), batch_size
                        ):

                            batch = raw_data[
                                start:start + batch_size
                            ]

                            inputs = preprocess(batch).to(DEVICE)

                            with torch.no_grad():
                                outputs = model(inputs)
                                probs = torch.softmax(outputs, dim=1)

                            all_results.extend(
                                probs.cpu().numpy()
                            )

                            progress.progress(
                                min(
                                    (start + batch_size) / len(raw_data),
                                    1.0
                                )
                            )

                        results = np.array(all_results)

                        predicted_indices = np.argmax(
                            results,
                            axis=1
                        )

                        predicted_names = [
                            CLASS_NAMES[i]
                            for i in predicted_indices
                        ]

                        st.success("Classification completed!")

                        st.markdown("---")

                        # First patch details
                        first_result = results[0]
                        first_index = int(np.argmax(first_result))

                        st.subheader("Prediction Details")

                        left, right = st.columns(2)

                        with left:
                            st.metric(
                                "Predicted Class",
                                CLASS_NAMES[first_index]
                            )

                        with right:
                            st.metric(
                                "Confidence",
                                f"{first_result[first_index] * 100:.2f}%"
                            )

                        st.subheader("Class Probabilities")

                        probability_df = pd.DataFrame({
                            "Class": CLASS_NAMES,
                            "Probability": first_result
                        })

                        st.bar_chart(
                            probability_df.set_index("Class")
                        )

                        # Summary of all patches
                        st.markdown("---")
                        st.subheader("Classification Summary")

                        counts = pd.Series(
                            predicted_names
                        ).value_counts()

                        summary_cols = st.columns(3)

                        for i, class_name in enumerate(CLASS_NAMES):

                            with summary_cols[i]:

                                st.metric(
                                    class_name,
                                    int(counts.get(class_name, 0))
                                )

                        # Results table
                        results_df = pd.DataFrame({
                            "Patch": np.arange(1, len(results) + 1),
                            "Prediction": predicted_names,
                            "Confidence (%)": (
                                np.max(results, axis=1) * 100
                            ).round(2)
                        })

                        st.markdown("---")
                        st.subheader("All Predictions")

                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        csv = results_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "⬇️ Download Results CSV",
                            data=csv,
                            file_name="satellite_classification_results.csv",
                            mime="text/csv"
                        )

                    except Exception as error:

                        st.error(
                            f"Classification error: {error}"
                        )

        except Exception as error:

            st.error(f"File processing error: {error}")

# ==================================================
# ABOUT THE MODEL
# ==================================================

elif page == "About the Model":

    st.title("About the AI Model")

    st.write(
        "SatelliteVision AI uses a convolutional neural "
        "network based on the ResNet18 architecture."
    )

    st.subheader("Model Information")

    model_info = {
        "Architecture": "ResNet18",
        "Starting Weights": "EuroSAT pretrained",
        "Fine-tuning Strategy": "Full fine-tuning",
        "Training Samples": "105",
        "Patches per Class": "35",
        "Number of Classes": "3",
        "Input Dimensions": "64 × 64",
        "Input Bands": "B04, B03, B02"
    }

    for key, value in model_info.items():
        st.write(f"**{key}:** {value}")

    st.markdown("---")

    st.subheader("How Classification Works")

    st.write("""
    1. Upload a satellite patch in NumPy format.
    2. The application applies the research preprocessing.
    3. The ResNet18 model processes the patch.
    4. Softmax converts model outputs into class probabilities.
    5. The application displays the predicted land-cover class.
    """)

    st.markdown("---")

    st.subheader("Research Objective")

    st.write(
        "The research investigates how fine-tuning efficiency "
        "can be improved when adapting a pretrained ResNet18 "
        "model to geographically unseen satellite imagery."
    )

# ==================================================
# RESEARCH RESULTS
# ==================================================

elif page == "Research Results":

    st.title("Research Results")

    st.write(
        "External evaluation results from geographically "
        "unseen satellite imagery."
    )

    results_data = pd.DataFrame({
        "Evaluation Dataset": [
            "Wuhan - September 21",
            "Wuhan - September 26",
            "Hangzhou - External"
        ],
        "Accuracy (%)": [
            96.59,
            91.07,
            96.57
        ],
        "Macro F1 (%)": [
            93.47,
            86.03,
            95.81
        ]
    })

    st.dataframe(
        results_data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("External Evaluation Accuracy")

    st.bar_chart(
        results_data.set_index(
            "Evaluation Dataset"
        )[["Accuracy (%)"]]
    )

    st.subheader("External Evaluation Macro F1")

    st.bar_chart(
        results_data.set_index(
            "Evaluation Dataset"
        )[["Macro F1 (%)"]]
    )

    st.caption(
        "These are recorded research evaluation results, "
        "not measurements generated by the current upload session."
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#91a4ba;">
        SatelliteVision AI | ResNet18 Deep Learning Research
        <br>
        Computer Science Research Demonstration
    </div>
    """,
    unsafe_allow_html=True
)
