import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pickle
from streamlit_option_menu import option_menu
import base64
from pathlib import Path
import requests
import re
import random
import warnings
import tensorflow as tf

# ===================== PAGE CONFIG (MUST BE FIRST STREAMLIT CALL) =====================
st.set_page_config(page_title="Early Autism Detection", layout="wide")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only

# ===================== BACKGROUND IMAGE (BASE64) =====================
def get_base64_bg(image_file):
    return base64.b64encode(Path(image_file).read_bytes()).decode()

encoded = get_base64_bg("autism_bg.jpg")

# ===================== GLOBAL UI STYLES =====================
st.markdown(
    f"""
<style>

/* ================= BACKGROUND IMAGE ================= */
.stApp {{
    background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
    background-size: cover;
}}

/* ================= STRONG BLUR + DIM OVERLAY ================= */
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    z-index: 0;
}}

/* Ensure ALL content is above blur */
.stApp > * {{
    position: relative;
    z-index: 1;
}}

/* ================= TOP NAV SPACING FIX ================= */
div[data-testid="stHorizontalBlock"] {{
    margin-top: 12px;
    margin-bottom: 36px;
}}

/* ================= MAIN CONTENT FLOW ================= */
main {{
    padding-top: 0px;
    padding-bottom: 160px;
}}

/* ================= HOME PAGE (FULL WIDTH) ================= */
.block-container {{
    max-width: 100%;
    margin: 0 auto;
    padding: 2.5rem 6vw;
    background: transparent;
    box-shadow: none;
}}

/* =====================================================
   PAGE CARD (ALL NON-HOME PAGES)
===================================================== */
.page-card {{
    background: #ffffff;              /* FULLY SOLID */
    border-radius: 18px;
    padding: 2.4rem 2.6rem 3rem;
    margin: 1.8rem auto 2.8rem auto;
    max-width: 1050px;

    box-shadow:
        0 18px 45px rgba(0,0,0,0.22),
        0 0 0 1px rgba(0,0,0,0.04);

    position: relative;
    z-index: 2;
}}

/* ================= FORM ELEMENT SPACING ================= */
.page-card .stRadio,
.page-card .stSelectbox,
.page-card .stTextInput {{
    margin-bottom: 14px;
}}

/* Make inputs crystal clear */
.page-card input,
.page-card textarea,
.page-card select,
.page-card div[data-baseweb="select"] {{
    background: #ffffff !important;
    border-radius: 8px;
    width: 100% !important;
}}

/* ================= RESULT BANNER ================= */
.page-card .result-banner {{
    margin-top: 18px;
}}

/* ================= TYPOGRAPHY ================= */
h1 {{
    font-size: 38px;
    font-weight: 800;
    margin-top: 0;
}}

h2 {{ font-size: 24px; }}
h3 {{ font-size: 20px; }}

p, li, span {{
    font-size: 16px;
    line-height: 1.6;
}}

/* ================= BUTTONS ================= */
button, .stButton > button {{
    font-size: 15px;
    padding: 9px 20px;
    border-radius: 8px;
}}

/* ================= HIDE STREAMLIT STATUS ================= */
[data-testid="stStatusWidget"] {{
    display: none;
}}

/* ================= NO HORIZONTAL SCROLL ================= */
html, body {{
    overflow-x: hidden;
}}
/* 🔥 REMOVE THE WHITE NAV CONTAINER COMPLETELY */
div[data-testid="stHorizontalBlock"] {{
    background: transparent !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}}

/* 🔥 REMOVE THE INNER WHITE CARD */
div[data-testid="stHorizontalBlock"] > div {{
    background: transparent !important;
    box-shadow: none !important;
}}


</style>
""",
    unsafe_allow_html=True,
)

# ===================== GLOBAL VARIABLES & CACHES =====================


GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]


if "last_chat_answer" not in st.session_state:
    st.session_state.last_chat_answer = ""

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="The use_column_width parameter has been deprecated")

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

@st.cache_data(ttl=60, max_entries=50)
def google_autism_answer(question):
    query = f"autism {question}"

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": 8,
        "safe": "active"
    }

    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()

        if "items" not in data:
            return None

        clean_sentences = []

        for item in data["items"]:
            snippet = item.get("snippet", "")

            # Remove ellipses and extra spaces
            snippet = snippet.replace("...", "")
            snippet = re.sub(r"\s+", " ", snippet).strip()

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', snippet)

            for s in sentences:
                s = s.strip()

                # ✅ Keep ONLY complete sentences
                if (
                    len(s) > 45 and
                    s[-1] in ".!?" and
                    "theme park" not in s.lower() and
                    "engineer" not in s.lower()
                ):
                    clean_sentences.append(s)

        if not clean_sentences:
            return None

        # Shuffle to avoid repetition
        random.shuffle(clean_sentences)

        # Build 3–6 sentence answer
        final_sentences = []
        for s in clean_sentences:
            if s not in st.session_state.last_chat_answer:
                final_sentences.append(s)
            if len(final_sentences) == 6:
                break

        if len(final_sentences) < 3:
            final_sentences = clean_sentences[:3]

        final_answer = " ".join(final_sentences)

        st.session_state.last_chat_answer = final_answer
        return final_answer

    except Exception:
        return "Sorry, I couldn’t retrieve reliable autism-related information at the moment."

# Initialise session state for scores
if "img_score" not in st.session_state:
    st.session_state.img_score = None
if "quest_score" not in st.session_state:
    st.session_state.quest_score = None

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def cleanup_tf():
    tf.keras.backend.clear_session()
    import gc
    gc.collect()
# ===================== MODEL LOADERS (CACHED) =====================

@st.cache_resource
def load_cnn_model():
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(
        "Model/newmodel.h5",
        compile=False
    )
    return model

@st.cache_resource
def load_nb_model():
    """
    Loads the Naive Bayes questionnaire model from nbmodel1.pkl
    in the same folder as this Streamlit app.
    """
    with open("Model_2/nbmodel1.pkl", "rb") as f:
        model = pickle.load(f)
    return model

nb_model = load_nb_model()

# ===================== HELPER FUNCTIONS =====================

def preprocess_face(bgr_image, target_size=(128, 128)):
    """
    Try to detect a face using Haarcascade.
    If no face is found, fall back to center-cropping the image.
    Returns a batch of shape (1, 128, 128, 3).
    """
    h, w = bgr_image.shape[:2]

    # --- Try face detection ---
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) > 0:
        # Pick the largest detected face (more likely to be the main subject)
        x, y, w_box, h_box = max(faces, key=lambda box: box[2] * box[3])
        x1, y1 = x, y
        x2, y2 = x + w_box, y + h_box
        face = bgr_image[y1:y2, x1:x2]
    else:
        # --- Fallback: center-crop a square from the image ---
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        face = bgr_image[y1:y2, x1:x2]

    # Resize, normalize, and make it a batch
    face = cv2.resize(face, target_size)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def predict_image_from_pil(pil_img, cnn_model):
    rgb = pil_img.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    face_batch = preprocess_face(bgr)
    preds = cnn_model.predict(face_batch, verbose=0)

    if preds.shape[1] > 1:
        prob = float(preds[0][1]) * 100
    else:
        prob = float(preds[0][0]) * 100

    return prob

def generate_gradcam(img_array, model, layer_name=None):
    if layer_name is None:
        # automatically detect last conv layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, -1]  # assuming ASD = last class

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(len(pooled_grads)):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-7
    return heatmap

def overlay_heatmap(original_img, heatmap, alpha=0.6):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return overlay

def add_asd_feature_boxes(image):
    h, w = image.shape[:2]

    # Eye region box (RED)
    eye_top_left = (int(w * 0.25), int(h * 0.20))
    eye_bottom_right = (int(w * 0.75), int(h * 0.45))
    cv2.rectangle(image, eye_top_left, eye_bottom_right, (0, 0, 255), 2)

    # Mouth region (YELLOW)
    mouth_top_left = (int(w * 0.35), int(h * 0.60))
    mouth_bottom_right = (int(w * 0.65), int(h * 0.80))
    cv2.rectangle(image, mouth_top_left, mouth_bottom_right, (0, 255, 255), 2)

    # Full face boundary (GREEN)
    cv2.rectangle(image, (5, 5), (w-5, h-5), (0, 255, 0), 2)

    return image

def risk_label(score: float) -> str:
    """
    Converts probability (%) into risk category.
    """
    if score >= 80:
        return "High Risk of Autism"
    elif score >= 50:
        return "Moderate Risk of Autism"
    else:
        return "Low Risk of Autism"


def predict_questionnaire_vector(features: np.ndarray) -> float:
    """
    Takes a single feature vector (shape: (16,)) as defined in questionnaire_section,
    returns autism probability (%) from Naive Bayes model.
    """
    probs = nb_model.predict_proba(features.reshape(1, -1))
    return float(probs[0][1]) * 100

cleanup_tf()

# ===================== PAGE: HOME =====================

intro_text = """
Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition that affects
communication, social interaction and behaviour. Early screening and timely intervention
can significantly improve long-term outcomes.

This dashboard provides:
- Image-based screening using a Convolutional Neural Network (CNN)
- Questionnaire-based screening using a Machine Learning classifier
- An educational chatbot to answer autism-related questions
"""
def home_section():
    with st.container():

        # ======= CENTERED RED BIG TITLE =======
        st.markdown(
            """
            <h1 style="
                text-align: center;
                font-weight: 900;
                font-size: 56px;
                margin-bottom: 12px;
                color: #0b5ed7;
            ">
                Early Autism Spectrum Disorder Prediction System
            </h1>
            """,
            unsafe_allow_html=True
        )

        # ======= SUBTITLE =======
        st.markdown(
            """
            <p style="
                text-align: center;
                font-size: 19px;
                color: #444;
                margin-bottom: 6px;
            ">
                AI-based screening using Facial Image Analysis, Behavioural Questionnaire, and Chatbot Assistance.
            </p>
            <p style="
                text-align: center;
                font-size: 16px;
                font-weight: 600;
                color: #555;
            ">
                Powered by Deep Learning + Machine Learning
            </p>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # ======= REST OF YOUR ORIGINAL CONTENT (UNCHANGED) =======
        st.subheader("Why Early Diagnosis?")
        st.write("Early identification and timely therapy can improve:")
        st.markdown(
            """
            - Communication Skills  
            - Social Interaction  
            - Behavioural Development  
            """
        )

        st.divider()

        st.subheader("Application Features")

        col1, col2, col3 = st.columns([1.2, 1.2, 1.2], gap="large")

        with col1:
            st.image("https://cdn-icons-png.flaticon.com/512/2920/2920322.png", width=90)
            st.markdown("### Image Screening")
            st.write("CNN model analyzes facial characteristics linked to autism traits.")

        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=90)
            st.markdown("### Questionnaire Screening")
            st.write("Q-CHAT-10 based score prediction using ML classifier.")

        with col3:
            st.image("https://cdn-icons-png.flaticon.com/512/3846/3846840.png", width=90)
            st.markdown("### Autism Assistant")
            st.write("Ask autism-related questions to our AI-powered chatbot.")

        st.warning(
            "Screening results are not diagnostic. "
            "Please consult a medical specialist for confirmation."
        )

def show_score_banner(score: float, label: str):
    if score >= 80:
        color = "#d9534f"  # 🔴 High Risk
    elif score >= 50:
        color = "#f0ad4e"  # 🟠 Moderate Risk
    else:
        color = "#5cb85c"  # 🟢 Low Risk
    
    st.markdown(
        f"""
        <div style="
            background:{color};
            padding:15px;
            border-radius:12px;
            text-align:center;
            font-weight:700;
            font-size:19px;
            color:white;
            width:100%;
            margin-top:15px;
        ">
            Autism likelihood: {score:.2f}% — {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===================== PAGE: IMAGE ANALYSIS =====================

def image_analysis_section():
    with st.container():

        st.title("🧠 Image-based Autism Screening (CNN)")

        mode = st.radio(
            "Select Input Method",
            ["Upload Image", "Use Camera"],
            index=0
        )

        img = None
        if mode == "Upload Image":
            file = st.file_uploader(
                "Upload a clear face image",
                type=["jpg", "jpeg", "png"]
            )
            if file:
                img = resize_safe(Image.open(file))
        else:
            cam = st.camera_input("Take a picture")
            if cam:
                img = resize_safe(Image.open(cam))

        show_cam = st.checkbox(
            "Show AI Highlight Map (High RAM usage)",
            value=False
        )

        if img is not None:
            st.image(img, caption="Uploaded Face", width=250)

            if st.button("Predict from Image"):

                with st.spinner("Loading AI model..."):
                    cnn_model = load_cnn_model()

                score = predict_image_from_pil(img, cnn_model)
                st.session_state.img_score = score
                label = risk_label(score)

                show_score_banner(score, label)

                if show_cam:
                    rgb = img.convert("RGB")
                    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
                    face_batch = preprocess_face(bgr)

                    heatmap = generate_gradcam(face_batch, cnn_model)
                    overlay = overlay_heatmap(bgr, heatmap)
                    overlay_display = add_asd_feature_boxes(overlay.copy())

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 🖼️ Original")
                        st.image(rgb, width=250)

                    with col2:
                        st.markdown("### 🔍 AI Highlight Map")
                        st.image(overlay_display, width=250)

                    st.info(
                        "Highlighted areas indicate facial regions analyzed by the AI."
                    )

                cleanup_tf()


def resize_safe(pil_img, max_size=512):
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil_img = pil_img.resize(
            (int(w * scale), int(h * scale)),
            Image.LANCZOS
        )
    return pil_img

# ===================== PAGE: QUESTIONNAIRE =====================

def questionnaire_section():
    with st.container():

        st.title("Behavioural Questionnaire-based Screening")
        st.write("Please answer the following questions about your child:")

    q = {}
    q['Q1'] = st.radio(
        "Is your child often attentive to faint sounds that others might not notice?",
        ["No", "Yes"],
    )
    q['Q2'] = st.radio(
        "Does your child typically focus more on the big picture rather than small details?",
        ["No", "Yes"],
    )
    q['Q3'] = st.radio(
        "Does your child struggle to handle multiple tasks at once?",
        ["No", "Yes"],
    )
    q['Q4'] = st.radio(
        "Does your child take a long time to resume tasks after an interruption?",
        ["No", "Yes"],
    )
    q['Q5'] = st.radio(
        "Does your child find it hard to 'read between the lines' in conversations?",
        ["No", "Yes"],
    )
    q['Q6'] = st.radio(
        "Does your child struggle to judge if someone is interested in what they are saying?",
        ["No", "Yes"],
    )
    q['Q7'] = st.radio(
        "Does your child struggle to understand characters' intentions in stories?",
        ["No", "Yes"],
    )
    q['Q8'] = st.radio(
        "Does your child show little interest in understanding people's feelings from facial expressions?",
        ["No", "Yes"],
    )
    q['Q9'] = st.radio(
        "Does your child find it challenging to infer other people's intentions?",
        ["No", "Yes"],
    )
    q['Q10'] = st.radio(
        "Does your child find it challenging to switch between different topics or tasks?",
        ["No", "Yes"],
    )

    age_options = [str(a) for a in range(12, 37)]
    q['age'] = st.selectbox("Age of your child (years)", age_options)

    q['gender'] = st.radio("Gender of your child", ["Male", "Female"])

    q['ethnicity'] = st.selectbox(
        "Ethnicity of your child",
        [
            "Hispanic",
            "Latino",
            "Native Indian",
            "Others",
            "Pacifica",
            "White Europian",
            "asian",
            "Black",
            "middle eastern",
            "mixed",
            "south asian",
        ],
    )

    q['jaundice'] = st.radio("Was your child born with jaundice?", ["No", "Yes"])
    q['family_history'] = st.radio(
        "Does any immediate family member have a history of ASD?",
        ["No", "Yes"],
    )
    q['completed_by'] = st.selectbox(
        "Who is completing this test?",
        ["Health", "others", "self", "family"],
    )

    if st.button("Predict from Questionnaire"):
        # Encode answers exactly as used when training nbmodel1.pkl
        q1 = ["No", "Yes"].index(q['Q1'])
        q2 = ["No", "Yes"].index(q['Q2'])
        q3 = ["No", "Yes"].index(q['Q3'])
        q4 = ["No", "Yes"].index(q['Q4'])
        q5 = ["No", "Yes"].index(q['Q5'])
        q6 = ["No", "Yes"].index(q['Q6'])
        q7 = ["No", "Yes"].index(q['Q7'])
        q8 = ["No", "Yes"].index(q['Q8'])
        q9 = ["No", "Yes"].index(q['Q9'])
        q10 = ["No", "Yes"].index(q['Q10'])

        age_idx = age_options.index(q['age'])
        gender = ["Male", "Female"].index(q['gender'])
        ethnicity = [
            "Hispanic",
            "Latino",
            "Native Indian",
            "Others",
            "Pacifica",
            "White Europian",
            "asian",
            "Black",
            "middle eastern",
            "mixed",
            "south asian",
        ].index(q['ethnicity'])
        jaundice = ["No", "Yes"].index(q['jaundice'])
        family = ["No", "Yes"].index(q['family_history'])
        who = ["Health", "others", "self", "family"].index(q['completed_by'])

        values = np.array(
            [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, age_idx, gender,
             ethnicity, jaundice, family, who],
            dtype=np.int64,
        )

        score = predict_questionnaire_vector(values)
        st.session_state.quest_score = score

        label = risk_label(score)
        show_score_banner(score, label)
# ===================== PAGE: CSV UPLOAD =====================

def csv_upload_section():
    with st.container():

        st.title("CSV Screening - Single or Bulk Prediction")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is None:
        return

    df = pd.read_csv(uploaded)

    if df.shape[0] > 300:
        st.error(
            "CSV too large for this system. "
            "Please upload a file with 300 rows or fewer."
        )
        return

    st.write(f"📌 Records found: {df.shape[0]}")

    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("/", "_")

    mode = st.radio(
        "Choose Mode:",
        ["Predict only one selected case", "Predict all cases in CSV"]
    )

    # Mapping dictionaries
    yesno = {"no": 0, "yes": 1, "0": 0, "1": 1}
    gender_map = {"m": 0, "male": 0, "f": 1, "female": 1}
    ethnicity_map = {
        "hispanic": 0, "latino": 1, "native indian": 2, "others": 3,
        "pacifica": 4, "white europian": 5, "asian": 6, "black": 7,
        "middle eastern": 8, "mixed": 9, "south asian": 10
    }
    who_map = {"health": 0, "others": 1, "self": 2, "family": 3}

    def preprocess_row(data):
        q_values = [yesno.get(data.get(f"a{i}", "0"), 0) for i in range(1, 11)]
        return np.array([
            *q_values,
            int(data.get("age_mons", 12)) - 12,
            gender_map.get(data.get("sex", "m"), 0),
            ethnicity_map.get(data.get("ethnicity", "others"), 3),
            yesno.get(data.get("jaundice", "no"), 0),
            yesno.get(data.get("family_mem_with_asd", "no"), 0),
            who_map.get(data.get("who_completed_the_test", "others"), 1)
        ], dtype=np.int64)

    # ---------- SINGLE ----------
    if mode == "Predict only one selected case":
        row_num = st.number_input(
            "Select Row Index to Predict",
            min_value=0, max_value=df.shape[0] - 1, value=0
        )

        st.dataframe(df.iloc[[row_num]])

        if st.button("Predict This Case"):
            data = df.iloc[row_num].astype(str).str.lower().str.strip()
            features = preprocess_row(data)
            score = predict_questionnaire_vector(features)
            st.session_state.csv_score = score
            show_score_banner(score, risk_label(score))

    # ---------- BULK ----------
    else:
        if st.button("Predict All"):
            features = np.array([
                preprocess_row(df.iloc[i].astype(str).str.lower().str.strip())
                for i in range(df.shape[0])
            ])

            probs = nb_model.predict_proba(features)[:, 1] * 100
            df["Autism_Probability(%)"] = probs
            df["Risk_Level"] = [risk_label(p) for p in probs]

            st.session_state.csv_score = float(np.mean(probs))

            st.success("Bulk predictions completed!")
            st.dataframe(df.head())

            st.download_button(
                "Download Predictions",
                df.to_csv(index=False).encode("utf-8"),
                "Autism_Predictions.csv",
                "text/csv"
            )


# ===================== COMMON CHART FUNCTION =====================

def display_score_chart(score, title="Autism Likelihood Score"):
    import matplotlib.pyplot as plt

    # Decide label + color based on score
    if score <= 40:
        color = "green"
        label = "Low Risk"
    elif score <= 60:
        color = "gold"
        label = "Moderate Risk"
    else:
        color = "red"
        label = "High Risk"

    # 🔹 Moderate professional size chart
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    
    ax.pie(
        [score, 100 - score],
        labels=[f"{label} ({score:.1f}%)", ""],
        colors=[color, "lightgrey"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    
    ax.set_title(title, fontsize=16)
    ax.axis("equal")  # Keep shape circular

    # Prevent full page stretch
    st.pyplot(fig, use_container_width=False)

# ===================== PAGE: COMBINED RESULT (INDEPENDENT) =====================

def combined_result_section():
    with st.container():

        st.title("📌 Combined Result")

    img_score = st.session_state.get("img_score")
    q_score = st.session_state.get("quest_score")
    csv_score = st.session_state.get("csv_score")

    available_scores = {
        "Image Analysis": img_score,
        "Questionnaire": q_score,
        "CSV Upload": csv_score
    }

    # No prediction yet?
    if all(v is None for v in available_scores.values()):
        st.warning("⚠️ Please complete a prediction first — Image / Questionnaire / CSV.")
        return

    st.markdown("### 🔍 Your Screening Results")

    # Show cards for only completed analyses
    for label, val in available_scores.items():
        if val is not None:
            st.info(f"**{label}:** {val:.2f}% — {risk_label(val)}")

    st.divider()

    # Final integrated score
    valid_scores = [v for v in available_scores.values() if v is not None]
    final_score = sum(valid_scores) / len(valid_scores)

    st.subheader("🧮 Final Combined Screening")
    st.success(f"Final Autism Prediction Score: **{final_score:.2f}%**")
    st.info(f"Risk Level: **{risk_label(final_score)}**")

    # Graph — Only for available predictions & smaller size
    labels = [k for k, v in available_scores.items() if v is not None]
    values = [v for v in available_scores.values() if v is not None]

    import matplotlib.pyplot as plt

    # Graph — Only for available predictions & smaller size
    labels = [k for k, v in available_scores.items() if v is not None]
    values = [v for v in available_scores.values() if v is not None]

    import matplotlib.pyplot as plt

    n = len(labels)
    fig_width = 4 + (1.2 * n)  # dynamic width

    fig, ax = plt.subplots(figsize=(fig_width, 5), dpi=110)

    bars = ax.bar(labels, values,
              color=["#1E88E5", "#43A047", "#FB8C00"][:n],
              width=0.55)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Likelihood (%)", fontsize=12)
    ax.set_title("Score Comparison", fontsize=18, weight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', labelsize=14)
    # Add percentage text above bars
    for bar, score in zip(bars, values):
        ax.text(
        bar.get_x() + bar.get_width() / 2,
        score + 2,
        f"{score:.1f}%",
        ha='center', va='bottom',
        fontsize=14, weight='bold'
    )

    fig.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=False)


def visualize():
    with st.container():

        st.title("📊 Visualize Prediction Results")

    img_score = st.session_state.get("img_score")
    q_score = st.session_state.get("quest_score")
    csv_score = st.session_state.get("csv_score")

    if img_score is None and q_score is None and csv_score is None:
        st.warning("⚠️ Please complete at least one prediction to view insights.")
        return

    # ================= SECTION 1 — Prediction Summary =================
    st.subheader("🔍 Prediction Summary")

    cols = st.columns(3)

    if img_score is not None:
        with cols[0]:
            st.markdown("### 🧠 Image Analysis")
            st.metric("Score", f"{img_score:.2f}%")
            st.caption(risk_label(img_score))

    if q_score is not None:
        with cols[1]:
            st.markdown("### 📋 Questionnaire")
            st.metric("Score", f"{q_score:.2f}%")
            st.caption(risk_label(q_score))

    if csv_score is not None:
        with cols[2]:
            st.markdown("### 📑 CSV Data")
            st.metric("Score", f"{csv_score:.2f}%")
            st.caption(risk_label(csv_score))

    st.divider()

    # ================= SECTION 2 — INDIVIDUAL ROADMAPS =================
    st.subheader("🗺️ Personalized Development Roadmaps")

    if img_score is not None:
        st.markdown("### 🧠 Image Analysis Roadmap")
        st.markdown("""
        **Purpose:** Support social attention and facial engagement  
        • Encourage face-to-face interaction during play  
        • Use imitation games (expressions, gestures)  
        • Observe eye contact consistency across environments  
        • Track changes over time using multiple images  
        """)

    if q_score is not None:
        st.markdown("### 📋 Questionnaire Roadmap")
        st.markdown("""
        **Purpose:** Support daily behavioral development  
        • Practice turn-taking and shared attention  
        • Encourage pretend play and storytelling  
        • Reduce pressure during task-switching  
        • Reinforce social cues calmly and consistently  
        """)

    if csv_score is not None:
        st.markdown("### 📑 CSV / Historical Data Roadmap")
        st.markdown("""
        **Purpose:** Support long-term observation  
        • Focus on patterns, not single values  
        • Monitor progress across weeks or months  
        • Compare before/after routine changes  
        • Maintain structured records for reference  
        """)

    st.divider()

    # ================= SECTION 3 — WHAT INFLUENCES THE SCORE =================
    st.subheader("📊 What Influences the Results")

    st.markdown("""
    • Image clarity, lighting, and facial orientation  
    • Accuracy and consistency of questionnaire responses  
    • Child’s age and natural developmental variation  
    • Environmental and social exposure  
    • Model confidence varies by data type  
    """)

    st.info(
        "These results are AI-based screening indicators, not a medical diagnosis. "
        "They are intended to support awareness and observation."
    )

# ===================== PAGE: AUTISM CHATBOT =====================
def chatbot_section():
    with st.container():

        st.title("🤖 Autism Support Chatbot")
        st.write("Ask autism-related questions for quick, reliable guidance.")
    question = st.text_input("Your Question:")

    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Finding the best answer..."):
            answer = google_autism_answer(question)

        if answer is None:
            st.warning("⚠ Please ask questions related to autism, ASD, early signs, or therapies only.")
            return

        st.markdown(
            f"""
            <div style="
                background:#ffffff;
                padding:14px 18px;
                border-left:5px solid #28a745;
                border-radius:8px;
                margin-top:12px;
                font-size:17px;
                line-height:1.45;
            ">
            <b>Answer:</b><br>{answer}
            </div>
            """,
            unsafe_allow_html=True
        )
# ===================== NAVIGATION =====================

selected = option_menu(
    menu_title=None,
    options=[
        "Home",
        "Image Analysis",
        "Questionnaire",
        "CSV Upload",
        "Combined Result",
        "Visualize",
        "Autism Chatbot",
    ],
    
    icons=["house", "camera", "list-task", "filetype-csv", 
           "bar-chart", "pie-chart", "robot"],
    orientation="horizontal",
    
)

def main():
    if selected == "Home":
        home_section()

    elif selected == "Image Analysis":
        image_analysis_section()

    elif selected == "Questionnaire":
        questionnaire_section()

    elif selected == "CSV Upload":
        csv_upload_section()

    elif selected == "Combined Result":
        combined_result_section()

    elif selected == "Visualize":
        visualize()

    elif selected == "Autism Chatbot":
        chatbot_section()

if __name__ == "__main__":
    main()
