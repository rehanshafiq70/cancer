"""
╔══════════════════════════════════════════════════════════════╗
║       SkinScan AI — Clinical Diagnostic System v12.0        ║
║       University of Agriculture Faisalabad | FYP            ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageFilter
import os, time, datetime, io, base64

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="SkinScan AI | Clinical Diagnostic System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════
# CSS — DARK MEDICAL THEME
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800;900&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:        #03080f;
    --bg2:       #061120;
    --card:      rgba(6,18,38,0.85);
    --border:    rgba(0,212,255,0.18);
    --teal:      #00d4ff;
    --green:     #00ff88;
    --red:       #ff3355;
    --yellow:    #ffcc00;
    --text:      #cce8ff;
    --muted:     #3a6080;
    --font1:     'Orbitron', monospace;
    --font2:     'Inter', sans-serif;
}

html, body, [data-testid="stAppViewContainer"], .main {
    background: var(--bg) !important;
    color: var(--text);
    font-family: var(--font2);
}

[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse 70% 35% at 50% 0%, rgba(0,212,255,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 40% 30% at 90% 80%, rgba(0,255,136,0.05) 0%, transparent 60%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061120 0%, #03080f 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.1) !important;
}

#MainMenu, footer, header, [data-testid="stDecoration"] { visibility: hidden; }
.stDeployButton { display: none; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #00d4ff44) !important;
    color: #00d4ff !important;
    border: 1px solid rgba(0,212,255,0.5) !important;
    border-radius: 8px !important;
    font-family: var(--font1) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #00d4ff66) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.3) !important;
    transform: translateY(-1px) !important;
}

/* Primary scan button — special */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #00d4ff, #0088ff) !important;
    color: #000 !important;
    border: none !important;
    font-size: 0.78rem !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.4) !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox > div > div {
    background: rgba(6,18,38,0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stMetricValue"] {
    color: var(--teal) !important;
    font-family: var(--font1) !important;
    font-size: 1.7rem !important;
}
[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0,212,255,0.12) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font1) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.05em !important;
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom: 2px solid var(--teal) !important;
    background: rgba(0,212,255,0.05) !important;
}

/* Progress */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00d4ff, #00ff88) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--teal); border-radius: 4px; }

hr { border-color: rgba(0,212,255,0.1) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font2) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════
def init_state():
    D = {
        "page":          "Home",
        "model":         None,
        "model_loaded":  False,
        "scan_history":  [],
        "total_scans":   0,
        "malignant_cnt": 0,
        "benign_cnt":    0,
        "last_result":   None,
    }
    for k, v in D.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════
# MODEL — gdown DOWNLOAD + LOAD
# ══════════════════════════════════════════════
GDRIVE_ID  = "18VE_D81425cZVYwAXjOn0gWti8_lZSML"
MODEL_FILE = "skin_cancer_model.h5"

@st.cache_resource(show_spinner=False)
def load_cnn_model():
    """Download from GDrive with gdown if needed, then load with Keras."""
    # ── Try TensorFlow import first ──
    try:
        import tensorflow as tf
    except Exception:
        return None, "demo"   # Python 3.14 / no TF → demo

    # ── Download if not cached ──
    if not os.path.exists(MODEL_FILE):
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
            gdown.download(url, MODEL_FILE, quiet=False)
        except Exception as e:
            # fallback: requests streaming
            try:
                import requests
                sess = requests.Session()
                URL  = "https://docs.google.com/uc?export=download"
                r    = sess.get(URL, params={"id": GDRIVE_ID}, stream=True)
                tok  = next((v for k, v in r.cookies.items()
                             if k.startswith("download_warning")), None)
                if tok:
                    r = sess.get(URL, params={"id": GDRIVE_ID, "confirm": tok}, stream=True)
                with open(MODEL_FILE, "wb") as f:
                    for chunk in r.iter_content(32768):
                        if chunk: f.write(chunk)
            except Exception as e2:
                return None, f"download_error: {e2}"

    # ── Load ──
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        return model, "loaded"
    except Exception as e:
        return None, f"load_error: {e}"

# ══════════════════════════════════════════════
# IMAGE VALIDATION
# ══════════════════════════════════════════════
def validate_image(img: Image.Image):
    """Returns (ok: bool, msg: str)"""
    # Size
    if img.width < 64 or img.height < 64:
        return False, "Image resolution too low (min 64×64 px)."
    # Mode
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32)
    # Brightness
    mean_b = arr.mean()
    if mean_b < 25:
        return False, "Image is too dark. Upload a properly lit dermoscopic image."
    if mean_b > 230:
        return False, "Image is overexposed. Upload a properly lit dermoscopic image."
    # Blur (edge variance)
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    variance = np.array(edges, dtype=np.float32).var()
    if variance < 40:
        return False, "Image appears blurry or out of focus. Upload a sharp dermoscopic image."
    return True, "ok"

# ══════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def predict(model, img: Image.Image, status: str) -> dict:
    tensor = preprocess(img)
    if model is not None and status == "loaded":
        raw = model.predict(tensor, verbose=0)
        if raw.shape[-1] == 1:
            mal = float(raw[0][0])
        else:
            mal = float(raw[0][1])
    else:
        # Demo — realistic simulation
        np.random.seed(int(time.time()) % 9999)
        mal = float(np.random.beta(2, 4))

    ben = 1.0 - mal
    label = "Malignant" if mal >= 0.5 else "Benign"
    conf  = round(max(mal, ben) * 100, 1)

    if label == "Malignant":
        risk = "Critical" if conf >= 80 else "Medium"
    else:
        risk = "Low" if conf >= 70 else "Medium"

    return {
        "label":    label,
        "conf":     conf,
        "mal_pct":  round(mal * 100, 1),
        "ben_pct":  round(ben * 100, 1),
        "risk":     risk,
        "demo":     status != "loaded",
    }

# ══════════════════════════════════════════════
# REPORT GENERATOR
# ══════════════════════════════════════════════
def make_report(pat: dict, res: dict) -> str:
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rid = f"RPT-{int(time.time())}"
    if res["label"] == "Malignant":
        tx = "• Immediate oncology referral\n• Surgical biopsy recommended\n• Avoid UV exposure\n• Follow up within 7 days"
        rec = "URGENT: Consult a dermatologist / oncologist immediately."
    else:
        tx  = "• Topical moisturiser if symptomatic\n• SPF 50+ daily sunscreen\n• Annual skin check\n• Self-examination monthly"
        rec = "Routine monitoring advised. Recheck in 6–12 months."

    return f"""
╔══════════════════════════════════════════════════════════╗
         SKINSCAN AI — CLINICAL DIAGNOSTIC REPORT
╚══════════════════════════════════════════════════════════╝
Report ID      : {rid}
Generated      : {ts}
System Version : SkinScan AI v12.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PATIENT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name           : {pat.get('name','N/A')}
Age            : {pat.get('age','N/A')}
Gender         : {pat.get('gender','N/A')}
Patient ID     : {pat.get('pid','N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AI DIAGNOSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classification : {res['label'].upper()}
Confidence     : {res['conf']}%
Malignant Prob : {res['mal_pct']}%
Benign Prob    : {res['ben_pct']}%
Risk Level     : {res['risk'].upper()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CLINICAL RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{rec}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TREATMENT PLAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{tx}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This report is generated by an AI system and is intended
to assist clinicians only. It does NOT replace professional
medical diagnosis. Always confirm with a licensed physician.

──────────────────────────────────────────────────────────
University of Agriculture Faisalabad | Rehan Shafique
rehanshafiq6540@gmail.com | SkinScan AI v12.0
══════════════════════════════════════════════════════════
""".strip()

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="padding:24px 16px 16px;text-align:center;">
          <div style="display:inline-flex;align-items:center;justify-content:center;
                      width:58px;height:58px;border-radius:14px;
                      background:linear-gradient(135deg,#00d4ff33,#00ff8833);
                      border:1px solid rgba(0,212,255,0.4);
                      margin-bottom:10px;">
            <span style="font-size:28px;">🧬</span>
          </div>
          <div style="font-family:'Orbitron',monospace;font-size:1rem;font-weight:700;
                      color:#00d4ff;letter-spacing:0.05em;">SkinScan AI</div>
          <div style="font-size:0.6rem;color:#3a6080;letter-spacing:0.15em;
                      text-transform:uppercase;margin-top:3px;">Clinical Diagnostic v12.0</div>
        </div>
        """, unsafe_allow_html=True)

        # Model status
        if st.session_state.model_loaded:
            model, status = st.session_state.model, "loaded"
        else:
            model, status = None, "not_loaded"

        is_live = st.session_state.model_loaded and st.session_state.model is not None
        dot_col = "#00ff88" if is_live else "#ffcc00"
        dot_lbl = "Model Online" if is_live else "Demo Mode"
        st.markdown(f"""
        <div style="margin:0 12px 14px;padding:7px 12px;border-radius:8px;
                    background:rgba(0,255,136,0.05);border:1px solid rgba(0,255,136,0.15);
                    display:flex;align-items:center;gap:8px;">
          <span style="width:7px;height:7px;border-radius:50%;background:{dot_col};
                       box-shadow:0 0 8px {dot_col};display:inline-block;"></span>
          <span style="font-size:0.72rem;color:{dot_col};
                       font-family:'Orbitron',monospace;">{dot_lbl}</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        nav = {
            "🏠  Home":            "Home",
            "🔬  AI Scan":         "Scan",
            "📊  Dashboard":       "Dashboard",
            "🗂️  Patient Registry": "Registry",
            "📘  User Guide":      "Guide",
        }
        for label, key in nav.items():
            active = st.session_state.page == key
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding:0 8px 8px;font-size:0.68rem;color:#3a6080;line-height:1.8;">
          <div>🔬 Total Scans: <b style="color:#00d4ff;">{st.session_state.total_scans}</b></div>
          <div>⚠️ Malignant:  <b style="color:#ff3355;">{st.session_state.malignant_cnt}</b></div>
          <div>✅ Benign:     <b style="color:#00ff88;">{st.session_state.benign_cnt}</b></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════
def page_home():
    # ── HERO ──
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 2rem;">
      <div style="display:inline-flex;align-items:center;justify-content:center;
                  width:90px;height:90px;border-radius:22px;
                  background:linear-gradient(135deg,#00d4ff22,#00ff8822);
                  border:1.5px solid rgba(0,212,255,0.4);
                  box-shadow:0 0 40px rgba(0,212,255,0.15);
                  margin-bottom:1.2rem;">
        <span style="font-size:44px;">🧬</span>
      </div>
      <h1 style="font-family:'Orbitron',monospace;font-size:2.6rem;font-weight:900;
                 background:linear-gradient(90deg,#00d4ff,#00ff88);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 letter-spacing:0.04em;margin-bottom:0.5rem;">SkinScan AI</h1>
      <div style="font-size:1rem;color:#5a90b0;letter-spacing:0.15em;
                  text-transform:uppercase;margin-bottom:0.8rem;">
        Clinical Diagnostic System v12.0
      </div>
      <div style="font-size:1.1rem;color:#8ab8d0;max-width:600px;
                  margin:0 auto 2rem;line-height:1.7;">
        Hospital-grade AI skin cancer detection powered by Deep Learning.<br>
        Instant <b style="color:#00d4ff;">Benign</b> vs 
        <b style="color:#ff3355;">Malignant</b> classification with clinical reports.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── START BUTTON ──
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("⚡  START DIAGNOSIS", use_container_width=True, type="primary"):
            st.session_state.page = "Scan"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── STEP BY STEP GUIDE ──
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem;">
      <span style="font-family:'Orbitron',monospace;font-size:0.85rem;
                   color:#00d4ff;letter-spacing:0.15em;text-transform:uppercase;">
        📘 How It Works
      </span>
    </div>""", unsafe_allow_html=True)

    steps_data = [
        ("01", "🖼️", "Upload Image",    "Drag & drop a dermoscopic skin lesion image (JPG/PNG)"),
        ("02", "⚡", "Run AI Scan",     "Click Execute Deep Scan to trigger the CNN model"),
        ("03", "🧠", "AI Processing",  "Model analyses the lesion in real-time"),
        ("04", "📊", "View Result",     "See diagnosis, confidence %, and risk level"),
        ("05", "📄", "Export Report",   "Download medical report or export patient CSV"),
    ]

    cols = st.columns(5)
    for i, (num, icon, title, desc) in enumerate(steps_data):
        with cols[i]:
            st.markdown(f"""
            <div style="background:rgba(6,18,38,0.8);border:1px solid rgba(0,212,255,0.15);
                        border-radius:14px;padding:1.2rem 0.8rem;text-align:center;
                        height:160px;">
              <div style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:900;
                          color:rgba(0,212,255,0.2);margin-bottom:6px;">{num}</div>
              <div style="font-size:1.5rem;margin-bottom:8px;">{icon}</div>
              <div style="font-family:'Orbitron',monospace;font-size:0.6rem;
                          color:#00d4ff;font-weight:700;letter-spacing:0.05em;
                          margin-bottom:6px;">{title}</div>
              <div style="font-size:0.72rem;color:#3a6080;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── FEATURES SECTION ──
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem;">
      <span style="font-family:'Orbitron',monospace;font-size:0.85rem;
                   color:#00d4ff;letter-spacing:0.15em;text-transform:uppercase;">
        🔬 Key Features
      </span>
    </div>""", unsafe_allow_html=True)

    feats = [
        ("🧠", "CNN Detection",       "Deep learning model trained on melanoma dermoscopy dataset"),
        ("⚡", "Real-time Prediction", "Instant results with confidence scoring and risk levels"),
        ("📄", "Medical Reports",      "Auto-generated clinical reports ready for download"),
        ("📊", "Analytics Dashboard",  "Full scan history, charts, and statistical insights"),
        ("🗂️", "Patient Registry",     "Structured patient data management and CSV export"),
        ("✅", "Image Validation",      "Auto-rejects blurry, dark, or low-quality images"),
    ]
    fc1, fc2, fc3 = st.columns(3)
    for i, (icon, title, desc) in enumerate(feats):
        col = [fc1, fc2, fc3][i % 3]
        with col:
            st.markdown(f"""
            <div style="background:rgba(6,18,38,0.7);border:1px solid rgba(0,212,255,0.12);
                        border-radius:12px;padding:1rem 1.1rem;margin-bottom:0.8rem;
                        display:flex;gap:12px;align-items:flex-start;">
              <span style="font-size:1.5rem;flex-shrink:0;">{icon}</span>
              <div>
                <div style="font-family:'Orbitron',monospace;font-size:0.65rem;
                            color:#00d4ff;font-weight:700;margin-bottom:4px;">{title}</div>
                <div style="font-size:0.75rem;color:#3a6080;line-height:1.5;">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── DATASET INFO ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.2rem;">
      <span style="font-family:'Orbitron',monospace;font-size:0.85rem;
                   color:#00d4ff;letter-spacing:0.15em;text-transform:uppercase;">
        📂 Dataset Information
      </span>
    </div>""", unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    for col, val, lbl in [
        (d1, "10,015",  "Total Images"),
        (d2, "6,705",   "Benign Cases"),
        (d3, "3,310",   "Malignant Cases"),
        (d4, "Kaggle",  "Source"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:rgba(6,18,38,0.8);border:1px solid rgba(0,212,255,0.15);
                        border-radius:12px;padding:1rem;text-align:center;">
              <div style="font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:800;
                          color:#00d4ff;">{val}</div>
              <div style="font-size:0.72rem;color:#3a6080;margin-top:4px;">{lbl}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE: AI SCAN
# ══════════════════════════════════════════════
def page_scan():
    st.markdown("""
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:800;
               color:#00d4ff;margin-bottom:0.3rem;">🔬 AI Skin Analysis</h1>
    <p style="color:#3a6080;font-size:0.85rem;margin-bottom:1.5rem;">
        Upload a dermoscopic image for CNN-powered skin cancer detection
    </p>""", unsafe_allow_html=True)

    # ── Load model once ──
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Initialising AI model…"):
            m, status = load_cnn_model()
            st.session_state.model        = m
            st.session_state.model_status = status
            st.session_state.model_loaded = True

    model  = st.session_state.model
    status = getattr(st.session_state, "model_status", "demo")

    # Status badge
    if status == "loaded":
        st.markdown("""
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.25);
                    border-radius:8px;padding:6px 14px;margin-bottom:1rem;">
          <span style="width:7px;height:7px;border-radius:50%;background:#00ff88;
                       box-shadow:0 0 8px #00ff88;display:inline-block;"></span>
          <span style="font-size:0.72rem;color:#00ff88;
                       font-family:'Orbitron',monospace;">CNN Model Online</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:rgba(255,204,0,0.08);border:1px solid rgba(255,204,0,0.25);
                    border-radius:8px;padding:6px 14px;margin-bottom:1rem;">
          <span style="width:7px;height:7px;border-radius:50%;background:#ffcc00;
                       box-shadow:0 0 8px #ffcc00;display:inline-block;"></span>
          <span style="font-size:0.72rem;color:#ffcc00;
                       font-family:'Orbitron',monospace;">Demo Mode
                       (TF not available on Python 3.14)</span>
        </div>""", unsafe_allow_html=True)

    # ── Patient form ──
    with st.expander("👤 Patient Information", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        p_name   = c1.text_input("Full Name",   placeholder="e.g. Ahmed Ali")
        p_age    = c2.number_input("Age", 1, 120, 30)
        p_gender = c3.selectbox("Gender", ["Male", "Female", "Other"])
        p_id     = c4.text_input("Patient ID",  placeholder="PT-001")

    # ── Upload ──
    st.markdown("#### 🖼️ Upload Dermoscopic Image")
    uploaded = st.file_uploader(
        "Supported: JPG, JPEG, PNG, BMP",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

    if not uploaded:
        st.markdown("""
        <div style="border:2px dashed rgba(0,212,255,0.2);border-radius:14px;
                    padding:3rem;text-align:center;color:#3a6080;">
          <div style="font-size:3rem;margin-bottom:0.5rem;">🖼️</div>
          <div style="font-size:0.85rem;">Drag & drop a dermoscopic skin image here</div>
          <div style="font-size:0.72rem;margin-top:6px;">JPG · PNG · BMP · min 64×64px</div>
        </div>""", unsafe_allow_html=True)
        return

    img = Image.open(uploaded)

    col_img, col_ctrl = st.columns([1, 1])

    with col_img:
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.markdown(f"""
        <div style="font-size:0.72rem;color:#3a6080;margin-top:4px;">
          Resolution: {img.width}×{img.height}px | Mode: {img.mode}
        </div>""", unsafe_allow_html=True)

    with col_ctrl:
        # Validate
        ok, msg = validate_image(img)
        if not ok:
            st.markdown(f"""
            <div style="background:rgba(255,51,85,0.08);border:1px solid rgba(255,51,85,0.3);
                        border-radius:10px;padding:1rem 1.2rem;color:#ff3355;">
              <b>❌ Invalid Image</b><br>
              <span style="font-size:0.82rem;">{msg}</span><br><br>
              <span style="font-size:0.78rem;color:#3a6080;">
                Please upload a clear dermoscopic skin image.
              </span>
            </div>""", unsafe_allow_html=True)
            return

        st.markdown("""
        <div style="background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.2);
                    border-radius:10px;padding:0.7rem 1rem;color:#00ff88;
                    font-size:0.8rem;margin-bottom:1rem;">
          ✅ Image quality validated — ready for scan
        </div>""", unsafe_allow_html=True)

        if p_age < 1:
            st.warning("Please fill in patient information above.")

        conf_warn = status != "loaded"
        if conf_warn:
            st.markdown("""
            <div style="background:rgba(255,204,0,0.06);border:1px solid rgba(255,204,0,0.2);
                        border-radius:8px;padding:0.6rem 1rem;font-size:0.75rem;
                        color:#ffcc00;margin-bottom:1rem;">
              ⚠️ Running in Demo Mode — install TensorFlow locally for real predictions.
            </div>""", unsafe_allow_html=True)

        run = st.button("⚡  EXECUTE DEEP SCAN", use_container_width=True, type="primary")

    if run:
        # Animated progress
        prog = st.progress(0, text="Initialising neural network…")
        phases = [
            (15, "Preprocessing image…"),
            (35, "Extracting feature maps…"),
            (60, "Running convolutional layers…"),
            (80, "Computing softmax output…"),
            (100, "Finalising diagnosis…"),
        ]
        for pct, txt in phases:
            time.sleep(0.25)
            prog.progress(pct, text=txt)
        prog.empty()

        res = predict(model, img, status)

        # ── LOW CONFIDENCE WARNING ──
        if res["conf"] < 60:
            st.markdown(f"""
            <div style="background:rgba(255,204,0,0.07);border:1px solid rgba(255,204,0,0.3);
                        border-radius:10px;padding:0.8rem 1.2rem;color:#ffcc00;
                        font-size:0.8rem;margin-bottom:1rem;">
              ⚠️ <b>Low Reliability Warning</b> — Confidence is {res['conf']}% (&lt;60%).
              Result may be inconclusive. Please upload a higher-quality image.
            </div>""", unsafe_allow_html=True)

        # ── RESULT CARD ──
        label_color = "#ff3355" if res["label"] == "Malignant" else "#00ff88"
        risk_color  = {"Critical": "#ff3355", "Medium": "#ffcc00", "Low": "#00ff88"}[res["risk"]]
        icon_label  = "⚠️" if res["label"] == "Malignant" else "✅"

        st.markdown(f"""
        <div style="background:rgba(6,18,38,0.9);border:1.5px solid {label_color}44;
                    border-radius:16px;padding:1.6rem 1.8rem;
                    box-shadow:0 0 30px {label_color}15;margin-top:0.5rem;">

          <div style="display:flex;align-items:center;gap:14px;margin-bottom:1.2rem;">
            <div style="width:52px;height:52px;border-radius:12px;
                        background:{label_color}18;border:1px solid {label_color}44;
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.6rem;">{icon_label}</div>
            <div>
              <div style="font-family:'Orbitron',monospace;font-size:1.4rem;
                          font-weight:800;color:{label_color};">{res['label'].upper()}</div>
              <div style="font-size:0.75rem;color:#3a6080;">AI Classification</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <div style="font-family:'Orbitron',monospace;font-size:2rem;
                          font-weight:900;color:#cce8ff;">{res['conf']}%</div>
              <div style="font-size:0.7rem;color:#3a6080;">Confidence</div>
            </div>
          </div>

          <div style="height:5px;background:rgba(255,255,255,0.05);
                      border-radius:5px;margin-bottom:1.2rem;">
            <div style="height:100%;width:{res['conf']}%;
                        background:linear-gradient(90deg,{label_color}66,{label_color});
                        border-radius:5px;"></div>
          </div>

          <div style="display:flex;gap:10px;">
            <div style="flex:1;background:rgba(255,255,255,0.03);border-radius:8px;
                        padding:8px 12px;border:1px solid rgba(255,255,255,0.05);">
              <div style="font-size:0.65rem;color:#3a6080;text-transform:uppercase;
                          letter-spacing:0.1em;">Risk Level</div>
              <div style="font-family:'Orbitron',monospace;font-size:0.9rem;
                          font-weight:700;color:{risk_color};">{res['risk']}</div>
            </div>
            <div style="flex:1;background:rgba(255,255,255,0.03);border-radius:8px;
                        padding:8px 12px;border:1px solid rgba(255,255,255,0.05);">
              <div style="font-size:0.65rem;color:#3a6080;text-transform:uppercase;
                          letter-spacing:0.1em;">Malignant</div>
              <div style="font-family:'Orbitron',monospace;font-size:0.9rem;
                          font-weight:700;color:#ff3355;">{res['mal_pct']}%</div>
            </div>
            <div style="flex:1;background:rgba(255,255,255,0.03);border-radius:8px;
                        padding:8px 12px;border:1px solid rgba(255,255,255,0.05);">
              <div style="font-size:0.65rem;color:#3a6080;text-transform:uppercase;
                          letter-spacing:0.1em;">Benign</div>
              <div style="font-family:'Orbitron',monospace;font-size:0.9rem;
                          font-weight:700;color:#00ff88;">{res['ben_pct']}%</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ANALYSIS CHARTS ──
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_bar = go.Figure(go.Bar(
                x=["Benign", "Malignant"],
                y=[res["ben_pct"], res["mal_pct"]],
                marker_color=["#00ff88", "#ff3355"],
                text=[f"{res['ben_pct']}%", f"{res['mal_pct']}%"],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Probability Distribution",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cce8ff", font_family="Orbitron",
                yaxis=dict(range=[0, 110], showgrid=False, color="#3a6080"),
                xaxis=dict(color="#3a6080"),
                height=280
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with ch2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["conf"],
                title={"text": "Confidence Gauge", "font": {"color": "#cce8ff", "family": "Orbitron"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#3a6080"},
                    "bar":  {"color": label_color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0,  50],  "color": "rgba(255,51,85,0.1)"},
                        {"range": [50, 75],  "color": "rgba(255,204,0,0.1)"},
                        {"range": [75, 100], "color": "rgba(0,255,136,0.1)"},
                    ],
                },
                number={"suffix": "%", "font": {"color": "#cce8ff", "family": "Orbitron"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cce8ff", height=280
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── MEDICAL RECOMMENDATIONS ──
        if res["label"] == "Malignant":
            st.markdown(f"""
            <div style="background:rgba(255,51,85,0.06);border:1px solid rgba(255,51,85,0.3);
                        border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
              <div style="font-family:'Orbitron',monospace;font-size:0.75rem;
                          color:#ff3355;font-weight:700;margin-bottom:0.7rem;">
                🚨 MALIGNANT DETECTED — URGENT ACTION REQUIRED
              </div>
              <div style="font-size:0.82rem;color:#8ab8d0;line-height:2;">
                ▸ Immediate oncology referral<br>
                ▸ Surgical biopsy strongly recommended<br>
                ▸ Avoid direct UV / sun exposure<br>
                ▸ Follow-up appointment within 7 days<br>
                ▸ Do not attempt self-treatment
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(0,255,136,0.05);border:1px solid rgba(0,255,136,0.2);
                        border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
              <div style="font-family:'Orbitron',monospace;font-size:0.75rem;
                          color:#00ff88;font-weight:700;margin-bottom:0.7rem;">
                ✅ BENIGN — ROUTINE MONITORING ADVISED
              </div>
              <div style="font-size:0.82rem;color:#8ab8d0;line-height:2;">
                ▸ Schedule annual skin screening<br>
                ▸ Apply SPF 50+ sunscreen daily<br>
                ▸ Perform monthly self-examination<br>
                ▸ Consult dermatologist if lesion changes<br>
                ▸ Maintain a healthy lifestyle
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Save record ──
        patient = {
            "name":   p_name or "Unknown",
            "age":    p_age,
            "gender": p_gender,
            "pid":    p_id or f"PT-{st.session_state.total_scans+1:03d}",
        }
        st.session_state.total_scans   += 1
        if res["label"] == "Malignant": st.session_state.malignant_cnt += 1
        else:                           st.session_state.benign_cnt    += 1
        st.session_state.scan_history.append({
            "scan_id":    f"SC-{st.session_state.total_scans:03d}",
            "patient_id": patient["pid"],
            "name":       patient["name"],
            "diagnosis":  res["label"],
            "confidence": f"{res['conf']}%",
            "risk":       res["risk"],
            "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        st.session_state.last_result = (patient, res)

        # ── ACTION BUTTONS ──
        st.markdown("#### 📋 Actions")
        ba, bb, bc, bd = st.columns(4)

        report_txt = make_report(patient, res)
        with ba:
            st.download_button("📄 Medical Report",
                               report_txt,
                               f"Report_{patient['pid']}.txt",
                               "text/plain",
                               use_container_width=True)
        with bb:
            st.success(f"💾 Record saved — {patient['pid']}")

        with bc:
            if st.button("📊 View Dashboard", use_container_width=True):
                st.session_state.page = "Dashboard"
                st.rerun()

        with bd:
            if st.session_state.scan_history:
                df_csv = pd.DataFrame(st.session_state.scan_history).to_csv(index=False)
                st.download_button("📥 Export CSV",
                                   df_csv, "SkinScan_Records.csv",
                                   "text/csv", use_container_width=True)

        # ── AUTO DISCLAIMER ──
        st.markdown("""
        <div style="margin-top:1rem;padding:0.8rem 1rem;
                    background:rgba(255,204,0,0.05);border:1px solid rgba(255,204,0,0.15);
                    border-radius:8px;font-size:0.72rem;color:#3a6080;line-height:1.7;">
          <b style="color:#ffcc00;">⚠️ Medical Disclaimer:</b>
          This AI result is intended to assist clinicians only and does NOT replace professional
          medical diagnosis. Always confirm findings with a licensed dermatologist or oncologist.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════
def page_dashboard():
    st.markdown("""
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:800;
               color:#00d4ff;margin-bottom:0.3rem;">📊 Analytics Dashboard</h1>
    <p style="color:#3a6080;font-size:0.85rem;margin-bottom:1.5rem;">
        Real-time diagnostic metrics and session analytics
    </p>""", unsafe_allow_html=True)

    history = st.session_state.scan_history
    total   = st.session_state.total_scans
    mal     = st.session_state.malignant_cnt
    ben     = st.session_state.benign_cnt
    confs   = [float(h["confidence"].replace("%", "")) for h in history] if history else []
    avg_c   = round(np.mean(confs), 1) if confs else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔬 Total Scans",        total)
    c2.metric("🎯 Avg Confidence",     f"{avg_c}%")
    c3.metric("⚠️ Malignant Cases",    mal)
    c4.metric("✅ Benign Cases",        ben)

    st.markdown("<br>", unsafe_allow_html=True)

    if not history:
        st.markdown("""
        <div style="background:rgba(6,18,38,0.7);border:1px solid rgba(0,212,255,0.12);
                    border-radius:14px;padding:3rem;text-align:center;">
          <div style="font-size:2.5rem;margin-bottom:0.6rem;">📊</div>
          <div style="color:#3a6080;">No scan data yet. Go to AI Scan to get started.</div>
        </div>""", unsafe_allow_html=True)
        return

    col_l, col_r = st.columns(2)

    with col_l:
        fig_pie = px.pie(
            names=["Benign", "Malignant"],
            values=[ben, mal],
            color_discrete_sequence=["#00ff88", "#ff3355"],
            hole=0.55, title="Diagnosis Distribution"
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cce8ff", title_font_family="Orbitron",
            legend_font_color="#3a6080"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        colors_bar = ["#ff3355" if h["diagnosis"] == "Malignant" else "#00ff88"
                      for h in history[-20:]]
        fig_bar = go.Figure(go.Bar(
            y=confs[-20:],
            x=list(range(1, len(confs[-20:]) + 1)),
            marker_color=colors_bar
        ))
        fig_bar.update_layout(
            title="Confidence per Scan (last 20)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cce8ff", title_font_family="Orbitron",
            xaxis=dict(showgrid=False, color="#3a6080"),
            yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.04)", color="#3a6080"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Trend line
    if len(confs) >= 2:
        fig_line = go.Figure(go.Scatter(
            y=confs, mode="lines+markers",
            line=dict(color="#00d4ff", width=2),
            marker=dict(color="#00d4ff", size=5),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"
        ))
        fig_line.update_layout(
            title="Confidence Trend",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cce8ff", title_font_family="Orbitron",
            xaxis=dict(showgrid=False, color="#3a6080"),
            yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.04)", color="#3a6080"),
        )
        st.plotly_chart(fig_line, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE: PATIENT REGISTRY
# ══════════════════════════════════════════════
def page_registry():
    st.markdown("""
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:800;
               color:#00d4ff;margin-bottom:0.3rem;">🗂️ Patient Registry</h1>
    <p style="color:#3a6080;font-size:0.85rem;margin-bottom:1.5rem;">
        Scan records and patient management
    </p>""", unsafe_allow_html=True)

    if not st.session_state.scan_history:
        st.markdown("""
        <div style="background:rgba(6,18,38,0.7);border:1px solid rgba(0,212,255,0.12);
                    border-radius:14px;padding:3rem;text-align:center;">
          <div style="font-size:2.5rem;margin-bottom:0.6rem;">🗂️</div>
          <div style="color:#3a6080;">No patient records yet.</div>
        </div>""", unsafe_allow_html=True)
        return

    df = pd.DataFrame(st.session_state.scan_history)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button("📥 Export All Records (CSV)", csv,
                       "SkinScan_Registry.csv", "text/csv")

# ══════════════════════════════════════════════
# PAGE: USER GUIDE
# ══════════════════════════════════════════════
def page_guide():
    st.markdown("""
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:800;
               color:#00d4ff;margin-bottom:0.3rem;">📘 User Guide</h1>
    <p style="color:#3a6080;font-size:0.85rem;margin-bottom:1.5rem;">
        Step-by-step system walkthrough
    </p>""", unsafe_allow_html=True)

    steps = [
        ("01", "👤", "Enter Patient Details",
         "Fill in the patient name, age, gender, and ID in the AI Scan page before uploading an image."),
        ("02", "🖼️", "Upload Dermoscopic Image",
         "Upload a clear, well-lit JPEG or PNG dermoscopic skin image. Minimum resolution: 64x64 pixels."),
        ("03", "⚡", "Execute Deep Scan",
         "Click the EXECUTE DEEP SCAN button. The CNN model will process the image through multiple layers."),
        ("04", "🧬", "View AI Result",
         "Review the Malignant or Benign classification, confidence percentage, risk level, and probability bars."),
        ("05", "📄", "Download Medical Report",
         "Click Generate Medical Report to download a complete clinical report (.txt) for your records."),
        ("06", "📊", "View Dashboard",
         "Navigate to the Dashboard to see analytics, charts, and historical scan statistics."),
        ("07", "📥", "Export Patient Data",
         "Export all session scan records as a CSV file from the Registry page."),
    ]

    for num, icon, title, desc in steps:
        st.markdown(f"""
        <div style="background:rgba(6,18,38,0.75);border:1px solid rgba(0,212,255,0.15);
                    border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:0.8rem;
                    display:flex;align-items:flex-start;gap:14px;">
          <div style="min-width:42px;height:42px;border-radius:10px;
                      background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.25);
                      display:flex;align-items:center;justify-content:center;
                      font-size:1.3rem;flex-shrink:0;">{icon}</div>
          <div>
            <div style="font-family:'Orbitron',monospace;font-size:0.72rem;font-weight:700;
                        color:#00d4ff;margin-bottom:4px;">STEP {num} — {title.upper()}</div>
            <div style="font-size:0.82rem;color:#5a90b0;line-height:1.65;">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(255,204,0,0.05);border:1px solid rgba(255,204,0,0.2);
                border-radius:12px;padding:1rem 1.4rem;margin-top:0.5rem;">
      <div style="font-family:'Orbitron',monospace;font-size:0.7rem;
                  color:#ffcc00;margin-bottom:6px;">⚠️ CLINICAL DISCLAIMER</div>
      <div style="font-size:0.8rem;color:#3a6080;line-height:1.7;">
        SkinScan AI is an assistive diagnostic tool designed to support, not replace,
        professional dermatological diagnosis. All results must be confirmed by a
        licensed clinician before initiating any treatment.
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════
def render_footer():
    st.markdown("<br><br><hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;
                font-family:'Orbitron',monospace;font-size:0.6rem;
                color:#1a3a5a;letter-spacing:0.1em;line-height:2.2;">
      <div>University of Agriculture Faisalabad</div>
      <div style="color:#00d4ff22;">Rehan Shafique &nbsp;|&nbsp; rehanshafiq6540@gmail.com</div>
      <div>SkinScan AI — Medical Diagnostic System v12.0</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    render_sidebar()

    page = st.session_state.page
    if   page == "Home":      page_home()
    elif page == "Scan":      page_scan()
    elif page == "Dashboard": page_dashboard()
    elif page == "Registry":  page_registry()
    elif page == "Guide":     page_guide()

    render_footer()

if __name__ == "__main__":
    main()
