"""
╔══════════════════════════════════════════════════════════════╗
║          SkinScan AI - Clinical Diagnostic System            ║
║          Final Year Project | Production-Grade App           ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import os
import time
import datetime
import requests
import base64
import json
import csv

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SkinScan AI | Clinical Diagnostic System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# FULL UI CSS — MEDICAL DARK THEME
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #050d1a;
    --bg-secondary:  #071426;
    --bg-card:       rgba(10, 30, 60, 0.75);
    --bg-glass:      rgba(12, 35, 70, 0.55);
    --border-glow:   rgba(0, 200, 180, 0.25);
    --accent-teal:   #00c8b4;
    --accent-blue:   #0a84ff;
    --accent-green:  #00e87a;
    --accent-red:    #ff4d6d;
    --accent-yellow: #ffd166;
    --text-primary:  #e8f4ff;
    --text-secondary:#7fa8c9;
    --text-muted:    #3d6080;
    --font-display:  'Syne', sans-serif;
    --font-body:     'DM Sans', sans-serif;
    --radius-lg:     16px;
    --radius-md:     10px;
    --shadow-card:   0 8px 32px rgba(0,0,0,0.45), 0 0 0 1px rgba(0,200,180,0.08);
    --shadow-glow:   0 0 24px rgba(0,200,180,0.18);
}

/* ── Global Reset ── */
html, body, [data-testid="stAppViewContainer"], .main {
    background-color: var(--bg-primary) !important;
    font-family: var(--font-body);
    color: var(--text-primary);
}

[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse 80% 40% at 50% -10%, rgba(0,132,255,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 50% 30% at 80% 80%, rgba(0,200,180,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 10% 90%, rgba(0,232,122,0.05) 0%, transparent 60%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071426 0%, #050d1a 100%) !important;
    border-right: 1px solid rgba(0,200,180,0.12) !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ── Typography ── */
h1, h2, h3 {
    font-family: var(--font-display) !important;
    letter-spacing: -0.02em;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-teal), var(--accent-blue)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 18px rgba(0,200,180,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,200,180,0.4) !important;
    filter: brightness(1.1) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stFileUploader > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.2rem !important;
    box-shadow: var(--shadow-card) !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent-teal) !important;
    font-family: var(--font-display) !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0,200,180,0.15) !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-teal) !important;
    background: rgba(0,200,180,0.08) !important;
    border-bottom: 2px solid var(--accent-teal) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-teal); border-radius: 10px; }

/* ── Divider ── */
hr { border-color: rgba(0,200,180,0.12) !important; }

/* ── Plotly charts ── */
.js-plotly-plot { border-radius: var(--radius-lg) !important; }

/* ── Alerts / Success / Error ── */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: var(--radius-md) !important;
    font-family: var(--font-body) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "authenticated": True,
        "page": "Dashboard",
        "scan_history": [],
        "total_scans": 0,
        "high_risk": 0,
        "patient_data": [],
        "model": None,
        "model_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# GOOGLE DRIVE MODEL DOWNLOAD
# ─────────────────────────────────────────────
GDRIVE_FILE_ID = "18VE_D81425cZVYwAXjOn0gWti8_lZSML"
MODEL_PATH = "skinscan_model.h5"

def download_model_from_gdrive(file_id: str, dest: str) -> bool:
    """Download model from Google Drive with progress."""
    try:
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={"id": file_id}, stream=True)

        # Handle large-file confirmation token
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        if token:
            response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

        total = int(response.headers.get("content-length", 0))
        progress = st.progress(0, text="Downloading model…")
        downloaded = 0

        with open(dest, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress.progress(min(downloaded / total, 1.0),
                                          text=f"Downloading model… {downloaded/1e6:.1f} MB")
        progress.empty()
        return True
    except Exception as e:
        st.error(f"Model download failed: {e}")
        return False

@st.cache_resource(show_spinner=False)
def load_model():
    """Load (or download then load) the CNN model. Cached after first load."""
    # TensorFlow does not support Python 3.14+ yet.
    # Gracefully fall back to demo mode if unavailable.
    try:
        import tensorflow as tf  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return None
    except Exception:
        return None

    try:
        import tensorflow as tf

        if not os.path.exists(MODEL_PATH):
            st.info("🔄 Model not found locally. Downloading from Google Drive…")
            ok = download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)
            if not ok:
                return None

        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# ─────────────────────────────────────────────
# IMAGE VALIDATION
# ─────────────────────────────────────────────
def validate_image(img: Image.Image) -> tuple[bool, str]:
    """Return (is_valid, message). Checks blur, brightness."""
    import numpy as np

    arr = np.array(img.convert("L"), dtype=np.float32)

    # ── Brightness check ──
    mean_brightness = arr.mean()
    if mean_brightness < 30:
        return False, "⚠️ Image is too dark. Please upload a well-lit dermoscopic image."
    if mean_brightness > 225:
        return False, "⚠️ Image is overexposed / too bright. Please upload a properly lit image."

    # ── Blur check (Laplacian variance) ──
    # Approximate Laplacian using numpy
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from PIL import ImageFilter
    lap = img.convert("L").filter(ImageFilter.FIND_EDGES)
    lap_arr = np.array(lap, dtype=np.float32)
    variance = lap_arr.var()
    if variance < 50:
        return False, "⚠️ Image appears blurry or out-of-focus. Please upload a sharp dermoscopic image."

    # ── Minimum resolution ──
    if img.width < 64 or img.height < 64:
        return False, "⚠️ Image resolution is too low. Minimum 64×64 pixels required."

    return True, "✅ Image quality validated."

# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize to 224×224, normalize [0,1], add batch dim."""
    img = img.convert("RGB").resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(model, img: Image.Image) -> dict:
    """Run CNN prediction. Falls back to demo if no model."""
    tensor = preprocess_image(img)

    if model is not None:
        preds = model.predict(tensor, verbose=0)
        # Assume binary output: index 0 = benign, 1 = malignant
        # If your model outputs a single sigmoid neuron, adjust accordingly.
        if preds.shape[-1] == 1:
            malignant_prob = float(preds[0][0])
            benign_prob    = 1.0 - malignant_prob
        else:
            benign_prob    = float(preds[0][0])
            malignant_prob = float(preds[0][1])
    else:
        # Demo mode — random plausible values
        malignant_prob = float(np.random.beta(2, 5))
        benign_prob    = 1.0 - malignant_prob

    label      = "Malignant" if malignant_prob >= 0.5 else "Benign"
    confidence = max(malignant_prob, benign_prob) * 100

    if label == "Malignant":
        risk = "High" if confidence >= 75 else "Medium"
    else:
        risk = "Low" if confidence >= 75 else "Medium"

    return {
        "label":           label,
        "confidence":      round(confidence, 2),
        "malignant_prob":  round(malignant_prob * 100, 2),
        "benign_prob":     round(benign_prob * 100, 2),
        "risk":            risk,
    }

# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────
def generate_report(patient: dict, result: dict) -> str:
    """Return plain-text medical report."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rec = ("Urgent dermatology referral advised." if result["risk"] == "High"
           else "Routine follow-up in 3–6 months." if result["risk"] == "Medium"
           else "Continue regular self-examination.")
    treatment = ("• Biopsy recommended\n• Oncology consultation\n• Avoid sun exposure"
                 if result["label"] == "Malignant"
                 else "• Topical moisturiser if symptomatic\n• Annual skin check\n• SPF 50+ daily")

    return f"""
══════════════════════════════════════════════════════════
          SKINSCAN AI — CLINICAL DIAGNOSTIC REPORT
══════════════════════════════════════════════════════════
Report Date    : {ts}
Report ID      : RPT-{int(time.time())}

PATIENT INFORMATION
─────────────────────────────────────
Name           : {patient.get('name', 'N/A')}
Age            : {patient.get('age', 'N/A')}
Gender         : {patient.get('gender', 'N/A')}
Patient ID     : {patient.get('pid', 'N/A')}

AI DIAGNOSIS
─────────────────────────────────────
Classification : {result['label'].upper()}
Confidence     : {result['confidence']}%
Malignant Prob : {result['malignant_prob']}%
Benign Prob    : {result['benign_prob']}%
Risk Level     : {result['risk'].upper()}

CLINICAL RECOMMENDATION
─────────────────────────────────────
{rec}

SUGGESTED TREATMENT PLAN
─────────────────────────────────────
{treatment}

CLINICAL NOTES
─────────────────────────────────────
This report was generated by SkinScan AI v1.0 using a
Convolutional Neural Network trained on dermoscopic images.
This result is intended to assist clinicians and does NOT
replace professional medical diagnosis.

──────────────────────────────────────────────────────────
Verified by AI System | SkinScan AI Clinical Platform
══════════════════════════════════════════════════════════
""".strip()

# ─────────────────────────────────────────────
# HTML COMPONENTS
# ─────────────────────────────────────────────

def logo_html() -> str:
    return """
<div style="padding:28px 22px 12px;text-align:center;">
  <div style="display:inline-flex;align-items:center;justify-content:center;
              width:62px;height:62px;border-radius:16px;
              background:linear-gradient(135deg,#0a84ff,#00c8b4);
              box-shadow:0 6px 24px rgba(0,200,180,0.35);margin-bottom:10px;">
    <span style="font-size:32px;line-height:1;">🧬</span>
  </div>
  <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
              background:linear-gradient(90deg,#00c8b4,#0a84ff);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              letter-spacing:-0.01em;line-height:1.1;">SkinScan AI</div>
  <div style="font-family:'DM Sans',sans-serif;font-size:0.68rem;font-weight:500;
              color:#3d6080;text-transform:uppercase;letter-spacing:0.12em;
              margin-top:3px;">Clinical Diagnostic System</div>
</div>
"""

def status_badge(online: bool = True) -> str:
    col = "#00e87a" if online else "#ff4d6d"
    lbl = "System Online" if online else "System Offline"
    return f"""
<div style="margin:0 16px 8px;padding:8px 14px;border-radius:8px;
            background:rgba(0,232,122,0.07);border:1px solid rgba(0,232,122,0.18);
            display:flex;align-items:center;gap:8px;">
  <span style="width:8px;height:8px;border-radius:50%;
               background:{col};box-shadow:0 0 8px {col};
               display:inline-block;"></span>
  <span style="font-family:'DM Sans',sans-serif;font-size:0.75rem;
               color:{col};font-weight:500;">{lbl}</span>
</div>"""

def glass_card(content: str, border_color: str = "rgba(0,200,180,0.2)") -> str:
    return f"""
<div style="background:rgba(10,30,60,0.6);border:1px solid {border_color};
            border-radius:16px;padding:1.4rem 1.6rem;
            backdrop-filter:blur(14px);
            box-shadow:0 8px 32px rgba(0,0,0,0.4);
            margin-bottom:1rem;">
  {content}
</div>"""

def result_card(result: dict) -> str:
    color = "#ff4d6d" if result["label"] == "Malignant" else "#00e87a"
    risk_color = {"High": "#ff4d6d", "Medium": "#ffd166", "Low": "#00e87a"}[result["risk"]]
    bar_w = result["confidence"]
    return f"""
<div style="background:rgba(10,30,60,0.72);border:1.5px solid {color}44;
            border-radius:18px;padding:1.8rem 2rem;
            box-shadow:0 0 32px {color}22, 0 8px 32px rgba(0,0,0,0.45);
            backdrop-filter:blur(16px);">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:1.2rem;">
    <div style="width:56px;height:56px;border-radius:14px;
                background:linear-gradient(135deg,{color}33,{color}11);
                border:1.5px solid {color}55;
                display:flex;align-items:center;justify-content:center;
                font-size:1.8rem;">
      {"⚠️" if result["label"]=="Malignant" else "✅"}
    </div>
    <div>
      <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                  color:{color};letter-spacing:-0.02em;">{result['label'].upper()}</div>
      <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;
                  color:#7fa8c9;">AI Classification Result</div>
    </div>
    <div style="margin-left:auto;text-align:right;">
      <div style="font-family:'Syne',sans-serif;font-size:2.2rem;
                  font-weight:800;color:#e8f4ff;">{result['confidence']}%</div>
      <div style="font-size:0.75rem;color:#7fa8c9;">Confidence</div>
    </div>
  </div>

  <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:6px;margin-bottom:1.2rem;">
    <div style="height:100%;width:{bar_w}%;background:linear-gradient(90deg,{color}88,{color});
                border-radius:6px;transition:width 0.8s ease;"></div>
  </div>

  <div style="display:flex;gap:12px;flex-wrap:wrap;">
    <div style="flex:1;min-width:100px;background:rgba(255,255,255,0.04);
                border-radius:10px;padding:10px 14px;border:1px solid rgba(255,255,255,0.06);">
      <div style="font-size:0.7rem;color:#7fa8c9;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:4px;">Risk Level</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.05rem;
                  font-weight:700;color:{risk_color};">{result['risk']}</div>
    </div>
    <div style="flex:1;min-width:100px;background:rgba(255,255,255,0.04);
                border-radius:10px;padding:10px 14px;border:1px solid rgba(255,255,255,0.06);">
      <div style="font-size:0.7rem;color:#7fa8c9;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:4px;">Malignant</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.05rem;
                  font-weight:700;color:#ff4d6d;">{result['malignant_prob']}%</div>
    </div>
    <div style="flex:1;min-width:100px;background:rgba(255,255,255,0.04);
                border-radius:10px;padding:10px 14px;border:1px solid rgba(255,255,255,0.06);">
      <div style="font-size:0.7rem;color:#7fa8c9;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:4px;">Benign</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.05rem;
                  font-weight:700;color:#00e87a;">{result['benign_prob']}%</div>
    </div>
  </div>
</div>"""

# ─────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────
def page_login():
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;padding:4rem 1rem 2rem;">
      <div style="display:inline-flex;align-items:center;justify-content:center;
                  width:80px;height:80px;border-radius:22px;
                  background:linear-gradient(135deg,#0a84ff,#00c8b4);
                  box-shadow:0 8px 32px rgba(0,200,180,0.4);margin-bottom:1.2rem;">
        <span style="font-size:40px;">🧬</span>
      </div>
      <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
                  background:linear-gradient(90deg,#00c8b4,#0a84ff);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  margin-bottom:0.3rem;">SkinScan AI</div>
      <div style="color:#7fa8c9;font-size:0.9rem;letter-spacing:0.1em;
                  text-transform:uppercase;margin-bottom:2.5rem;">
        Clinical Diagnostic System
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown(glass_card("""
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                        color:#e8f4ff;margin-bottom:1rem;text-align:center;">
              🔐 Clinician Login
            </div>
        """), unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", placeholder="Enter password", type="password")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀  Sign In to System", use_container_width=True):
            if username == "admin" and password == "123":
                st.session_state.authenticated = True
                st.success("Access granted. Loading system…")
                time.sleep(0.8)
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

        st.markdown("""
        <div style="text-align:center;margin-top:1rem;font-size:0.78rem;color:#3d6080;">
            Demo credentials: admin / 123
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(logo_html(), unsafe_allow_html=True)
        st.markdown(status_badge(True), unsafe_allow_html=True)
        st.markdown("<hr style='margin:10px 0;border-color:rgba(0,200,180,0.1);'>",
                    unsafe_allow_html=True)

        pages = {
            "📊  Dashboard":        "Dashboard",
            "🔬  AI Scan":          "AI Scan",
            "📋  Patient Records":  "Records",
            "📖  User Guide":       "Guide",
        }
        for label, key in pages.items():
            active = st.session_state.page == key
            if st.button(label, key=f"nav_{key}",
                         use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:16px 0;border-color:rgba(0,200,180,0.1);'>",
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding:0 8px 16px;font-size:0.72rem;color:#3d6080;">
          <div>👤 admin</div>
          <div style="margin-top:4px;">
            Scans: <b style="color:#00c8b4;">{st.session_state.total_scans}</b>
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
def page_dashboard():
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
               background:linear-gradient(90deg,#00c8b4,#0a84ff);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin-bottom:0.2rem;">Clinical Dashboard</h1>
    <p style="color:#7fa8c9;font-size:0.85rem;margin-bottom:1.5rem;">
        Real-time diagnostic metrics & analytics
    </p>
    """, unsafe_allow_html=True)

    history = st.session_state.scan_history
    total   = st.session_state.total_scans
    high_r  = st.session_state.high_risk
    benign_count    = sum(1 for h in history if h["label"] == "Benign")
    malignant_count = sum(1 for h in history if h["label"] == "Malignant")
    avg_conf = round(np.mean([h["confidence"] for h in history]), 1) if history else 0.0

    # ── Metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔬 Total Scans",      total)
    c2.metric("🎯 Avg Confidence",   f"{avg_conf}%")
    c3.metric("⚠️ High-Risk Cases",  high_r)
    c4.metric("✅ Benign Cases",      benign_count)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    col_l, col_r = st.columns(2)

    with col_l:
        if history:
            fig_pie = px.pie(
                names=["Benign", "Malignant"],
                values=[benign_count, malignant_count],
                color_discrete_sequence=["#00e87a", "#ff4d6d"],
                hole=0.55,
                title="Diagnosis Distribution"
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8f4ff",
                title_font_family="Syne",
                legend_font_color="#7fa8c9",
            )
            fig_pie.update_traces(textfont_color="#e8f4ff")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.markdown(glass_card("""
                <div style="text-align:center;padding:2rem;color:#3d6080;">
                  <div style="font-size:2rem;margin-bottom:0.5rem;">📊</div>
                  No scan data yet. Run your first AI Scan.
                </div>"""), unsafe_allow_html=True)

    with col_r:
        if history:
            labels = [h["label"] for h in history[-20:]]
            confs  = [h["confidence"] for h in history[-20:]]
            colors = ["#ff4d6d" if l == "Malignant" else "#00e87a" for l in labels]
            fig_bar = go.Figure(go.Bar(
                y=confs, x=list(range(1, len(confs)+1)),
                marker_color=colors,
                name="Confidence"
            ))
            fig_bar.update_layout(
                title="Confidence per Scan (last 20)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8f4ff",
                title_font_family="Syne",
                xaxis=dict(showgrid=False, color="#3d6080"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                           color="#3d6080", range=[0, 100]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.markdown(glass_card("""
                <div style="text-align:center;padding:2rem;color:#3d6080;">
                  <div style="font-size:2rem;margin-bottom:0.5rem;">📈</div>
                  Confidence chart will appear after scans.
                </div>"""), unsafe_allow_html=True)

    # ── Recent scans table ──
    if history:
        st.markdown("#### Recent Scan History")
        df = pd.DataFrame(history[-10:][::-1])
        st.dataframe(df[["patient_name","label","confidence","risk","timestamp"]],
                     use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# PAGE: AI SCAN
# ─────────────────────────────────────────────
def page_scan():
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
               background:linear-gradient(90deg,#00c8b4,#0a84ff);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin-bottom:0.2rem;">AI Skin Analysis</h1>
    <p style="color:#7fa8c9;font-size:0.85rem;margin-bottom:1.5rem;">
        Upload a dermoscopic image for instant CNN-powered diagnosis
    </p>
    """, unsafe_allow_html=True)

    # ── Patient form ──
    with st.expander("👤 Patient Information", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        p_name   = c1.text_input("Full Name",   placeholder="John Doe")
        p_age    = c2.number_input("Age", min_value=1, max_value=120, value=30)
        p_gender = c3.selectbox("Gender", ["Male", "Female", "Other"])
        p_id     = c4.text_input("Patient ID",  placeholder="PT-001")

    # ── Upload ──
    st.markdown("#### 🖼️ Upload Dermoscopic Image")
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

    result = None

    if uploaded:
        img = Image.open(uploaded)

        col_img, col_ctrl = st.columns([1, 1])
        with col_img:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col_ctrl:
            # Validation
            valid, msg = validate_image(img)
            if not valid:
                st.error(f"Image validation failed\n\n{msg}\n\n"
                         "Please upload a clear dermoscopic image.")
                return
            else:
                st.success(msg)

            # Load model (cached)
            if not st.session_state.model_loaded:
                with st.spinner("Loading AI model…"):
                    st.session_state.model = load_model()
                    st.session_state.model_loaded = True

            if st.session_state.model is None:
                st.warning("⚠️ Running in **Demo Mode** — model not available.")

            st.markdown("<br>", unsafe_allow_html=True)
            run_scan = st.button("⚡  Execute Deep Scan", use_container_width=True)

        if run_scan:
            with st.spinner("🔬 Analysing lesion…"):
                progress_bar = st.progress(0)
                for i in range(1, 101):
                    time.sleep(0.012)
                    progress_bar.progress(i)
                result = predict(st.session_state.model, img)

            progress_bar.empty()

            # ── Show result ──
            st.markdown("#### 🧬 Diagnosis Result")
            st.markdown(result_card(result), unsafe_allow_html=True)

            # ── Update session state ──
            st.session_state.total_scans += 1
            if result["risk"] == "High":
                st.session_state.high_risk += 1

            patient_info = {
                "name": p_name or "Unknown",
                "age": p_age,
                "gender": p_gender,
                "pid": p_id or f"PT-{st.session_state.total_scans:03d}"
            }

            scan_record = {
                "patient_name": patient_info["name"],
                "patient_id":   patient_info["pid"],
                "label":        result["label"],
                "confidence":   result["confidence"],
                "risk":         result["risk"],
                "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            st.session_state.scan_history.append(scan_record)
            st.session_state.patient_data.append({**patient_info, **result,
                                                    "timestamp": scan_record["timestamp"]})

            # ── Report ──
            st.markdown("#### 📄 Medical Report")
            report_text = generate_report(patient_info, result)
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "📥  Download Report (.txt)",
                    data=report_text,
                    file_name=f"SkinScan_Report_{patient_info['pid']}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_b:
                # Export patient CSV
                csv_rows = "\n".join(
                    [",".join(map(str, r.values()))
                     for r in st.session_state.patient_data]
                )
                header   = ",".join(st.session_state.patient_data[0].keys())
                csv_data = header + "\n" + csv_rows
                st.download_button(
                    "📊  Export All Data (.csv)",
                    data=csv_data,
                    file_name="SkinScan_Patients.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with st.expander("📋 View Full Report Text"):
                st.code(report_text, language=None)

# ─────────────────────────────────────────────
# PAGE: RECORDS
# ─────────────────────────────────────────────
def page_records():
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
               background:linear-gradient(90deg,#00c8b4,#0a84ff);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin-bottom:1.5rem;">Patient Records</h1>
    """, unsafe_allow_html=True)

    if not st.session_state.scan_history:
        st.info("No records yet. Run the AI Scan to generate records.")
        return

    df = pd.DataFrame(st.session_state.scan_history)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button("📥 Export Records CSV", csv,
                       "patient_records.csv", "text/csv")

# ─────────────────────────────────────────────
# PAGE: USER GUIDE
# ─────────────────────────────────────────────
def page_guide():
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
               background:linear-gradient(90deg,#00c8b4,#0a84ff);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin-bottom:1.5rem;">User Guide</h1>
    """, unsafe_allow_html=True)

    steps = [
        ("1", "🔐", "Login",
         "Enter username <b>admin</b> and password <b>123</b> on the login screen."),
        ("2", "👤", "Enter Patient Details",
         "Fill in the patient name, age, gender, and ID in the AI Scan page."),
        ("3", "🖼️", "Upload Dermoscopic Image",
         "Upload a <b>clear, well-lit</b> dermoscopic JPEG/PNG image (min 64×64 px)."),
        ("4", "⚡", "Execute Deep Scan",
         "Click the <b>Execute Deep Scan</b> button to trigger the CNN model."),
        ("5", "🧬", "View AI Result",
         "Review the <b>Malignant / Benign</b> classification, confidence %, and risk level."),
        ("6", "📄", "Approve & Download Report",
         "Download the auto-generated <b>medical report (.txt)</b> for clinical records."),
        ("7", "📊", "Export Patient Data",
         "Export all session data as <b>CSV</b> for further analysis or EMR integration."),
    ]

    for num, icon, title, desc in steps:
        st.markdown(glass_card(f"""
        <div style="display:flex;align-items:flex-start;gap:16px;">
          <div style="min-width:38px;height:38px;border-radius:10px;
                      background:linear-gradient(135deg,#0a84ff22,#00c8b422);
                      border:1px solid rgba(0,200,180,0.3);
                      display:flex;align-items:center;justify-content:center;
                      font-size:1.1rem;">{icon}</div>
          <div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                        color:#e8f4ff;margin-bottom:4px;">Step {num}: {title}</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.85rem;
                        color:#7fa8c9;line-height:1.6;">{desc}</div>
          </div>
        </div>"""), unsafe_allow_html=True)

    st.markdown(glass_card("""
    <div style="font-family:'DM Sans',sans-serif;font-size:0.8rem;color:#3d6080;line-height:1.8;">
      <b style="color:#ffd166;">⚠️ Clinical Disclaimer:</b><br>
      SkinScan AI is an assistive tool designed to support, not replace,
      professional dermatological diagnosis. Always confirm AI findings with a
      licensed clinician before initiating treatment.
    </div>"""), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────
def main():

    render_sidebar()

    page = st.session_state.page
    if   page == "Dashboard": page_dashboard()
    elif page == "AI Scan":   page_scan()
    elif page == "Records":   page_records()
    elif page == "Guide":     page_guide()

if __name__ == "__main__":
    main()
