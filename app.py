"""
╔══════════════════════════════════════════════════════════════════╗
║        SkinScan AI — Real-Time Skin Cancer Detection             ║
║        University of Agriculture Faisalabad | FYP v12.0         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageFilter
import os, time, datetime, io

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be the very first Streamlit call
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SkinScan AI | Clinical Diagnostic System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS — DARK MEDICAL / NEON THEME
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800;900&family=Inter:wght@300;400;500;600&display=swap');

/* ── Variables ── */
:root{
  --bg      : #03080f;
  --bg2     : #06111f;
  --card    : rgba(5,16,34,0.90);
  --border  : rgba(0,212,255,0.16);
  --teal    : #00d4ff;
  --green   : #00ff88;
  --red     : #ff3355;
  --yellow  : #ffcc00;
  --text    : #c8e4f8;
  --muted   : #2a5070;
  --f1      : 'Orbitron', monospace;
  --f2      : 'Inter', sans-serif;
}

/* ── Base ── */
html,body,[data-testid="stAppViewContainer"],.main{
  background:var(--bg) !important;
  color:var(--text);
  font-family:var(--f2);
}
[data-testid="stAppViewContainer"]{
  background-image:
    radial-gradient(ellipse 75% 35% at 50% 0%,rgba(0,212,255,.07) 0%,transparent 70%),
    radial-gradient(ellipse 40% 28% at 92% 82%,rgba(0,255,136,.05) 0%,transparent 60%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#061120 0%,#03080f 100%) !important;
  border-right:1px solid rgba(0,212,255,.10) !important;
}

/* ── Hide chrome ── */
#MainMenu,footer,header,[data-testid="stDecoration"]{visibility:hidden;}
.stDeployButton{display:none;}

/* ── Buttons ── */
.stButton>button{
  background:linear-gradient(135deg,rgba(0,212,255,.14),rgba(0,212,255,.28)) !important;
  color:var(--teal) !important;
  border:1px solid rgba(0,212,255,.45) !important;
  border-radius:8px !important;
  font-family:var(--f1) !important;
  font-size:.68rem !important;
  font-weight:600 !important;
  letter-spacing:.08em !important;
  transition:all .2s !important;
}
.stButton>button:hover{
  background:linear-gradient(135deg,rgba(0,212,255,.28),rgba(0,212,255,.45)) !important;
  box-shadow:0 0 22px rgba(0,212,255,.30) !important;
  transform:translateY(-1px) !important;
}
/* Primary scan button */
div[data-testid="stButton"] button[kind="primary"]{
  background:linear-gradient(135deg,#00d4ff,#0077ff) !important;
  color:#000 !important; border:none !important;
  font-size:.75rem !important;
  box-shadow:0 4px 22px rgba(0,212,255,.40) !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover{
  box-shadow:0 6px 30px rgba(0,212,255,.55) !important;
}

/* ── Inputs ── */
.stTextInput input,.stNumberInput input,.stSelectbox>div>div{
  background:rgba(5,16,34,.9) !important;
  border:1px solid var(--border) !important;
  border-radius:8px !important;
  color:var(--text) !important;
}

/* ── Metrics ── */
[data-testid="metric-container"]{
  background:var(--card) !important;
  border:1px solid var(--border) !important;
  border-radius:12px !important;
  padding:1rem 1.2rem !important;
  backdrop-filter:blur(10px) !important;
}
[data-testid="stMetricValue"]{
  color:var(--teal) !important;
  font-family:var(--f1) !important;
  font-size:1.65rem !important;
}
[data-testid="stMetricLabel"] p{
  color:var(--muted) !important;
  font-size:.68rem !important;
  text-transform:uppercase;
  letter-spacing:.10em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
  background:transparent !important;
  border-bottom:1px solid rgba(0,212,255,.10) !important;
}
.stTabs [data-baseweb="tab"]{
  font-family:var(--f1) !important;
  font-size:.62rem !important;
  letter-spacing:.05em !important;
  color:var(--muted) !important;
}
.stTabs [aria-selected="true"]{
  color:var(--teal) !important;
  border-bottom:2px solid var(--teal) !important;
  background:rgba(0,212,255,.05) !important;
}

/* ── Progress bar ── */
.stProgress>div>div>div{
  background:linear-gradient(90deg,#00d4ff,#00ff88) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--teal);border-radius:4px;}

/* ── Misc ── */
hr{border-color:rgba(0,212,255,.09) !important;}
.streamlit-expanderHeader{
  background:var(--card) !important;
  border:1px solid var(--border) !important;
  border-radius:8px !important;
  color:var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "page":           "Home",
    "model":          None,
    "model_status":   "pending",   # pending | loading | loaded | failed
    "scan_history":   [],
    "total_scans":    0,
    "malignant_cnt":  0,
    "benign_cnt":     0,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# MODEL — REAL INFERENCE ONLY
# ══════════════════════════════════════════════════════════════════
GDRIVE_ID  = "18VE_D81425cZVYwAXjOn0gWti8_lZSML"
MODEL_PATH = "skin_cancer_model.h5"

@st.cache_resource(show_spinner=False)
def load_cnn_model():
    """
    Download from Google Drive (if not cached) then load via Keras.
    Returns (model, status_str).
    status_str: 'loaded' | 'download_error:<msg>' | 'load_error:<msg>' | 'tf_missing'
    """
    # 1. TensorFlow availability check
    try:
        import tensorflow as tf
    except Exception as exc:
        return None, f"tf_missing:{exc}"

    # 2. Download if needed
    if not os.path.exists(MODEL_PATH):
        downloaded = False

        # Primary: gdown
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000:
                downloaded = True
        except Exception:
            pass

        # Fallback: requests streaming
        if not downloaded:
            try:
                import requests
                sess = requests.Session()
                BASE = "https://docs.google.com/uc?export=download"
                r    = sess.get(BASE, params={"id": GDRIVE_ID}, stream=True)
                tok  = next(
                    (v for k, v in r.cookies.items() if k.startswith("download_warning")),
                    None
                )
                if tok:
                    r = sess.get(BASE,
                                 params={"id": GDRIVE_ID, "confirm": tok},
                                 stream=True)
                with open(MODEL_PATH, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=32_768):
                        if chunk:
                            fh.write(chunk)
                if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000:
                    downloaded = True
            except Exception as exc2:
                return None, f"download_error:{exc2}"

        if not downloaded:
            return None, "download_error:file not created or too small"

    # 3. Load
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, "loaded"
    except Exception as exc:
        return None, f"load_error:{exc}"


def get_model():
    """Ensure model is loaded into session state (runs once per session)."""
    if st.session_state.model_status == "pending":
        st.session_state.model_status = "loading"
        with st.spinner("🔄 Loading AI model — please wait…"):
            m, status = load_cnn_model()
        st.session_state.model        = m
        st.session_state.model_status = status
    return st.session_state.model, st.session_state.model_status

# ══════════════════════════════════════════════════════════════════
# IMAGE VALIDATION
# ══════════════════════════════════════════════════════════════════
def validate_image(img: Image.Image):
    """Return (ok: bool, message: str)."""
    if img.mode not in ("RGB", "RGBA", "L"):
        return False, "Image must be in RGB format."
    w, h = img.size
    if w < 128 or h < 128:
        return False, f"Image too small ({w}×{h}). Minimum 128×128 px required."
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    mean_b = arr.mean()
    if mean_b < 20:
        return False, "Image is too dark. Please upload a well-lit dermoscopic image."
    if mean_b > 235:
        return False, "Image is overexposed. Please upload a properly lit image."
    edges   = img.convert("L").filter(ImageFilter.FIND_EDGES)
    var_lap = np.array(edges, dtype=np.float32).var()
    if var_lap < 35:
        return False, "Image appears blurry. Please upload a sharp dermoscopic image."
    return True, "ok"

# ══════════════════════════════════════════════════════════════════
# PREDICTION PIPELINE  — REAL INFERENCE ONLY
# ══════════════════════════════════════════════════════════════════
def predict(model, img: Image.Image) -> dict:
    """
    Steps:
      1. Resize  → 224×224
      2. RGB
      3. Normalize → /255
      4. Expand dims → (1,224,224,3)
      5. model.predict()
    Returns dict with label, conf, mal_pct, ben_pct, risk.
    """
    img_resized = img.convert("RGB").resize((224, 224), Image.LANCZOS)
    arr         = np.array(img_resized, dtype=np.float32) / 255.0
    tensor      = np.expand_dims(arr, axis=0)           # (1,224,224,3)

    raw = model.predict(tensor, verbose=0)

    # Handle both sigmoid (shape [1,1]) and softmax (shape [1,2])
    if raw.shape[-1] == 1:
        mal_p = float(raw[0][0])
        ben_p = 1.0 - mal_p
    else:
        ben_p = float(raw[0][0])
        mal_p = float(raw[0][1])

    label = "Malignant" if mal_p >= 0.50 else "Benign"
    conf  = round(max(mal_p, ben_p) * 100.0, 1)

    if conf < 50:
        risk = "Low"
    elif conf < 75:
        risk = "Medium"
    else:
        risk = "Critical" if label == "Malignant" else "Low"

    return {
        "label":   label,
        "conf":    conf,
        "mal_pct": round(mal_p * 100.0, 1),
        "ben_pct": round(ben_p * 100.0, 1),
        "risk":    risk,
    }

# ══════════════════════════════════════════════════════════════════
# MEDICAL REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════
def make_report(pat: dict, res: dict) -> str:
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rid = f"RPT-{int(time.time())}"
    if res["label"] == "Malignant":
        tx  = ("• Immediate oncology referral\n"
               "• Surgical biopsy strongly recommended\n"
               "• Avoid direct UV / sun exposure\n"
               "• Emergency follow-up within 7 days\n"
               "• Do NOT attempt self-treatment")
        rec = "URGENT: Consult a dermatologist / oncologist immediately."
    else:
        tx  = ("• Schedule annual skin screening\n"
               "• Apply SPF 50+ sunscreen daily\n"
               "• Monthly self-examination\n"
               "• Consult dermatologist if lesion changes\n"
               "• Maintain a healthy lifestyle")
        rec = "Routine monitoring advised. Recheck in 6–12 months."

    return (
        "╔══════════════════════════════════════════════════════════╗\n"
        "         SKINSCAN AI — CLINICAL DIAGNOSTIC REPORT\n"
        "╚══════════════════════════════════════════════════════════╝\n"
        f"Report ID      : {rid}\n"
        f"Generated      : {ts}\n"
        f"System Version : SkinScan AI v12.0\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        " PATIENT INFORMATION\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Name           : {pat.get('name','N/A')}\n"
        f"Age            : {pat.get('age','N/A')}\n"
        f"Gender         : {pat.get('gender','N/A')}\n"
        f"Patient ID     : {pat.get('pid','N/A')}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        " AI DIAGNOSIS\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Classification : {res['label'].upper()}\n"
        f"Confidence     : {res['conf']}%\n"
        f"Malignant Prob : {res['mal_pct']}%\n"
        f"Benign Prob    : {res['ben_pct']}%\n"
        f"Risk Level     : {res['risk'].upper()}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        " CLINICAL RECOMMENDATION\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{rec}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        " TREATMENT PLAN\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{tx}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        " DISCLAIMER\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "This report is generated by an AI system and is intended to\n"
        "assist clinicians only. It does NOT replace professional\n"
        "medical diagnosis. Always confirm with a licensed physician.\n\n"
        "──────────────────────────────────────────────────────────\n"
        "University of Agriculture Faisalabad | Rehan Shafique\n"
        "rehanshafiq6540@gmail.com | SkinScan AI v12.0\n"
        "══════════════════════════════════════════════════════════"
    )

# ══════════════════════════════════════════════════════════════════
# REUSABLE HTML HELPERS
# ══════════════════════════════════════════════════════════════════
def _card(html: str, border: str = "rgba(0,212,255,0.16)") -> str:
    return (
        f'<div style="background:rgba(5,16,34,0.88);border:1px solid {border};'
        f'border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:.8rem;'
        f'backdrop-filter:blur(10px);">{html}</div>'
    )

def _badge(color: str, label: str) -> str:
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'background:{color}18;border:1px solid {color}55;'
        f'border-radius:6px;padding:3px 10px;font-size:.68rem;'
        f'font-family:Orbitron,monospace;color:{color};">'
        f'<span style="width:6px;height:6px;border-radius:50%;'
        f'background:{color};box-shadow:0 0 8px {color};'
        f'display:inline-block;"></span>{label}</span>'
    )

def _section_title(txt: str) -> str:
    return (
        f'<div style="text-align:center;margin-bottom:1.4rem;">'
        f'<span style="font-family:Orbitron,monospace;font-size:.8rem;'
        f'color:#00d4ff;letter-spacing:.15em;text-transform:uppercase;">'
        f'{txt}</span></div>'
    )

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="padding:22px 14px 14px;text-align:center;">
          <div style="display:inline-flex;align-items:center;justify-content:center;
                      width:60px;height:60px;border-radius:15px;
                      background:linear-gradient(135deg,rgba(0,212,255,.18),rgba(0,255,136,.12));
                      border:1.5px solid rgba(0,212,255,.40);
                      box-shadow:0 0 28px rgba(0,212,255,.18);
                      margin-bottom:10px;">
            <span style="font-size:28px;">🧬</span>
          </div>
          <div style="font-family:Orbitron,monospace;font-size:.95rem;font-weight:800;
                      color:#00d4ff;letter-spacing:.04em;">SkinScan AI</div>
          <div style="font-size:.58rem;color:#2a5070;letter-spacing:.15em;
                      text-transform:uppercase;margin-top:3px;">
            Clinical Diagnostic v12.0
          </div>
        </div>""", unsafe_allow_html=True)

        # Model status
        status = st.session_state.model_status
        if status == "loaded":
            st.markdown(_badge("#00ff88", "CNN Model Online"), unsafe_allow_html=True)
        elif status in ("pending", "loading"):
            st.markdown(_badge("#ffcc00", "Model Loading…"), unsafe_allow_html=True)
        else:
            short = status[:35] if len(status) > 35 else status
            st.markdown(_badge("#ff3355", f"Model Error"), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        nav = {
            "🏠  Home":             "Home",
            "🔬  AI Scan":          "Scan",
            "📊  Dashboard":        "Dashboard",
            "🗂️  Patient Registry":  "Registry",
            "📘  User Guide":       "Guide",
        }
        for label, key in nav.items():
            active = st.session_state.page == key
            if st.button(label, key=f"nav_{key}",
                         use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="padding:0 6px 8px;font-size:.68rem;'
            f'color:#2a5070;line-height:2;">'
            f'🔬 Total Scans: <b style="color:#00d4ff;">'
            f'{st.session_state.total_scans}</b><br>'
            f'⚠️ Malignant: <b style="color:#ff3355;">'
            f'{st.session_state.malignant_cnt}</b><br>'
            f'✅ Benign: <b style="color:#00ff88;">'
            f'{st.session_state.benign_cnt}</b></div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════
# PAGE — HOME
# ══════════════════════════════════════════════════════════════════
def page_home():
    # Hero
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 2.2rem;">
      <div style="display:inline-flex;align-items:center;justify-content:center;
                  width:96px;height:96px;border-radius:24px;
                  background:linear-gradient(135deg,rgba(0,212,255,.18),rgba(0,255,136,.12));
                  border:1.5px solid rgba(0,212,255,.45);
                  box-shadow:0 0 50px rgba(0,212,255,.20);
                  margin-bottom:1.3rem;">
        <span style="font-size:46px;">🧬</span>
      </div>
      <h1 style="font-family:Orbitron,monospace;font-size:2.7rem;font-weight:900;
                 background:linear-gradient(90deg,#00d4ff,#00ff88);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 letter-spacing:.04em;margin-bottom:.5rem;">SkinScan AI</h1>
      <div style="font-size:.85rem;color:#2a5070;letter-spacing:.18em;
                  text-transform:uppercase;margin-bottom:.9rem;">
        Clinical Diagnostic System v12.0
      </div>
      <div style="font-size:1.05rem;color:#7aaccc;max-width:580px;
                  margin:0 auto 2.2rem;line-height:1.8;">
        Hospital-grade AI skin cancer detection powered by Deep Learning.<br>
        Instant <b style="color:#00ff88;">Benign</b> &nbsp;vs&nbsp;
        <b style="color:#ff3355;">Malignant</b> classification with clinical reports.
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1.2, 2])
    with c2:
        if st.button("⚡  START DIAGNOSIS", use_container_width=True, type="primary"):
            st.session_state.page = "Scan"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Step-by-step guide
    st.markdown(_section_title("📘  How It Works — Step by Step"), unsafe_allow_html=True)
    steps = [
        ("01", "🖼️", "Upload Image",    "Drag & drop a dermoscopic skin lesion image (JPG / PNG, min 128×128)"),
        ("02", "⚡",  "Run AI Scan",    "Click Execute Deep Scan to trigger the CNN model"),
        ("03", "🧠",  "AI Processing",  "Real-time inference through convolutional layers"),
        ("04", "📊",  "View Result",    "Diagnosis, confidence %, risk level & probability charts"),
        ("05", "📄",  "Export Report",  "Download medical report or export full patient CSV"),
    ]
    cols = st.columns(5)
    for i, (num, icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(
                f'<div style="background:rgba(5,16,34,.80);'
                f'border:1px solid rgba(0,212,255,.13);'
                f'border-radius:14px;padding:1.1rem .7rem;text-align:center;height:168px;">'
                f'<div style="font-family:Orbitron,monospace;font-size:1.5rem;'
                f'font-weight:900;color:rgba(0,212,255,.18);margin-bottom:5px;">{num}</div>'
                f'<div style="font-size:1.5rem;margin-bottom:8px;">{icon}</div>'
                f'<div style="font-family:Orbitron,monospace;font-size:.58rem;'
                f'color:#00d4ff;font-weight:700;letter-spacing:.04em;margin-bottom:5px;">{title}</div>'
                f'<div style="font-size:.70rem;color:#2a5070;line-height:1.5;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Features
    st.markdown(_section_title("🔬  Key Features"), unsafe_allow_html=True)
    feats = [
        ("🧠", "CNN Detection",        "Deep learning model trained on the Kaggle Melanoma dataset"),
        ("⚡",  "Real-time Inference",  "Instant prediction — no simulation, no random output"),
        ("📄",  "Medical Reports",      "Auto-generated clinical reports ready for download"),
        ("📊",  "Analytics Dashboard",  "Scan history, pie charts, and confidence trend graphs"),
        ("🗂️",  "Patient Registry",     "Structured patient data management & CSV export"),
        ("✅",  "Image Validation",     "Rejects blurry, dark, or low-quality images automatically"),
    ]
    fc1, fc2, fc3 = st.columns(3)
    for i, (icon, title, desc) in enumerate(feats):
        col = [fc1, fc2, fc3][i % 3]
        with col:
            st.markdown(
                f'<div style="background:rgba(5,16,34,.75);'
                f'border:1px solid rgba(0,212,255,.11);'
                f'border-radius:12px;padding:.9rem 1rem;margin-bottom:.7rem;'
                f'display:flex;gap:11px;align-items:flex-start;">'
                f'<span style="font-size:1.4rem;flex-shrink:0;">{icon}</span>'
                f'<div>'
                f'<div style="font-family:Orbitron,monospace;font-size:.62rem;'
                f'color:#00d4ff;font-weight:700;margin-bottom:3px;">{title}</div>'
                f'<div style="font-size:.73rem;color:#2a5070;line-height:1.5;">{desc}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset stats
    st.markdown(_section_title("📂  Dataset Information — Melanoma (Kaggle)"), unsafe_allow_html=True)
    ds_cols = st.columns(4)
    for col, val, lbl in [
        (ds_cols[0], "10,015",   "Total Images"),
        (ds_cols[1], "6,705",    "Benign Cases"),
        (ds_cols[2], "3,310",    "Malignant Cases"),
        (ds_cols[3], "Kaggle",   "Dataset Source"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:rgba(5,16,34,.80);'
                f'border:1px solid rgba(0,212,255,.14);'
                f'border-radius:12px;padding:.95rem;text-align:center;">'
                f'<div style="font-family:Orbitron,monospace;font-size:1.35rem;'
                f'font-weight:800;color:#00d4ff;">{val}</div>'
                f'<div style="font-size:.70rem;color:#2a5070;margin-top:4px;">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════
# PAGE — AI SCAN
# ══════════════════════════════════════════════════════════════════
def page_scan():
    st.markdown(
        '<h1 style="font-family:Orbitron,monospace;font-size:1.55rem;'
        'font-weight:800;color:#00d4ff;margin-bottom:.25rem;">🔬 AI Skin Analysis</h1>'
        '<p style="color:#2a5070;font-size:.83rem;margin-bottom:1.4rem;">'
        'Upload a dermoscopic image for real-time CNN-powered skin cancer detection</p>',
        unsafe_allow_html=True,
    )

    # Load model
    model, status = get_model()

    # Status indicator
    if status == "loaded":
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:8px;'
            'background:rgba(0,255,136,.07);border:1px solid rgba(0,255,136,.25);'
            'border-radius:8px;padding:5px 13px;margin-bottom:1rem;">'
            '<span style="width:7px;height:7px;border-radius:50%;background:#00ff88;'
            'box-shadow:0 0 8px #00ff88;display:inline-block;"></span>'
            '<span style="font-size:.70rem;color:#00ff88;'
            'font-family:Orbitron,monospace;">CNN Model Online — Real Inference Active</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        detail = status.replace("tf_missing:", "").replace("download_error:", "").replace("load_error:", "")
        st.markdown(
            f'<div style="background:rgba(255,51,85,.07);border:1px solid rgba(255,51,85,.30);'
            f'border-radius:10px;padding:.9rem 1.1rem;margin-bottom:1rem;">'
            f'<b style="color:#ff3355;">❌ Model Unavailable</b><br>'
            f'<span style="font-size:.78rem;color:#2a5070;">{detail}</span><br>'
            f'<span style="font-size:.74rem;color:#2a5070;">'
            f'Check runtime.txt sets Python 3.11 and requirements.txt includes tensorflow==2.15.0'
            f'</span></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # Patient info
    with st.expander("👤 Patient Information", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        p_name   = c1.text_input("Full Name",  placeholder="e.g. Ahmed Ali")
        p_age    = c2.number_input("Age", 1, 120, 30)
        p_gender = c3.selectbox("Gender", ["Male", "Female", "Other"])
        p_id     = c4.text_input("Patient ID", placeholder="PT-001")

    # Upload
    st.markdown("#### 🖼️ Upload Dermoscopic Image")
    uploaded = st.file_uploader(
        "JPG / JPEG / PNG / BMP — min 128×128 px",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown(
            '<div style="border:2px dashed rgba(0,212,255,.18);border-radius:14px;'
            'padding:3rem;text-align:center;color:#2a5070;">'
            '<div style="font-size:3rem;margin-bottom:.5rem;">🖼️</div>'
            '<div style="font-size:.85rem;">Drag & drop a dermoscopic image here</div>'
            '<div style="font-size:.70rem;margin-top:5px;">JPG · PNG · BMP · min 128×128 px</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    img = Image.open(uploaded)
    col_img, col_ctrl = st.columns([1, 1])

    with col_img:
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.markdown(
            f'<div style="font-size:.70rem;color:#2a5070;margin-top:3px;">'
            f'Resolution: {img.width}×{img.height}px &nbsp;|&nbsp; Mode: {img.mode}</div>',
            unsafe_allow_html=True,
        )

    with col_ctrl:
        ok, msg = validate_image(img)
        if not ok:
            st.markdown(
                f'<div style="background:rgba(255,51,85,.07);'
                f'border:1px solid rgba(255,51,85,.28);'
                f'border-radius:10px;padding:1rem 1.2rem;color:#ff3355;">'
                f'<b>❌ Invalid Image</b><br>'
                f'<span style="font-size:.80rem;">{msg}</span><br><br>'
                f'<span style="font-size:.75rem;color:#2a5070;">'
                f'Please upload a proper dermoscopic skin image.</span></div>',
                unsafe_allow_html=True,
            )
            return

        st.markdown(
            '<div style="background:rgba(0,255,136,.06);'
            'border:1px solid rgba(0,255,136,.20);'
            'border-radius:10px;padding:.65rem 1rem;'
            'color:#00ff88;font-size:.78rem;margin-bottom:1rem;">'
            '✅ Image quality validated — ready for scan</div>',
            unsafe_allow_html=True,
        )

        run = st.button("⚡  EXECUTE DEEP SCAN", use_container_width=True, type="primary")

    if not run:
        return

    # ── Inference ──
    prog = st.progress(0, text="Initialising neural network…")
    for pct, txt in [(20, "Preprocessing image…"),
                     (45, "Extracting feature maps…"),
                     (70, "Running inference…"),
                     (90, "Computing probabilities…"),
                     (100,"Finalising…")]:
        time.sleep(0.22)
        prog.progress(pct, text=txt)
    prog.empty()

    try:
        res = predict(model, img)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    # Low-confidence warning
    if res["conf"] < 60:
        st.markdown(
            f'<div style="background:rgba(255,204,0,.06);'
            f'border:1px solid rgba(255,204,0,.28);'
            f'border-radius:10px;padding:.75rem 1.1rem;'
            f'color:#ffcc00;font-size:.78rem;margin-bottom:1rem;">'
            f'⚠️ <b>Low Reliability</b> — Confidence is {res["conf"]}% (&lt;60%). '
            f'Result may be inconclusive. Upload a higher-quality image.</div>',
            unsafe_allow_html=True,
        )

    # ── Result card ──
    lc  = "#ff3355" if res["label"] == "Malignant" else "#00ff88"
    rc  = {"Critical": "#ff3355", "Medium": "#ffcc00", "Low": "#00ff88"}[res["risk"]]
    ico = "⚠️" if res["label"] == "Malignant" else "✅"

    st.markdown(
        f'<div style="background:rgba(5,16,34,.92);border:1.5px solid {lc}44;'
        f'border-radius:16px;padding:1.6rem 1.9rem;'
        f'box-shadow:0 0 32px {lc}12;margin-top:.4rem;">'

        f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:1.2rem;">'
        f'<div style="width:54px;height:54px;border-radius:12px;'
        f'background:{lc}16;border:1px solid {lc}44;'
        f'display:flex;align-items:center;justify-content:center;font-size:1.6rem;">{ico}</div>'
        f'<div>'
        f'<div style="font-family:Orbitron,monospace;font-size:1.4rem;'
        f'font-weight:800;color:{lc};">{res["label"].upper()}</div>'
        f'<div style="font-size:.73rem;color:#2a5070;">AI Classification</div>'
        f'</div>'
        f'<div style="margin-left:auto;text-align:right;">'
        f'<div style="font-family:Orbitron,monospace;font-size:2rem;'
        f'font-weight:900;color:#c8e4f8;">{res["conf"]}%</div>'
        f'<div style="font-size:.68rem;color:#2a5070;">Confidence</div>'
        f'</div></div>'

        f'<div style="height:5px;background:rgba(255,255,255,.05);'
        f'border-radius:5px;margin-bottom:1.2rem;">'
        f'<div style="height:100%;width:{res["conf"]}%;'
        f'background:linear-gradient(90deg,{lc}66,{lc});border-radius:5px;"></div></div>'

        f'<div style="display:flex;gap:10px;">'
        + "".join(
            f'<div style="flex:1;background:rgba(255,255,255,.03);'
            f'border-radius:8px;padding:8px 11px;'
            f'border:1px solid rgba(255,255,255,.05);">'
            f'<div style="font-size:.63rem;color:#2a5070;text-transform:uppercase;'
            f'letter-spacing:.08em;">{k}</div>'
            f'<div style="font-family:Orbitron,monospace;font-size:.88rem;'
            f'font-weight:700;color:{vc};">{vv}</div></div>'
            for k, vv, vc in [
                ("Risk Level",  res["risk"],         rc),
                ("Malignant",   f'{res["mal_pct"]}%', "#ff3355"),
                ("Benign",      f'{res["ben_pct"]}%', "#00ff88"),
            ]
        )
        + '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analysis charts ──
    ch1, ch2 = st.columns(2)
    with ch1:
        fig = go.Figure(go.Bar(
            x=["Benign", "Malignant"],
            y=[res["ben_pct"], res["mal_pct"]],
            marker_color=["#00ff88", "#ff3355"],
            text=[f'{res["ben_pct"]}%', f'{res["mal_pct"]}%'],
            textposition="outside",
        ))
        fig.update_layout(
            title="Probability Distribution",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8e4f8", font_family="Orbitron",
            yaxis=dict(range=[0, 115], showgrid=False, color="#2a5070"),
            xaxis=dict(color="#2a5070"), height=270,
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=res["conf"],
            title={"text": "Confidence Gauge",
                   "font": {"color": "#c8e4f8", "family": "Orbitron", "size": 13}},
            gauge={
                "axis":    {"range": [0, 100], "tickcolor": "#2a5070"},
                "bar":     {"color": lc},
                "bgcolor": "rgba(0,0,0,0)",
                "steps":   [
                    {"range": [0,  50],  "color": "rgba(255,51,85,.09)"},
                    {"range": [50, 75],  "color": "rgba(255,204,0,.09)"},
                    {"range": [75, 100], "color": "rgba(0,255,136,.09)"},
                ],
            },
            number={"suffix": "%", "font": {"color": "#c8e4f8", "family": "Orbitron"}},
        ))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#c8e4f8", height=270)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Medical recommendations ──
    if res["label"] == "Malignant":
        st.markdown(
            '<div style="background:rgba(255,51,85,.06);'
            'border:1px solid rgba(255,51,85,.28);'
            'border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:.9rem;">'
            '<div style="font-family:Orbitron,monospace;font-size:.72rem;'
            'color:#ff3355;font-weight:700;margin-bottom:.6rem;">'
            '🚨 MALIGNANT DETECTED — URGENT ACTION REQUIRED</div>'
            '<div style="font-size:.80rem;color:#8ab0cc;line-height:2.1;">'
            '▸ Immediate oncology referral<br>'
            '▸ Surgical biopsy strongly recommended<br>'
            '▸ Avoid direct UV / sun exposure<br>'
            '▸ Emergency follow-up within 7 days<br>'
            '▸ Do NOT attempt self-treatment</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:rgba(0,255,136,.05);'
            'border:1px solid rgba(0,255,136,.20);'
            'border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:.9rem;">'
            '<div style="font-family:Orbitron,monospace;font-size:.72rem;'
            'color:#00ff88;font-weight:700;margin-bottom:.6rem;">'
            '✅ BENIGN — ROUTINE MONITORING ADVISED</div>'
            '<div style="font-size:.80rem;color:#8ab0cc;line-height:2.1;">'
            '▸ Schedule annual skin screening<br>'
            '▸ Apply SPF 50+ sunscreen daily<br>'
            '▸ Perform monthly self-examination<br>'
            '▸ Consult dermatologist if lesion changes<br>'
            '▸ Maintain a healthy lifestyle</div></div>',
            unsafe_allow_html=True,
        )

    # ── Save to session ──
    patient = {
        "name":   p_name   or "Unknown",
        "age":    p_age,
        "gender": p_gender,
        "pid":    p_id     or f"PT-{st.session_state.total_scans + 1:03d}",
    }
    st.session_state.total_scans += 1
    if res["label"] == "Malignant":
        st.session_state.malignant_cnt += 1
    else:
        st.session_state.benign_cnt += 1
    st.session_state.scan_history.append({
        "scan_id":    f"SC-{st.session_state.total_scans:03d}",
        "patient_id": patient["pid"],
        "name":       patient["name"],
        "diagnosis":  res["label"],
        "confidence": f'{res["conf"]}%',
        "risk":       res["risk"],
        "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

    # ── Action buttons ──
    st.markdown("#### 📋 Actions")
    b1, b2, b3, b4 = st.columns(4)
    report_txt = make_report(patient, res)

    with b1:
        st.download_button("📄 Medical Report", report_txt,
                           f'Report_{patient["pid"]}.txt', "text/plain",
                           use_container_width=True)
    with b2:
        st.markdown(
            f'<div style="background:rgba(0,255,136,.06);'
            f'border:1px solid rgba(0,255,136,.20);border-radius:8px;'
            f'padding:.55rem .8rem;font-size:.72rem;color:#00ff88;text-align:center;">'
            f'💾 Record saved — {patient["pid"]}</div>',
            unsafe_allow_html=True,
        )
    with b3:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()
    with b4:
        if st.session_state.scan_history:
            csv_data = pd.DataFrame(st.session_state.scan_history).to_csv(index=False)
            st.download_button("📥 Export CSV", csv_data,
                               "SkinScan_Records.csv", "text/csv",
                               use_container_width=True)

    # Auto disclaimer
    st.markdown(
        '<div style="margin-top:1rem;padding:.75rem 1rem;'
        'background:rgba(255,204,0,.04);border:1px solid rgba(255,204,0,.14);'
        'border-radius:8px;font-size:.70rem;color:#2a5070;line-height:1.7;">'
        '<b style="color:#ffcc00;">⚠️ Medical Disclaimer:</b> '
        'This AI result is intended to assist clinicians only and does NOT '
        'replace professional medical diagnosis. Always confirm findings with '
        'a licensed dermatologist or oncologist.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# PAGE — DASHBOARD
# ══════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown(
        '<h1 style="font-family:Orbitron,monospace;font-size:1.55rem;'
        'font-weight:800;color:#00d4ff;margin-bottom:.25rem;">📊 Analytics Dashboard</h1>'
        '<p style="color:#2a5070;font-size:.83rem;margin-bottom:1.4rem;">'
        'Real-time diagnostic metrics and session analytics</p>',
        unsafe_allow_html=True,
    )

    history = st.session_state.scan_history
    total   = st.session_state.total_scans
    mal     = st.session_state.malignant_cnt
    ben     = st.session_state.benign_cnt
    confs   = [float(h["confidence"].replace("%", "")) for h in history] if history else []
    avg_c   = round(np.mean(confs), 1) if confs else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔬 Total Scans",       total)
    m2.metric("🎯 Avg Confidence",    f"{avg_c}%")
    m3.metric("⚠️ Malignant Cases",   mal)
    m4.metric("✅ Benign Cases",       ben)

    st.markdown("<br>", unsafe_allow_html=True)

    if not history:
        st.markdown(
            '<div style="background:rgba(5,16,34,.75);'
            'border:1px solid rgba(0,212,255,.12);'
            'border-radius:14px;padding:3rem;text-align:center;">'
            '<div style="font-size:2.5rem;margin-bottom:.6rem;">📊</div>'
            '<div style="color:#2a5070;font-size:.85rem;">'
            'No scan data yet. Go to AI Scan to get started.</div></div>',
            unsafe_allow_html=True,
        )
        return

    cl, cr = st.columns(2)
    with cl:
        fig_pie = px.pie(
            names=["Benign", "Malignant"], values=[ben, mal],
            color_discrete_sequence=["#00ff88", "#ff3355"],
            hole=0.55, title="Diagnosis Distribution",
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8e4f8", title_font_family="Orbitron",
            legend_font_color="#2a5070",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with cr:
        bar_colors = ["#ff3355" if h["diagnosis"] == "Malignant" else "#00ff88"
                      for h in history[-20:]]
        fig_bar = go.Figure(go.Bar(
            y=confs[-20:], x=list(range(1, len(confs[-20:]) + 1)),
            marker_color=bar_colors,
        ))
        fig_bar.update_layout(
            title="Confidence per Scan (last 20)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8e4f8", title_font_family="Orbitron",
            xaxis=dict(showgrid=False, color="#2a5070"),
            yaxis=dict(range=[0, 112], gridcolor="rgba(255,255,255,.04)", color="#2a5070"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    if len(confs) >= 2:
        fig_line = go.Figure(go.Scatter(
            y=confs, mode="lines+markers",
            line=dict(color="#00d4ff", width=2),
            marker=dict(color="#00d4ff", size=5),
            fill="tozeroy", fillcolor="rgba(0,212,255,.05)",
        ))
        fig_line.update_layout(
            title="Confidence Trend",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8e4f8", title_font_family="Orbitron",
            xaxis=dict(showgrid=False, color="#2a5070"),
            yaxis=dict(range=[0, 112], gridcolor="rgba(255,255,255,.04)", color="#2a5070"),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Dataset reference
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_section_title("📂  Dataset Reference (Kaggle Melanoma)"), unsafe_allow_html=True)
    ds1, ds2, ds3, ds4 = st.columns(4)
    for col, val, lbl in [
        (ds1, "10,015", "Total Images"),
        (ds2, "6,705",  "Benign"),
        (ds3, "3,310",  "Malignant"),
        (ds4, "66.9%",  "Benign Ratio"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:rgba(5,16,34,.80);'
                f'border:1px solid rgba(0,212,255,.13);'
                f'border-radius:12px;padding:.9rem;text-align:center;">'
                f'<div style="font-family:Orbitron,monospace;font-size:1.25rem;'
                f'font-weight:800;color:#00d4ff;">{val}</div>'
                f'<div style="font-size:.68rem;color:#2a5070;margin-top:4px;">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════
# PAGE — PATIENT REGISTRY
# ══════════════════════════════════════════════════════════════════
def page_registry():
    st.markdown(
        '<h1 style="font-family:Orbitron,monospace;font-size:1.55rem;'
        'font-weight:800;color:#00d4ff;margin-bottom:.25rem;">🗂️ Patient Registry</h1>'
        '<p style="color:#2a5070;font-size:.83rem;margin-bottom:1.4rem;">'
        'Scan records and patient data management</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.scan_history:
        st.markdown(
            '<div style="background:rgba(5,16,34,.75);'
            'border:1px solid rgba(0,212,255,.12);'
            'border-radius:14px;padding:3rem;text-align:center;">'
            '<div style="font-size:2.5rem;margin-bottom:.6rem;">🗂️</div>'
            '<div style="color:#2a5070;">No patient records yet.</div></div>',
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(st.session_state.scan_history)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("📥 Export Registry (CSV)",
                       df.to_csv(index=False),
                       "SkinScan_Registry.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════
# PAGE — USER GUIDE
# ══════════════════════════════════════════════════════════════════
def page_guide():
    st.markdown(
        '<h1 style="font-family:Orbitron,monospace;font-size:1.55rem;'
        'font-weight:800;color:#00d4ff;margin-bottom:.25rem;">📘 User Guide</h1>'
        '<p style="color:#2a5070;font-size:.83rem;margin-bottom:1.4rem;">'
        'Complete step-by-step system walkthrough</p>',
        unsafe_allow_html=True,
    )

    steps = [
        ("01", "👤", "Enter Patient Details",
         "Fill in the patient name, age, gender, and ID in the AI Scan page before uploading an image."),
        ("02", "🖼️", "Upload Dermoscopic Image",
         "Upload a clear, well-lit JPEG or PNG dermoscopic skin image. Minimum size: 128x128 pixels."),
        ("03", "⚡", "Execute Deep Scan",
         "Click EXECUTE DEEP SCAN. The CNN model runs real inference — no simulation, no random output."),
        ("04", "🧬", "View AI Result",
         "Review the Malignant or Benign classification, confidence %, risk level, and probability charts."),
        ("05", "📄", "Download Medical Report",
         "Click Generate Medical Report to download a complete clinical report (.txt) for your records."),
        ("06", "📊", "View Dashboard",
         "Navigate to Dashboard to view analytics, pie charts, and confidence trend history."),
        ("07", "📥", "Export Patient Data",
         "Export all session scan records as a structured CSV from the Patient Registry page."),
    ]

    for num, icon, title, desc in steps:
        st.markdown(
            f'<div style="background:rgba(5,16,34,.78);'
            f'border:1px solid rgba(0,212,255,.14);'
            f'border-radius:14px;padding:1rem 1.3rem;margin-bottom:.75rem;'
            f'display:flex;align-items:flex-start;gap:13px;">'
            f'<div style="min-width:42px;height:42px;border-radius:10px;'
            f'background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.22);'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:1.25rem;flex-shrink:0;">{icon}</div>'
            f'<div>'
            f'<div style="font-family:Orbitron,monospace;font-size:.68rem;font-weight:700;'
            f'color:#00d4ff;margin-bottom:3px;">STEP {num} — {title.upper()}</div>'
            f'<div style="font-size:.80rem;color:#4a7a9a;line-height:1.65;">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="background:rgba(255,204,0,.05);'
        'border:1px solid rgba(255,204,0,.18);'
        'border-radius:12px;padding:1rem 1.3rem;margin-top:.4rem;">'
        '<div style="font-family:Orbitron,monospace;font-size:.68rem;'
        'color:#ffcc00;margin-bottom:5px;">⚠️ CLINICAL DISCLAIMER</div>'
        '<div style="font-size:.78rem;color:#2a5070;line-height:1.7;">'
        'SkinScan AI is an assistive diagnostic tool designed to support, not replace, '
        'professional dermatological diagnosis. All AI results must be confirmed by a '
        'licensed clinician before initiating any treatment.</div></div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
def render_footer():
    st.markdown("<br><br><hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;padding:.8rem 0;'
        'font-family:Orbitron,monospace;line-height:2.2;">'
        '<div style="font-size:.58rem;color:#1a3a5a;">University of Agriculture Faisalabad</div>'
        '<div style="font-size:.58rem;color:#1a3a5a;">'
        'Rehan Shafique &nbsp;|&nbsp; rehanshafiq6540@gmail.com</div>'
        '<div style="font-size:.58rem;color:#1a3a5a;">'
        'SkinScan AI — Medical Diagnostic System v12.0</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════
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
