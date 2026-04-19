"""
╔══════════════════════════════════════════════════════════════════╗
║        SkinScan AI — Real-Time Skin Cancer Detection             ║
║        ONNX Runtime — Works on Python 3.14 (Streamlit Cloud)     ║
║        University of Agriculture Faisalabad | FYP v12.0          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageFilter
import os, time, datetime

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SkinScan AI | Clinical Diagnostic System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS — DARK MEDICAL THEME
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800;900&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:     #03080f;
    --bg2:    #06111f;
    --card:   rgba(5,16,34,0.90);
    --border: rgba(0,212,255,0.16);
    --teal:   #00d4ff;
    --green:  #00ff88;
    --red:    #ff3355;
    --yellow: #ffcc00;
    --text:   #c8e4f8;
    --muted:  #2a5070;
    --f1:     'Orbitron', monospace;
    --f2:     'Inter', sans-serif;
}

html,body,[data-testid="stAppViewContainer"],.main {
    background: var(--bg) !important;
    color: var(--text);
    font-family: var(--f2);
}
[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse 75% 35% at 50% 0%, rgba(0,212,255,.07) 0%, transparent 70%),
        radial-gradient(ellipse 40% 28% at 92% 82%, rgba(0,255,136,.05) 0%, transparent 60%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#061120 0%,#03080f 100%) !important;
    border-right: 1px solid rgba(0,212,255,.10) !important;
}
#MainMenu,footer,header,[data-testid="stDecoration"] { visibility: hidden; }
.stDeployButton { display: none; }

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg,rgba(0,212,255,.14),rgba(0,212,255,.28)) !important;
    color: var(--teal) !important;
    border: 1px solid rgba(0,212,255,.45) !important;
    border-radius: 8px !important;
    font-family: var(--f1) !important;
    font-size: .68rem !important;
    font-weight: 600 !important;
    letter-spacing: .08em !important;
    transition: all .2s !important;
}
.stButton>button:hover {
    background: linear-gradient(135deg,rgba(0,212,255,.28),rgba(0,212,255,.45)) !important;
    box-shadow: 0 0 22px rgba(0,212,255,.30) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg,#00d4ff,#0077ff) !important;
    color: #000 !important;
    border: none !important;
    font-size: .75rem !important;
    box-shadow: 0 4px 22px rgba(0,212,255,.40) !important;
}

/* Inputs */
.stTextInput input,.stNumberInput input,.stSelectbox>div>div {
    background: rgba(5,16,34,.9) !important;
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
    backdrop-filter: blur(10px) !important;
}
[data-testid="stMetricValue"] {
    color: var(--teal) !important;
    font-family: var(--f1) !important;
    font-size: 1.65rem !important;
}
[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: .68rem !important;
    text-transform: uppercase;
    letter-spacing: .10em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0,212,255,.10) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--f1) !important;
    font-size: .62rem !important;
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom: 2px solid var(--teal) !important;
    background: rgba(0,212,255,.05) !important;
}

/* Progress */
.stProgress>div>div>div {
    background: linear-gradient(90deg,#00d4ff,#00ff88) !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--teal); border-radius: 4px; }
hr { border-color: rgba(0,212,255,.09) !important; }
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ⚠️  IMPORTANT — REPLACE THIS WITH YOUR NEW ONNX FILE ID
# After running convert_model.py in Colab, paste the new File ID here
# ══════════════════════════════════════════════════════════════════
ONNX_GDRIVE_ID = "PASTE_YOUR_ONNX_FILE_ID_HERE"   # ← Change this!
ONNX_PATH      = "skin_cancer_model.onnx"

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
_DEF = {
    "page":          "Home",
    "sess":          None,          # ONNX InferenceSession
    "model_status":  "pending",     # pending | loaded | failed
    "model_msg":     "",
    "scan_history":  [],
    "total_scans":   0,
    "malignant_cnt": 0,
    "benign_cnt":    0,
}
for k, v in _DEF.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# MODEL — ONNX DOWNLOAD + LOAD  (Python 3.14 compatible)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_onnx_model():
    """
    Download ONNX model from Google Drive (if needed) and
    return an onnxruntime InferenceSession.
    Returns (session, status, message)
    """
    # 1. Check onnxruntime is available
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "failed", "onnxruntime not installed. Add it to requirements.txt"

    # 2. Check file ID is set
    if ONNX_GDRIVE_ID == "PASTE_YOUR_ONNX_FILE_ID_HERE":
        return None, "failed", (
            "ONNX File ID not set!\n"
            "1. Run convert_model.py in Google Colab\n"
            "2. Upload the .onnx file to Google Drive\n"
            "3. Paste the new File ID in app.py → ONNX_GDRIVE_ID"
        )

    # 3. Download if not cached
    if not os.path.exists(ONNX_PATH):
        downloaded = False

        # Try gdown first
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={ONNX_GDRIVE_ID}"
            gdown.download(url, ONNX_PATH, quiet=False)
            if os.path.exists(ONNX_PATH) and os.path.getsize(ONNX_PATH) > 10_000:
                downloaded = True
        except Exception:
            pass

        # Fallback: requests streaming
        if not downloaded:
            try:
                import requests
                sess = requests.Session()
                BASE = "https://docs.google.com/uc?export=download"
                r    = sess.get(BASE, params={"id": ONNX_GDRIVE_ID}, stream=True)
                tok  = next(
                    (v for k, v in r.cookies.items() if k.startswith("download_warning")),
                    None
                )
                if tok:
                    r = sess.get(BASE, params={"id": ONNX_GDRIVE_ID, "confirm": tok}, stream=True)
                with open(ONNX_PATH, "wb") as f:
                    for chunk in r.iter_content(32768):
                        if chunk: f.write(chunk)
                if os.path.exists(ONNX_PATH) and os.path.getsize(ONNX_PATH) > 10_000:
                    downloaded = True
            except Exception as e:
                return None, "failed", f"Download failed: {e}"

        if not downloaded:
            return None, "failed", "Model file could not be downloaded or is too small."

    # 4. Load ONNX session
    try:
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        session   = ort.InferenceSession(ONNX_PATH, providers=providers)
        # Quick sanity check
        inp_name  = session.get_inputs()[0].name
        dummy     = np.random.rand(1, 224, 224, 3).astype(np.float32)
        session.run(None, {inp_name: dummy})
        return session, "loaded", "ONNX Model Online"
    except Exception as e:
        return None, "failed", f"Model load error: {e}"


def get_model():
    if st.session_state.model_status == "pending":
        with st.spinner("🔄 Loading AI model — please wait…"):
            sess, status, msg = load_onnx_model()
        st.session_state.sess         = sess
        st.session_state.model_status = status
        st.session_state.model_msg    = msg
    return st.session_state.sess, st.session_state.model_status, st.session_state.model_msg

# ══════════════════════════════════════════════════════════════════
# IMAGE VALIDATION
# ══════════════════════════════════════════════════════════════════
def validate_image(img: Image.Image):
    if img.mode not in ("RGB", "RGBA", "L"):
        return False, "Image must be RGB format."
    if img.width < 128 or img.height < 128:
        return False, f"Too small ({img.width}×{img.height}px). Minimum 128×128 required."
    arr  = np.array(img.convert("RGB"), dtype=np.float32)
    mean = arr.mean()
    if mean < 20:
        return False, "Image is too dark. Upload a well-lit dermoscopic image."
    if mean > 235:
        return False, "Image is overexposed. Upload a properly lit image."
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    if np.array(edges, dtype=np.float32).var() < 35:
        return False, "Image is blurry. Upload a sharp dermoscopic image."
    return True, "ok"

# ══════════════════════════════════════════════════════════════════
# PREDICTION — REAL ONNX INFERENCE
# ══════════════════════════════════════════════════════════════════
def predict(session, img: Image.Image) -> dict:
    """
    Pipeline:
      1. Resize → 224×224
      2. Convert to RGB
      3. Normalize → /255
      4. Expand dims → (1,224,224,3)
      5. ONNX inference
    """
    arr    = np.array(img.convert("RGB").resize((224, 224), Image.LANCZOS),
                      dtype=np.float32) / 255.0
    tensor = np.expand_dims(arr, axis=0)          # (1,224,224,3)

    inp_name = session.get_inputs()[0].name
    raw      = session.run(None, {inp_name: tensor})[0]   # shape (1,1) or (1,2)

    if raw.shape[-1] == 1:
        mal_p = float(raw[0][0])
        ben_p = 1.0 - mal_p
    else:
        ben_p = float(raw[0][0])
        mal_p = float(raw[0][1])

    label = "Malignant" if mal_p >= 0.50 else "Benign"
    conf  = round(max(mal_p, ben_p) * 100.0, 1)
    risk  = ("Critical" if conf >= 75 and label == "Malignant"
             else "Low"  if conf >= 75 and label == "Benign"
             else "Medium")

    return {
        "label":   label,
        "conf":    conf,
        "mal_pct": round(mal_p * 100.0, 1),
        "ben_pct": round(ben_p * 100.0, 1),
        "risk":    risk,
    }

# ══════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════
def make_report(pat: dict, res: dict) -> str:
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rid = f"RPT-{int(time.time())}"
    if res["label"] == "Malignant":
        tx  = ("• Immediate oncology referral\n"
               "• Surgical biopsy strongly recommended\n"
               "• Avoid direct UV / sun exposure\n"
               "• Emergency follow-up within 7 days")
        rec = "URGENT: Consult a dermatologist / oncologist immediately."
    else:
        tx  = ("• Schedule annual skin screening\n"
               "• Apply SPF 50+ sunscreen daily\n"
               "• Monthly self-examination\n"
               "• Consult dermatologist if lesion changes")
        rec = "Routine monitoring advised. Recheck in 6–12 months."
    return (
        "╔══════════════════════════════════════════════════════════╗\n"
        "       SKINSCAN AI — CLINICAL DIAGNOSTIC REPORT\n"
        "╚══════════════════════════════════════════════════════════╝\n"
        f"Report ID  : {rid}\n"
        f"Generated  : {ts}\n"
        f"Version    : SkinScan AI v12.0 (ONNX Runtime)\n\n"
        "━━━━━━━━ PATIENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Name       : {pat.get('name','N/A')}\n"
        f"Age        : {pat.get('age','N/A')}\n"
        f"Gender     : {pat.get('gender','N/A')}\n"
        f"Patient ID : {pat.get('pid','N/A')}\n\n"
        "━━━━━━━━ AI DIAGNOSIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Class      : {res['label'].upper()}\n"
        f"Confidence : {res['conf']}%\n"
        f"Malignant  : {res['mal_pct']}%\n"
        f"Benign     : {res['ben_pct']}%\n"
        f"Risk       : {res['risk'].upper()}\n\n"
        "━━━━━━━━ RECOMMENDATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{rec}\n\n"
        "━━━━━━━━ TREATMENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{tx}\n\n"
        "━━━━━━━━ DISCLAIMER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "AI result assists clinicians only. NOT a replacement for\n"
        "professional medical diagnosis. Confirm with a physician.\n\n"
        "University of Agriculture Faisalabad | Rehan Shafique\n"
        "rehanshafiq6540@gmail.com | SkinScan AI v12.0\n"
        "══════════════════════════════════════════════════════════"
    )

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:22px 14px 14px;text-align:center;">
          <div style="display:inline-flex;align-items:center;justify-content:center;
                      width:60px;height:60px;border-radius:15px;
                      background:linear-gradient(135deg,rgba(0,212,255,.18),rgba(0,255,136,.12));
                      border:1.5px solid rgba(0,212,255,.40);margin-bottom:10px;">
            <span style="font-size:28px;">🧬</span>
          </div>
          <div style="font-family:Orbitron,monospace;font-size:.95rem;font-weight:800;
                      color:#00d4ff;">SkinScan AI</div>
          <div style="font-size:.58rem;color:#2a5070;letter-spacing:.15em;
                      text-transform:uppercase;margin-top:3px;">
            Clinical Diagnostic v12.0
          </div>
        </div>""", unsafe_allow_html=True)

        # Model status badge
        s = st.session_state.model_status
        col = "#00ff88" if s == "loaded" else "#ffcc00" if s == "pending" else "#ff3355"
        lbl = ("CNN Online (ONNX)" if s == "loaded"
               else "Loading…"     if s == "pending"
               else "Model Error")
        st.markdown(
            f'<div style="margin:0 10px 12px;padding:6px 12px;border-radius:8px;'
            f'background:{col}10;border:1px solid {col}44;'
            f'display:flex;align-items:center;gap:8px;">'
            f'<span style="width:7px;height:7px;border-radius:50%;background:{col};'
            f'box-shadow:0 0 8px {col};display:inline-block;"></span>'
            f'<span style="font-size:.68rem;color:{col};'
            f'font-family:Orbitron,monospace;">{lbl}</span></div>',
            unsafe_allow_html=True,
        )

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
            f'<div style="padding:0 6px 8px;font-size:.68rem;color:#2a5070;line-height:2;">'
            f'🔬 Total: <b style="color:#00d4ff;">{st.session_state.total_scans}</b><br>'
            f'⚠️ Malignant: <b style="color:#ff3355;">{st.session_state.malignant_cnt}</b><br>'
            f'✅ Benign: <b style="color:#00ff88;">{st.session_state.benign_cnt}</b></div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════
# PAGE — HOME
# ══════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 2rem;">
      <div style="display:inline-flex;align-items:center;justify-content:center;
                  width:96px;height:96px;border-radius:24px;
                  background:linear-gradient(135deg,rgba(0,212,255,.18),rgba(0,255,136,.12));
                  border:1.5px solid rgba(0,212,255,.45);
                  box-shadow:0 0 50px rgba(0,212,255,.20);margin-bottom:1.3rem;">
        <span style="font-size:46px;">🧬</span>
      </div>
      <h1 style="font-family:Orbitron,monospace;font-size:2.7rem;font-weight:900;
                 background:linear-gradient(90deg,#00d4ff,#00ff88);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 margin-bottom:.5rem;">SkinScan AI</h1>
      <div style="font-size:.85rem;color:#2a5070;letter-spacing:.18em;
                  text-transform:uppercase;margin-bottom:.9rem;">
        Clinical Diagnostic System v12.0
      </div>
      <div style="font-size:1.05rem;color:#7aaccc;max-width:580px;
                  margin:0 auto 2rem;line-height:1.8;">
        Hospital-grade AI skin cancer detection powered by Deep Learning.<br>
        Instant <b style="color:#00ff88;">Benign</b> vs
        <b style="color:#ff3355;">Malignant</b> classification.
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1.2, 2])
    with c2:
        if st.button("⚡  START DIAGNOSIS", use_container_width=True, type="primary"):
            st.session_state.page = "Scan"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Steps
    st.markdown(
        '<div style="text-align:center;margin-bottom:1.4rem;">'
        '<span style="font-family:Orbitron,monospace;font-size:.8rem;color:#00d4ff;'
        'letter-spacing:.15em;text-transform:uppercase;">📘 How It Works</span></div>',
        unsafe_allow_html=True,
    )
    steps = [
        ("01","🖼️","Upload Image",   "Drag & drop a dermoscopic image (JPG/PNG, min 128×128)"),
        ("02","⚡", "Run AI Scan",    "Click Execute Deep Scan to trigger ONNX inference"),
        ("03","🧠","AI Processing",  "Real-time inference through neural network layers"),
        ("04","📊","View Result",    "Diagnosis, confidence %, risk level & charts"),
        ("05","📄","Export Report",  "Download medical report or export patient CSV"),
    ]
    cols = st.columns(5)
    for i, (num, icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(
                f'<div style="background:rgba(5,16,34,.80);border:1px solid rgba(0,212,255,.13);'
                f'border-radius:14px;padding:1.1rem .7rem;text-align:center;height:168px;">'
                f'<div style="font-family:Orbitron,monospace;font-size:1.5rem;font-weight:900;'
                f'color:rgba(0,212,255,.18);margin-bottom:5px;">{num}</div>'
                f'<div style="font-size:1.5rem;margin-bottom:8px;">{icon}</div>'
                f'<div style="font-family:Orbitron,monospace;font-size:.58rem;color:#00d4ff;'
                f'font-weight:700;margin-bottom:5px;">{title}</div>'
                f'<div style="font-size:.70rem;color:#2a5070;line-height:1.5;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Features
    st.markdown(
        '<div style="text-align:center;margin-bottom:1.4rem;">'
        '<span style="font-family:Orbitron,monospace;font-size:.8rem;color:#00d4ff;'
        'letter-spacing:.15em;text-transform:uppercase;">🔬 Key Features</span></div>',
        unsafe_allow_html=True,
    )
    feats = [
        ("🧠","ONNX Inference",      "Runs on Python 3.14 — no TensorFlow needed at runtime"),
        ("⚡","Real Prediction",     "No demo, no random output — genuine CNN model output"),
        ("📄","Medical Reports",     "Auto-generated clinical reports ready for download"),
        ("📊","Analytics Dashboard", "Scan history, pie charts, confidence trend graphs"),
        ("🗂️","Patient Registry",    "Structured patient data & CSV export"),
        ("✅","Image Validation",    "Rejects blurry, dark, or low-quality images"),
    ]
    fc1, fc2, fc3 = st.columns(3)
    for i, (icon, title, desc) in enumerate(feats):
        with [fc1, fc2, fc3][i % 3]:
            st.markdown(
                f'<div style="background:rgba(5,16,34,.75);border:1px solid rgba(0,212,255,.11);'
                f'border-radius:12px;padding:.9rem 1rem;margin-bottom:.7rem;'
                f'display:flex;gap:11px;align-items:flex-start;">'
                f'<span style="font-size:1.4rem;flex-shrink:0;">{icon}</span>'
                f'<div><div style="font-family:Orbitron,monospace;font-size:.62rem;color:#00d4ff;'
                f'font-weight:700;margin-bottom:3px;">{title}</div>'
                f'<div style="font-size:.73rem;color:#2a5070;line-height:1.5;">{desc}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # Dataset stats
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;margin-bottom:1.2rem;">'
        '<span style="font-family:Orbitron,monospace;font-size:.8rem;color:#00d4ff;'
        'letter-spacing:.15em;text-transform:uppercase;">📂 Dataset — Melanoma (Kaggle)</span></div>',
        unsafe_allow_html=True,
    )
    d1, d2, d3, d4 = st.columns(4)
    for col, val, lbl in [
        (d1,"10,015","Total Images"),
        (d2,"6,705", "Benign Cases"),
        (d3,"3,310", "Malignant Cases"),
        (d4,"Kaggle","Source"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:rgba(5,16,34,.80);border:1px solid rgba(0,212,255,.14);'
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
        '<h1 style="font-family:Orbitron,monospace;font-size:1.55rem;font-weight:800;'
        'color:#00d4ff;margin-bottom:.25rem;">🔬 AI Skin Analysis</h1>'
        '<p style="color:#2a5070;font-size:.83rem;margin-bottom:1.4rem;">'
        'Upload a dermoscopic image for ONNX-powered real-time skin cancer detection</p>',
        unsafe_allow_html=True,
    )

    sess, status, msg = get_model()

    # Status banner
    if status == "loaded":
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:8px;'
            'background:rgba(0,255,136,.07);border:1px solid rgba(0,255,136,.25);'
            'border-radius:8px;padding:5px 13px;margin-bottom:1rem;">'
            '<span style="width:7px;height:7px;border-radius:50%;background:#00ff88;'
            'box-shadow:0 0 8px #00ff88;display:inline-block;"></span>'
            '<span style="font-size:.70rem;color:#00ff88;'
            'font-family:Orbitron,monospace;">ONNX Model Online — Real Inference Active</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Show setup instructions if file ID not set
        if "ONNX File ID not set" in msg:
            st.markdown(
                '<div style="background:rgba(255,204,0,.07);border:1px solid rgba(255,204,0,.30);'
                'border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">'
                '<div style="font-family:Orbitron,monospace;font-size:.72rem;'
                'color:#ffcc00;font-weight:700;margin-bottom:.7rem;">'
                '⚙️ ONE-TIME SETUP REQUIRED</div>'
                '<div style="font-size:.82rem;color:#8ab0cc;line-height:2.1;">'
                '▸ Step 1: Open <b>convert_model.py</b> in Google Colab<br>'
                '▸ Step 2: Run all cells — it converts your .h5 to .onnx<br>'
                '▸ Step 3: Copy the new Google Drive File ID<br>'
                '▸ Step 4: Paste it in <b>app.py → ONNX_GDRIVE_ID</b><br>'
                '▸ Step 5: Push to GitHub — app will work everywhere!'
                '</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:rgba(255,51,85,.07);border:1px solid rgba(255,51,85,.30);'
                f'border-radius:10px;padding:.9rem 1.1rem;margin-bottom:1rem;">'
                f'<b style="color:#ff3355;">❌ Model Error</b><br>'
                f'<span style="font-size:.78rem;color:#8ab0cc;">{msg}</span></div>',
                unsafe_allow_html=True,
            )
        st.stop()

    # Patient info
    with st.expander("👤 Patient Information", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        p_name   = c1.text_input("Full Name",  placeholder="e.g. Ahmed Ali")
        p_age    = c2.number_input("Age", 1, 120, 30)
        p_gender = c3.selectbox("Gender", ["Male","Female","Other"])
        p_id     = c4.text_input("Patient ID", placeholder="PT-001")

    st.markdown("#### 🖼️ Upload Dermoscopic Image")
    uploaded = st.file_uploader(
        "JPG / PNG / BMP — min 128×128 px",
        type=["jpg","jpeg","png","bmp"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown(
            '<div style="border:2px dashed rgba(0,212,255,.18);border-radius:14px;'
            'padding:3rem;text-align:center;color:#2a5070;">'
            '<div style="font-size:3rem;margin-bottom:.5rem;">🖼️</div>'
            '<div>Drag & drop a dermoscopic image here</div>'
            '<div style="font-size:.72rem;margin-top:5px;">JPG · PNG · BMP · min 128×128 px</div>'
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
            f'Resolution: {img.width}×{img.height}px | Mode: {img.mode}</div>',
            unsafe_allow_html=True,
        )

    with col_ctrl:
        ok, vmsg = validate_image(img)
        if not ok:
            st.markdown(
                f'<div style="background:rgba(255,51,85,.07);border:1px solid rgba(255,51,85,.28);'
                f'border-radius:10px;padding:1rem 1.2rem;color:#ff3355;">'
                f'<b>❌ Invalid Image</b><br>'
                f'<span style="font-size:.80rem;">{vmsg}</span><br><br>'
                f'<span style="font-size:.75rem;color:#2a5070;">'
                f'Please upload a clear dermoscopic skin image.</span></div>',
                unsafe_allow_html=True,
            )
            return

        st.markdown(
            '<div style="background:rgba(0,255,136,.06);border:1px solid rgba(0,255,136,.20);'
            'border-radius:10px;padding:.65rem 1rem;color:#00ff88;font-size:.78rem;'
            'margin-bottom:1rem;">✅ Image quality validated — ready for scan</div>',
            unsafe_allow_html=True,
        )
        run = st.button("⚡  EXECUTE DEEP SCAN", use_container_width=True, type="primary")

    if not run:
        return

    # Inference
    prog = st.progress(0, text="Initialising neural network…")
    for pct, txt in [(20,"Preprocessing image…"),(45,"Extracting features…"),
                     (70,"Running ONNX inference…"),(90,"Computing output…"),(100,"Done!")]:
        time.sleep(0.20)
        prog.progress(pct, text=txt)
    prog.empty()

    try:
        res = predict(sess, img)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # Low-confidence warning
    if res["conf"] < 60:
        st.markdown(
            f'<div style="background:rgba(255,204,0,.06);border:1px solid rgba(255,204,0,.28);'
            f'border-radius:10px;padding:.75rem 1.1rem;color:#ffcc00;font-size:.78rem;'
            f'margin-bottom:1rem;">⚠️ <b>Low Reliability</b> — Confidence {res["conf"]}% '
            f'(&lt;60%). Upload a higher-quality image for better accuracy.</div>',
            unsafe_allow_html=True,
        )

    # Result card
    lc  = "#ff3355" if res["label"] == "Malignant" else "#00ff88"
    rc  = {"Critical":"#ff3355","Medium":"#ffcc00","Low":"#00ff88"}[res["risk"]]
    ico = "⚠️" if res["label"] == "Malignant" else "✅"

    st.markdown(
        f'<div style="background:rgba(5,16,34,.92);border:1.5px solid {lc}44;'
        f'border-radius:16px;padding:1.6rem 1.9rem;box-shadow:0 0 32px {lc}12;margin-top:.4rem;">'
        f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:1.2rem;">'
        f'<div style="width:54px;height:54px;border-radius:12px;background:{lc}16;'
        f'border:1px solid {lc}44;display:flex;align-items:center;justify-content:center;'
        f'font-size:1.6rem;">{ico}</div>'
        f'<div><div style="font-family:Orbitron,monospace;font-size:1.4rem;'
        f'font-weight:800;color:{lc};">{res["label"].upper()}</div>'
        f'<div style="font-size:.73rem;color:#2a5070;">AI Classification</div></div>'
        f'<div style="margin-left:auto;text-align:right;">'
        f'<div style="font-family:Orbitron,monospace;font-size:2rem;'
        f'font-weight:900;color:#c8e4f8;">{res["conf"]}%</div>'
        f'<div style="font-size:.68rem;color:#2a5070;">Confidence</div></div></div>'
        f'<div style="height:5px;background:rgba(255,255,255,.05);border-radius:5px;margin-bottom:1.2rem;">'
        f'<div style="height:100%;width:{res["conf"]}%;'
        f'background:linear-gradient(90deg,{lc}66,{lc});border-radius:5px;"></div></div>'
        f'<div style="display:flex;gap:10px;">'
        + "".join(
            f'<div style="flex:1;background:rgba(255,255,255,.03);border-radius:8px;'
            f'padding:8px 11px;border:1px solid rgba(255,255,255,.05);">'
            f'<div style="font-size:.63rem;color:#2a5070;text-transform:uppercase;'
            f'letter-spacing:.08em;">{k}</div>'
            f'<div style="font-family:Orbitron,monospace;font-size:.88rem;'
            f'font-weight:700;color:{vc};">{vv}</div></div>'
            for k, vv, vc in [
                ("Risk",      res["risk"],         rc),
                ("Malignant", f'{res["mal_pct"]}%', "#ff3355"),
                ("Benign",    f'{res["ben_pct"]}%', "#00ff88"),
            ]
        )
        + '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        fig = go.Figure(go.Bar(
            x=["Benign","Malignant"],
            y=[res["ben_pct"],res["mal_pct"]],
            marker_color=["#00ff88","#ff3355"],
            text=[f'{res["ben_pct"]}%',f'{res["mal_pct"]}%'],
            textposition="outside",
        ))
        fig.update_layout(
            title="Probability Distribution",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8e4f8", font_family="Orbitron",
            yaxis=dict(range=[0,115],showgrid=False,color="#2a5070"),
            xaxis=dict(color="#2a5070"), height=270,
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number", value=res["conf"],
            title={"text":"Confidence","font":{"color":"#c8e4f8","family":"Orbitron","size":13}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#2a5070"},
                "bar":{"color":lc},
                "bgcolor":"rgba(0,0,0,0)",
                "steps":[
                    {"range":[0,50],"color":"rgba(255,51,85,.09)"},
                    {"range":[50,75],"color":"rgba(255,204,0,.09)"},
                    {"range":[75,100],"color":"rgba(0,255,136,.09)"},
                ],
            },
            number={"suffix":"%","font":{"color":"#c8e4f8","family":"Orbitron"}},
        ))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#c8e4f8",height=270)
        st.plotly_chart(fig2, use_container_width=True)

    # Medical recommendations
    if res["label"] == "Malignant":
        st.markdown(
            '<div style="background:rgba(255,51,85,.06);border:1px solid rgba(255,51,85,.28);'
            'border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:.9rem;">'
            '<div style="font-family:Orbitron,monospace;font-size:.72rem;color:#ff3355;'
            'font-weight:700;margin-bottom:.6rem;">🚨 MALIGNANT — URGENT ACTION REQUIRED</div>'
            '<div style="font-size:.80rem;color:#8ab0cc;line-height:2.1;">'
            '▸ Immediate oncology referral<br>'
            '▸ Surgical biopsy strongly recommended<br>'
            '▸ Avoid UV / sun exposure<br>'
            '▸ Follow-up within 7 days<br>'
            '▸ Do NOT self-treat</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:rgba(0,255,136,.05);border:1px solid rgba(0,255,136,.20);'
            'border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:.9rem;">'
            '<div style="font-family:Orbitron,monospace;font-size:.72rem;color:#00ff88;'
            'font-weight:700;margin-bottom:.6rem;">✅ BENIGN — ROUTINE MONITORING ADVISED</div>'
            '<div style="font-size:.80rem;color:#8ab0cc;line-height:2.1;">'
            '▸ Annual skin screening<br>'
            '▸ SPF 50+ sunscreen daily<br>'
            '▸ Monthly self-examination<br>'
            '▸ Consult if lesion changes</div></div>',
            unsafe_allow_html=True,
        )

    # Save record
    patient = {
        "name":   p_name   or "Unknown",
        "age":    p_age,
        "gender": p_gender,
        "pid":    p_id     or f"PT-{st.session_state.total_scans+1:03d}",
    }
    st.session_state.total_scans += 1
    if res["label"] == "Malignant": st.session_state.malignant_cnt += 1
    else:                           st.session_state.benign_cnt    += 1
    st.session_state.scan_history.append({
        "scan_id":    f"SC-{st.session_state.total_scans:03d}",
        "patient_id": patient["pid"],
        "name":       patient["name"],
        "diagnosis":  res["label"],
        "confidence": f'{res["conf"]}%',
        "risk":       res["risk"],
        "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

    # Action buttons
    st.markdown("#### 📋 Actions")
    b1, b2, b3, b4 = st.columns(4)
    report = make_report(patient, res)
    with b1:
        st.download_button("📄 Medical Report", report,
                           f'Report_{patient["pid"]}.txt', "text/plain",
                           use_container_width=True)
    with b2:
        st.markdown(
            f'<div style="background:rgba(0,255,136,.06);border:1px solid rgba(0,255,136,.20);'
            f'border-radius:8px;padding:.55rem .8rem;font-size:.72rem;color:#00ff88;'
            f'text-align:center;">💾 Saved — {patient["pid"]}</div>',
            unsafe_allow_html=True,
        )
    with b3:
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"; st.rerun()
    with b4:
        csv = pd.DataFrame(st.session_state.scan_history).to_csv(index=False)
        st.download_button("📥 Export CSV", csv, "SkinScan_Records.csv",
                           "text/csv", use_container_width=True)

    st.markdown(
        '<div style="margin-top:1rem;padding:.75rem 1rem;'
        'background:rgba(255,204,0,.04);border:1px solid rgba(255,204,0,.14);'
        'border-radius:8px;font-size:.70rem;color:#2a5070;line-height:1.7;">'
        '<b style="color:#ffcc00;">⚠️ Disclaimer:</b> '
        'AI result assists clinicians only. Confirm with a licensed physician.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# PAGE — DASHBOARD
# ══════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown(
        '<h1 style="font-family:Orbitron,monospace;font-size:1.55rem;font-weight:800;'
        'color:#00d4ff;margin-bottom:.25rem;">📊 Analytics Dashboard</h1>'
        '<p style="color:#2a5070;font-size:.83rem;margin-bottom:1.4rem;">'
        'Real-time diagnostic metrics</p>',
        unsafe_allow_html=True,
    )
    history = st.session_state.scan_history
    confs   = [float(h["confidence"].replace("%","")) for h in history] if history else []
    avg_c   = round(np.mean(confs),1) if confs else 0.0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🔬 Total Scans",     st.session_state.total_scans)
    c2.metric("🎯 Avg Confidence",  f"{avg_c}%")
    c3.metric("⚠️ Malignant",       st.session_state.malignant_cnt)
    c4.metric("✅ Benign",           st.session_state.benign_cnt)

    st.markdown("<br>", unsafe_allow_html=True)

    if not history:
        st.markdown(
            '<div style="background:rgba(5,16,34,.75);border:1px solid rgba(0,212,255,.12);'
            'border-radius:14px;padding:3rem;text-align:center;">'
            '<div style="font-size:2.5rem;margin-bottom:.6rem;">📊</div>'
            '<div style="color:#2a5070;">No scan data yet.</div></div>',
            unsafe_allow_html=True,
        )
        return

    mal = st.session_state.malignant_cnt
    ben = st.session_state.benign_cnt
    cl, cr = st.columns(2)
    with cl:
        fig_pie = px.pie(names=["Benign","Malignant"],values=[ben,mal],
                         color_discrete_sequence=["#00ff88","#ff3355"],
                         hole=0.55,title="Diagnosis Distribution")
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#c8e4f8",title_font_family="Orbitron",
                              legend_font_color="#2a5070")
        st.plotly_chart(fig_pie, use_container_width=True)
    with cr:
        colors = ["#ff3355" if h["diagnosis"]=="Malignant" else "#00ff88" for h in history[-20:]]
        fig_bar = go.Figure(go.Bar(y=confs[-20:],x=list(range(1,len(confs[-20:])+1)),
                                   marker_color=colors))
        fig_bar.update_layout(title="Confidence per Scan (last 20)",
                              paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#c8e4f8",title_font_family="Orbitron",
                              xaxis=dict(showgrid=False,color="#2a5070"),
                              yaxis=dict(range=[0,112],gridcolor="rgba(255,255,255,.04)",
                                         color="#2a5070"))
        st.plotly_chart(fig_bar, use_container_width=True)

    if len(confs) >= 2:
        fig_line = go.Figure(go.Scatter(y=confs,mode="lines+markers",
                                        line=dict(color="#00d4ff",width=2),
                                        marker=dict(color="#00d4ff",size=5),
                                        fill="tozeroy",fillcolor="rgba(0,212,255,.05)"))
        fig_line.update_layout(title="Confidence Trend",
                               paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#c8e4f8",title_font_family="Orbitron",
                               xaxis=dict(showgrid=False,color="#2a5070"),
                               yaxis=dict(range=[0,112],gridcolor="rgba(255,255,255,.04)",
                                           color="#2a5070"))
        st.plotly_chart(fig_line, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE — REGISTRY
# ══════════════════════════════════════════════════════════════════
def page_registry():
    st.markdown(
        '<h1 style="font-family:Orbitron,monospace;font-size:1.55rem;font-weight:800;'
        'color:#00d4ff;margin-bottom:.25rem;">🗂️ Patient Registry</h1>'
        '<p style="color:#2a5070;font-size:.83rem;margin-bottom:1.4rem;">'
        'Scan records and patient data management</p>',
        unsafe_allow_html=True,
    )
    if not st.session_state.scan_history:
        st.markdown(
            '<div style="background:rgba(5,16,34,.75);border:1px solid rgba(0,212,255,.12);'
            'border-radius:14px;padding:3rem;text-align:center;">'
            '<div style="font-size:2.5rem;margin-bottom:.6rem;">🗂️</div>'
            '<div style="color:#2a5070;">No records yet.</div></div>',
            unsafe_allow_html=True,
        )
        return
    df = pd.DataFrame(st.session_state.scan_history)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("📥 Export CSV", df.to_csv(index=False),
                       "SkinScan_Registry.csv","text/csv")

# ══════════════════════════════════════════════════════════════════
# PAGE — USER GUIDE
# ══════════════════════════════════════════════════════════════════
def page_guide():
    st.markdown(
        '<h1 style="font-family:Orbitron,monospace;font-size:1.55rem;font-weight:800;'
        'color:#00d4ff;margin-bottom:.25rem;">📘 User Guide</h1>'
        '<p style="color:#2a5070;font-size:.83rem;margin-bottom:1.4rem;">'
        'Step-by-step system walkthrough</p>',
        unsafe_allow_html=True,
    )
    steps = [
        ("01","⚙️","One-Time Setup",
         "Run convert_model.py in Google Colab to convert .h5 to .onnx. Paste the new File ID in app.py → ONNX_GDRIVE_ID."),
        ("02","👤","Enter Patient Details",
         "Fill in name, age, gender, and patient ID in the AI Scan page."),
        ("03","🖼️","Upload Dermoscopic Image",
         "Upload a clear, well-lit JPEG or PNG skin image. Minimum 128x128 pixels."),
        ("04","⚡","Execute Deep Scan",
         "Click EXECUTE DEEP SCAN. Real ONNX inference — no simulation, no random output."),
        ("05","🧬","View AI Result",
         "Review Malignant or Benign classification, confidence %, risk level, and charts."),
        ("06","📄","Download Report",
         "Download the auto-generated medical report (.txt) for clinical records."),
        ("07","📥","Export Data",
         "Export all session records as CSV from Patient Registry."),
    ]
    for num, icon, title, desc in steps:
        st.markdown(
            f'<div style="background:rgba(5,16,34,.78);border:1px solid rgba(0,212,255,.14);'
            f'border-radius:14px;padding:1rem 1.3rem;margin-bottom:.75rem;'
            f'display:flex;align-items:flex-start;gap:13px;">'
            f'<div style="min-width:42px;height:42px;border-radius:10px;'
            f'background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.22);'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:1.25rem;flex-shrink:0;">{icon}</div>'
            f'<div><div style="font-family:Orbitron,monospace;font-size:.68rem;font-weight:700;'
            f'color:#00d4ff;margin-bottom:3px;">STEP {num} — {title.upper()}</div>'
            f'<div style="font-size:.80rem;color:#4a7a9a;line-height:1.65;">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="background:rgba(255,204,0,.05);border:1px solid rgba(255,204,0,.18);'
        'border-radius:12px;padding:1rem 1.3rem;margin-top:.4rem;">'
        '<div style="font-family:Orbitron,monospace;font-size:.68rem;color:#ffcc00;'
        'margin-bottom:5px;">⚠️ CLINICAL DISCLAIMER</div>'
        '<div style="font-size:.78rem;color:#2a5070;line-height:1.7;">'
        'SkinScan AI is an assistive tool. All results must be confirmed by a '
        'licensed clinician before any treatment.</div></div>',
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
        'SkinScan AI — Medical Diagnostic System v12.0</div></div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# MAIN
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
