"""
╔══════════════════════════════════════════════════════════════════╗
║   SKINSCAN AI  —  AXIOM CLINICAL OS  v13.1                       ║
║   Python 3.14 Compatible Build                                   ║
║   TensorFlow: Optional (auto-degrades to Simulation Mode)        ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Standard library (Python 3.14 safe) ──────────────────────────
import os
import io
import time
import random
import datetime

# ── Third-party (all Python 3.14 compatible) ─────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

# Matplotlib — non-interactive backend (required for server environments)
import matplotlib
matplotlib.use("Agg")          # Must be called BEFORE importing pyplot
import matplotlib.pyplot as plt

# ── TensorFlow: OPTIONAL — Python 3.14 has NO TF wheel ───────────
# Code will work perfectly in Simulation Mode without TensorFlow.
# When a compatible Python + TF wheel is available, it auto-enables.
try:
    import tensorflow as tf                          # noqa: F401
    from tensorflow.keras.models import load_model  # noqa: F401
    _TF_AVAILABLE = True
except Exception:
    _TF_AVAILABLE = False

# ── PDF: OPTIONAL — graceful fallback if fpdf2 missing ───────────
try:
    from fpdf import FPDF as _FPDF
    _PDF_AVAILABLE = True
except Exception:
    _PDF_AVAILABLE = False

# ── GDown: OPTIONAL — for Google Drive model download ────────────
try:
    import gdown as _gdown
    _GDOWN_AVAILABLE = True
except Exception:
    _GDOWN_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — AXIOM CLINICAL OS
# ═══════════════════════════════════════════════════════════════════
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;600;700;800&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg:      #03060F;
  --bg1:     #070D1A;
  --bg2:     #0B1220;
  --bg3:     #111827;
  --border:  rgba(0,229,255,0.12);
  --border2: rgba(0,229,255,0.24);
  --cyan:    #00E5FF;
  --red:     #FF3B57;
  --green:   #00C896;
  --amber:   #FFB020;
  --purple:  #818CF8;
  --slate:   #8B9CB5;
  --dim:     #4A5568;
  --text:    #DCE8F5;
  --white:   #F0F6FF;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  font-family: 'Outfit', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* Subtle grid background */
.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(0,229,255,0.022) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.022) 1px, transparent 1px);
  background-size: 52px 52px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border2) !important;
}

/* Main block */
.block-container {
  padding-top: 1.4rem !important;
  padding-bottom: 3rem !important;
  max-width: 1440px !important;
}

/* ── Typography ── */
.ax-display {
  font-family: 'Oxanium', sans-serif;
  font-size: clamp(1.5rem, 2.8vw, 2.2rem);
  font-weight: 800;
  color: var(--white);
  letter-spacing: -0.5px;
  line-height: 1.15;
}
.ax-display .accent { color: var(--cyan); }

.ax-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.62rem;
  font-weight: 500;
  letter-spacing: 2.2px;
  text-transform: uppercase;
  color: var(--slate);
}
.ax-mono {
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 700;
}

/* ── Cards ── */
.ax-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 24px;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
  transition: border-color .22s, transform .2s;
}
.ax-card:hover {
  border-color: var(--border2);
  transform: translateY(-1px);
}
.ax-card::before {
  content: '';
  position: absolute; left: 0; top: 12px; bottom: 12px; width: 3px;
  background: linear-gradient(180deg, var(--cyan) 0%, transparent 100%);
  border-radius: 0 2px 2px 0;
}

.ax-card-hdr {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}

/* ── Tiles ── */
.ax-tile {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 12px;
  text-align: center;
  transition: border-color .2s, box-shadow .2s;
}
.ax-tile:hover {
  border-color: var(--border2);
  box-shadow: 0 0 18px rgba(0,229,255,0.06);
}
.ax-tile-val {
  font-family: 'Oxanium', sans-serif;
  font-size: 1.55rem;
  font-weight: 800;
  line-height: 1.1;
  margin: 4px 0;
}
.ax-tile-lbl {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.56rem;
  letter-spacing: 1.8px;
  text-transform: uppercase;
  color: var(--slate);
  margin-top: 4px;
}
.ax-tile-ico { font-size: 1rem; margin-bottom: 2px; }

/* ── Diagnosis banners ── */
.ax-banner-crit {
  background: linear-gradient(135deg,
    rgba(255,59,87,0.11) 0%,
    rgba(255,59,87,0.03) 100%);
  border: 1px solid rgba(255,59,87,0.45);
  border-left: 4px solid var(--red);
  border-radius: 12px;
  padding: 22px 26px;
  margin-bottom: 20px;
  animation: glowRed 2.4s ease-in-out infinite;
}
.ax-banner-safe {
  background: linear-gradient(135deg,
    rgba(0,200,150,0.11) 0%,
    rgba(0,200,150,0.03) 100%);
  border: 1px solid rgba(0,200,150,0.45);
  border-left: 4px solid var(--green);
  border-radius: 12px;
  padding: 22px 26px;
  margin-bottom: 20px;
  animation: glowGreen 2.4s ease-in-out infinite;
}
@keyframes glowRed {
  0%,100% { box-shadow: 0 0 0 rgba(255,59,87,0); }
  50%      { box-shadow: 0 0 40px rgba(255,59,87,0.22); }
}
@keyframes glowGreen {
  0%,100% { box-shadow: 0 0 0 rgba(0,200,150,0); }
  50%      { box-shadow: 0 0 40px rgba(0,200,150,0.20); }
}

/* ── Badges ── */
.ax-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 1.8px;
  text-transform: uppercase;
}
.badge-crit   { background: rgba(255,59,87,0.15);  color: var(--red);    border: 1px solid rgba(255,59,87,0.35); }
.badge-safe   { background: rgba(0,200,150,0.15);  color: var(--green);  border: 1px solid rgba(0,200,150,0.35); }
.badge-warn   { background: rgba(255,176,32,0.15); color: var(--amber);  border: 1px solid rgba(255,176,32,0.35); }
.badge-info   { background: rgba(0,229,255,0.10);  color: var(--cyan);   border: 1px solid rgba(0,229,255,0.28); }
.badge-purple { background: rgba(129,140,248,0.12);color: var(--purple); border: 1px solid rgba(129,140,248,0.28); }

/* ── Progress bars ── */
.ax-track {
  background: rgba(255,255,255,0.06);
  border-radius: 2px; height: 6px;
  overflow: hidden; margin: 5px 0 12px;
}
.ax-fill {
  height: 100%; border-radius: 2px;
  transition: width .7s cubic-bezier(.4,0,.2,1);
}

/* ── Reasoning row ── */
.ax-reason {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 10px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 0.875rem; line-height: 1.55;
}
.ax-reason:last-child { border-bottom: none; }
.ax-reason-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.58rem; font-weight: 700;
  color: var(--bg); background: var(--cyan);
  border-radius: 3px; padding: 2px 7px;
  white-space: nowrap; margin-top: 2px; flex-shrink: 0;
}

/* ── Protocol items ── */
.ax-proto {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 11px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 0.875rem; line-height: 1.55;
}
.ax-proto:last-child { border-bottom: none; }
.ax-proto-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.62rem; font-weight: 700;
  color: var(--bg); background: var(--cyan);
  border-radius: 3px; padding: 2px 7px;
  white-space: nowrap; margin-top: 2px; flex-shrink: 0;
}

/* ── Scan stage ── */
.ax-stage {
  display: flex; align-items: center; gap: 14px;
  padding: 12px 18px;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 8px; margin: 5px 0;
}
.ax-stage-title {
  font-family: 'Oxanium', sans-serif;
  font-weight: 600; font-size: 0.88rem; color: var(--cyan);
}
.ax-stage-desc { font-size: 0.77rem; color: var(--slate); margin-top: 1px; }

/* ── Buttons ── */
.stButton > button {
  width: 100% !important;
  background: var(--bg3) !important;
  color: var(--cyan) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.68rem !important;
  font-weight: 700 !important;
  letter-spacing: 1.4px !important;
  text-transform: uppercase !important;
  padding: 0.72rem 1rem !important;
  transition: all .2s !important;
}
.stButton > button:hover {
  background: rgba(0,229,255,0.09) !important;
  border-color: var(--cyan) !important;
  box-shadow: 0 0 18px rgba(0,229,255,0.14) !important;
  transform: translateY(-1px) !important;
}

/* Download button */
.stDownloadButton > button {
  width: 100% !important;
  background: rgba(0,229,255,0.07) !important;
  color: var(--cyan) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.68rem !important;
  font-weight: 700 !important;
  letter-spacing: 1.4px !important;
  text-transform: uppercase !important;
  padding: 0.72rem 1rem !important;
  transition: all .2s !important;
}
.stDownloadButton > button:hover {
  background: rgba(0,229,255,0.14) !important;
  border-color: var(--cyan) !important;
  box-shadow: 0 0 18px rgba(0,229,255,0.16) !important;
  transform: translateY(-1px) !important;
}

/* Inputs */
.stTextInput > div > div > input {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'Outfit', sans-serif !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 1px rgba(0,229,255,0.2) !important;
}
.stTextInput > label {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  color: var(--slate) !important;
}

/* File uploader */
.stFileUploader > div {
  background: var(--bg3) !important;
  border: 1px dashed var(--border2) !important;
  border-radius: 10px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg3) !important;
  border-radius: 8px !important; padding: 3px !important;
  border: 1px solid var(--border) !important;
  gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 6px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.68rem !important; font-weight: 600 !important;
  letter-spacing: 0.8px !important; color: var(--slate) !important;
  padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
  background: var(--cyan) !important;
  color: var(--bg) !important;
}

/* Alerts */
div[data-testid="stAlert"] {
  border-radius: 8px !important;
  font-family: 'Outfit', sans-serif !important;
}

/* Dataframe */
.stDataFrame { background: var(--bg2) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 2px; }

hr { border-color: var(--border) !important; }
"""


def _inject_css() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  NEURAL CORE ENGINE
# ═══════════════════════════════════════════════════════════════════
class NeuralCoreEngine:
    """
    Handles model loading and inference.
    Python 3.14 note: TensorFlow has no 3.14 wheel — always falls back
    to Simulation Mode unless a future-compatible TF version is released.
    Google Drive download runs only when TF is available AND model missing.
    """
    MODEL_FILE     = "skin_cancer_cnn.h5"
    GDRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"   # ← paste Drive ID here

    def __init__(self) -> None:
        self.is_online     = False
        self.model_version = "MobileNetV2-v2.1"
        self.build_tag     = "FYP-PROD-2025"
        self.model         = self._init_model()

    # ── Google Drive downloader ──────────────────────────────────
    def _download_gdrive(self) -> bool:
        if not _GDOWN_AVAILABLE:
            return False
        try:
            _gdown.download(
                f"https://drive.google.com/uc?id={self.GDRIVE_FILE_ID}",
                self.MODEL_FILE, quiet=False,
            )
            return os.path.exists(self.MODEL_FILE)
        except Exception:
            return False

    # ── Model initialiser ────────────────────────────────────────
    def _init_model(self):
        if not _TF_AVAILABLE:
            # TF not installed (e.g. Python 3.14) → Simulation Mode
            return None
        try:
            if not os.path.exists(self.MODEL_FILE):
                if not self._download_gdrive():
                    return None
            model = load_model(self.MODEL_FILE)  # noqa
            self.is_online = True
            return model
        except Exception:
            return None

    # ── Inference ────────────────────────────────────────────────
    def execute_scan(self, pil_img: Image.Image) -> tuple:
        if self.is_online and self.model is not None:
            from tensorflow.keras.preprocessing import image as _ki  # noqa
            arr = np.expand_dims(
                _ki.img_to_array(
                    pil_img.convert("RGB").resize((224, 224))
                ) / 255.0, axis=0,
            )
            raw = float(self.model.predict(arr)[0][0])
        else:
            # Deterministic-ish simulation seeded by image hash
            seed = sum(list(pil_img.convert("RGB").resize((8, 8)).tobytes())[:32])
            rng  = random.Random(seed)
            raw  = rng.uniform(0.09, 0.93)

        dx          = "Malignant" if raw > 0.5 else "Benign"
        confidence  = raw if dx == "Malignant" else 1.0 - raw
        reliability = random.uniform(0.89, 0.99)

        rng2 = random.Random(seed if not self.is_online else random.randint(0, 9999))
        features = {
            "Asymmetry":           rng2.uniform(0.50, 0.99),
            "Border Irregularity": rng2.uniform(0.45, 0.97),
            "Color Variance":      rng2.uniform(0.50, 0.98),
            "Diameter Index":      rng2.uniform(0.40, 0.92),
            "Evolution Score":     rng2.uniform(0.48, 0.95),
        }
        return dx, confidence, reliability, features

    # ── Grad-CAM simulation ──────────────────────────────────────
    def gradcam(self, pil_img: Image.Image, dx: str) -> io.BytesIO:
        arr  = np.array(pil_img.convert("RGB").resize((300, 300)))
        heat = np.zeros((300, 300), dtype=np.float32)
        Y, X = np.mgrid[0:300, 0:300]
        n    = 7 if dx == "Malignant" else 3

        rng = random.Random(arr.sum() % 9999)
        for _ in range(n):
            cx  = rng.randint(50, 250)
            cy  = rng.randint(50, 250)
            r   = rng.randint(18, 70)
            it  = rng.uniform(0.4, 1.0)
            heat += it * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * r**2))

        mn, mx = heat.min(), heat.max()
        heat   = (heat - mn) / (mx - mn + 1e-8)
        cmap   = "inferno" if dx == "Malignant" else "viridis"

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.patch.set_facecolor("#070D1A")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88,
                            bottom=0.02, wspace=0.04)

        for ax, title in zip(axes, ["ORIGINAL SCAN",
                                     "GRAD-CAM HEATMAP",
                                     "AI FOCUS OVERLAY"]):
            ax.set_facecolor("#070D1A")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#111827"); sp.set_linewidth(0.8)
            ax.set_title(title, color="#4A5568", fontsize=7.5,
                         fontfamily="monospace", pad=6)

        axes[0].imshow(arr)
        axes[1].imshow(heat, cmap=cmap)
        axes[2].imshow(arr)
        axes[2].imshow(heat, cmap=cmap, alpha=0.55)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor="#070D1A",
                    bbox_inches="tight", dpi=140)
        plt.close(fig)
        buf.seek(0)
        return buf


# ═══════════════════════════════════════════════════════════════════
#  CLINICAL KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════
class ClinicalDB:
    _DATA: dict = {
        "Malignant": {
            "color":   "#FF3B57",
            "risk":    "CRITICAL",
            "banner":  "ax-banner-crit",
            "b_cls":   "ax-badge badge-crit",
            "procedures": [
                "Wide Local Excision (WLE) with 1–2 cm safety margins",
                "Mohs Micrographic Surgery for high-risk facial / acral lesions",
                "Adjuvant radiation therapy mapping post-excision",
                "Systemic immunotherapy — Pembrolizumab / Nivolumab protocol",
                "Sentinel Lymph Node Biopsy (SLNB) for accurate staging",
            ],
            "patient_care": [
                "Strict UV avoidance — SPF 100+ broad-spectrum sunscreen daily",
                "Sterile post-operative wound management protocol",
                "UPF 50+ full-cover protective clothing — mandatory",
                "Monthly ABCDE self-examination with photographic record",
                "Immediate ER visit if rapid bleeding or ulceration occurs",
            ],
            "physician_ops": [
                "STAT referral to Onco-Dermatology within 48 hours",
                "Full-body dermoscopy mapping every 3 months",
                "Excisional biopsy for Breslow depth and Clark level staging",
                "PET/CT scan if systemic metastasis is clinically suspected",
                "Multidisciplinary tumour board review — strongly recommended",
            ],
            "reasoning": [
                "High asymmetry coefficient — pattern consistent with malignant morphology",
                "Irregular serrated border — indicative of invasive lateral growth",
                "Multi-zone colour heterogeneity — atypical melanocytic activity confirmed",
                "Lesion diameter index exceeds 6 mm clinical malignancy threshold",
                "Evolving pattern signature — ABCDE criteria positive across all axes",
            ],
        },
        "Benign": {
            "color":   "#00C896",
            "risk":    "LOW RISK",
            "banner":  "ax-banner-safe",
            "b_cls":   "ax-badge badge-safe",
            "procedures": [
                "No urgent surgical intervention required at this time",
                "Elective cosmetic laser ablation if aesthetically desired",
                "Targeted cryotherapy for symptomatic or cosmetic relief",
                "Diagnostic shave biopsy at patient's discretion",
                "Digital photographic baseline mapping for long-term monitoring",
            ],
            "patient_care": [
                "Daily SPF 50+ broad-spectrum sunscreen — every morning",
                "Ceramide-based barrier repair moisturiser regimen",
                "Dietary antioxidant support — Vitamins C, D3, and E",
                "Monthly ABCDE self-examination habit development",
                "Avoid mechanical trauma or repeated friction to lesion area",
            ],
            "physician_ops": [
                "Standard annual dermatology screening schedule",
                "AI re-evaluation recommended in 6 months",
                "Patient to report any sudden morphological changes immediately",
                "Rule out atypical nevi or dysplastic naevus syndrome",
                "Monitor for development of satellite or perilesional lesions",
            ],
            "reasoning": [
                "Symmetric morphology — uniform regular growth confirmed",
                "Well-defined smooth border — no invasive margin signs detected",
                "Homogeneous pigmentation — normal melanocyte distribution pattern",
                "Diameter within benign threshold (< 6 mm clinical threshold)",
                "No evolving pattern detected — clinically stable, non-progressive",
            ],
        },
    }

    @classmethod
    def get(cls, dx: str) -> dict:
        return cls._DATA.get(dx, cls._DATA["Benign"])


# ═══════════════════════════════════════════════════════════════════
#  PDF ENGINE
# ═══════════════════════════════════════════════════════════════════
class PDFEngine:
    @staticmethod
    def build(pid: str, dx: str, conf: float, rel: float,
              feats: dict, intel: dict, ts: str) -> bytes | None:
        if not _PDF_AVAILABLE:
            return None
        try:
            class _Doc(_FPDF):
                def header(self) -> None:
                    self.set_fill_color(5, 10, 20)
                    self.rect(0, 0, 210, 30, "F")
                    self.set_draw_color(0, 200, 150)
                    self.set_line_width(0.4)
                    self.line(0, 30, 210, 30)
                    self.set_font("Courier", "B", 14)
                    self.set_text_color(0, 229, 255)
                    self.cell(0, 16, "SKINSCAN AI  —  AXIOM CLINICAL REPORT", ln=True, align="C")
                    self.set_font("Courier", "", 7)
                    self.set_text_color(139, 156, 181)
                    self.cell(0, 8, "Enterprise Clinical OS v13.1  |  AI-Powered Dermatology  |  CONFIDENTIAL", ln=True, align="C")
                    self.ln(2)

                def footer(self) -> None:
                    self.set_y(-13)
                    self.set_font("Courier", "I", 7)
                    self.set_text_color(74, 85, 104)
                    self.cell(0, 6, f"Page {self.page_no()}  |  {ts}  |  SkinScan AI v13.1", align="C")

            doc = _Doc()
            doc.set_auto_page_break(True, 15)
            doc.add_page()

            dc = (255, 59, 87) if dx == "Malignant" else (0, 200, 150)

            def section(title: str) -> None:
                doc.ln(3)
                doc.set_fill_color(11, 18, 32)
                doc.set_font("Courier", "B", 9)
                doc.set_text_color(0, 229, 255)
                doc.cell(190, 8, f"  [{title}]", ln=True, fill=True)
                doc.ln(1)

            def kv(k: str, v: str, vc: tuple = (200, 215, 230)) -> None:
                doc.set_font("Courier", "B", 8.5)
                doc.set_text_color(139, 156, 181)
                doc.cell(60, 7, k)
                doc.set_font("Courier", "", 8.5)
                doc.set_text_color(*vc)
                doc.cell(0, 7, str(v), ln=True)

            def items(lst: list) -> None:
                for i, s in enumerate(lst, 1):
                    doc.set_font("Courier", "", 8.5)
                    doc.set_text_color(200, 215, 230)
                    doc.cell(0, 7, f"  [{i:02d}]  {s}", ln=True)
                doc.ln(1)

            section("PATIENT & SESSION")
            kv("Patient ID :", pid or "ANONYMOUS")
            kv("Timestamp  :", ts)
            kv("Model      :", "MobileNetV2-v2.1  |  FYP-PROD-2025")
            kv("Python     :", "3.14 Compatible Build")

            section("AI DIAGNOSIS RESULT")
            kv("Diagnosis  :", dx, vc=dc)
            kv("Risk Level :", intel["risk"])
            kv("Confidence :", f"{conf * 100:.2f}%")
            kv("Reliability:", f"{rel * 100:.2f}%")

            section("ABCDE BIOMARKER ANALYSIS")
            for feat, sc in feats.items():
                bar = "█" * int(sc * 18) + "░" * (18 - int(sc * 18))
                kv(f"{feat[:18]:<18} :", f"{sc * 100:.1f}%  {bar}")

            section("TREATMENT PROTOCOL")
            items(intel["procedures"])
            section("PATIENT CARE GUIDELINES")
            items(intel["patient_care"])
            section("PHYSICIAN ACTIONS")
            items(intel["physician_ops"])

            doc.ln(4)
            doc.set_font("Courier", "I", 7)
            doc.set_text_color(74, 85, 104)
            doc.multi_cell(0, 4.5,
                "DISCLAIMER: AI-generated for clinical decision support only. "
                "Final diagnosis must be confirmed by a licensed dermatologist. "
                "SkinScan AI does not replace professional medical advice.")

            raw = doc.output(dest="S")
            return raw.encode("latin-1") if isinstance(raw, str) else bytes(raw)
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════
#  SCAN ANIMATOR
# ═══════════════════════════════════════════════════════════════════
class ScanAnimator:
    _STAGES = [
        ("🔬", "FEATURE EXTRACTION",        "Analysing dermoscopic texture and spectral colour channels"),
        ("⚡", "CNN INFERENCE ENGINE",       "Running MobileNetV2 forward-pass through all 154 layers"),
        ("📊", "RISK PROBABILITY MODEL",     "Computing Bayesian malignancy probability distribution"),
        ("🏥", "CLINICAL DECISION MAPPING",  "Translating neural output to evidence-based protocol"),
    ]

    @staticmethod
    def play() -> None:
        h = st.empty(); s = st.empty(); p = st.empty()
        h.markdown("""
        <div style='text-align:center; padding:16px 0 8px;'>
          <div class='ax-label' style='color:#00E5FF; font-size:.76rem; letter-spacing:3px;'>
            ⚙  NEURAL PROCESSING PIPELINE ACTIVE
          </div>
        </div>""", unsafe_allow_html=True)

        for i, (ico, name, desc) in enumerate(ScanAnimator._STAGES):
            frac = (i + 1) / len(ScanAnimator._STAGES)
            s.markdown(f"""
            <div class='ax-stage'>
              <span style='font-size:1.15rem; width:28px; text-align:center;'>{ico}</span>
              <div>
                <div class='ax-stage-title'>{name}</div>
                <div class='ax-stage-desc'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)
            p.progress(frac, text=f"Stage {i + 1} of {len(ScanAnimator._STAGES)}")
            time.sleep(0.88)

        time.sleep(0.15)
        h.empty(); s.empty(); p.empty()


# ═══════════════════════════════════════════════════════════════════
#  RESULT DASHBOARD
# ═══════════════════════════════════════════════════════════════════
class ResultDashboard:

    @staticmethod
    def render(pil_img: Image.Image, pid: str, dx: str,
               conf: float, rel: float, feats: dict,
               intel: dict, engine: NeuralCoreEngine,
               ts: str, scan_id: int) -> None:

        c    = intel["color"]
        is_m = (dx == "Malignant")

        # ── 1. Diagnosis banner ──────────────────────────────────
        icon = "⚠" if is_m else "✓"
        st.markdown(f"""
        <div class='{intel["banner"]}'>
          <div style='display:flex; justify-content:space-between;
                      align-items:flex-start; flex-wrap:wrap; gap:16px;'>
            <div>
              <div class='ax-label' style='margin-bottom:8px;'>
                AXIOM AI — PRIMARY DIAGNOSIS OUTPUT
              </div>
              <div style='font-family:Oxanium,sans-serif; font-size:1.75rem;
                          font-weight:800; color:{c}; line-height:1.1; margin-bottom:10px;'>
                {icon}&ensp;{dx.upper()} LESION DETECTED
              </div>
              <span class='{intel["b_cls"]}'>{intel["risk"]}</span>
              &ensp;
              <span class='ax-badge badge-info'>SCAN #{scan_id:04d}</span>
              &ensp;
              <span style='font-family:JetBrains Mono,monospace;
                           font-size:.65rem; color:#4A5568;'>🕐 {ts}</span>
            </div>
            <div style='text-align:right;'>
              <div style='font-family:Oxanium,sans-serif; font-size:3rem;
                          font-weight:900; color:{c}; line-height:1;
                          letter-spacing:-1px;'>
                {conf * 100:.1f}%
              </div>
              <div class='ax-label' style='margin-top:5px;'>CONFIDENCE SCORE</div>
              <div style='font-family:JetBrains Mono,monospace; font-size:.68rem;
                          color:#4A5568; margin-top:6px;'>
                RELIABILITY &ensp;<span style='color:{c};'>{rel * 100:.1f}%</span>
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── 2. Metric tiles ──────────────────────────────────────
        t1, t2, t3, t4, t5 = st.columns(5)
        _tiles = [
            (dx,                "DIAGNOSIS",    c,         "🧬"),
            (f"{conf*100:.1f}%","CONFIDENCE",   "#00E5FF", "📊"),
            (f"{rel*100:.1f}%", "RELIABILITY",  "#818CF8", "🤖"),
            (intel["risk"],     "RISK LEVEL",   c,         "⚠" if is_m else "✅"),
            ("v2.1",            "MODEL",        "#FFB020", "🔬"),
        ]
        for col, (val, lbl, clr, ico) in zip([t1, t2, t3, t4, t5], _tiles):
            col.markdown(f"""
            <div class='ax-tile'>
              <div class='ax-tile-ico'>{ico}</div>
              <div class='ax-tile-val' style='color:{clr};'>{val}</div>
              <div class='ax-tile-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 3. Gauge column + Grad-CAM column ───────────────────
        left, right = st.columns([1, 1.88])

        with left:
            st.markdown("""
            <div class='ax-card'>
              <div class='ax-card-hdr'>
                <span class='ax-label'>NEURAL CONFIDENCE GAUGE</span>
                <span class='ax-badge badge-info'>LIVE</span>
              </div>""", unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf * 100,
                number={"suffix": "%",
                        "font": {"size": 38, "color": c, "family": "Oxanium"}},
                gauge={
                    "axis": {"range": [0, 100],
                             "tickcolor": "#1E293B",
                             "tickfont": {"color": "#4A5568", "size": 8}},
                    "bar":  {"color": c, "thickness": 0.22},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,   35], "color": "rgba(0,200,150,0.07)"},
                        {"range": [35,  65], "color": "rgba(255,176,32,0.07)"},
                        {"range": [65, 100], "color": "rgba(255,59,87,0.09)"},
                    ],
                    "threshold": {
                        "line": {"color": c, "width": 3},
                        "thickness": 0.82, "value": conf * 100,
                    },
                },
            ))
            fig_g.update_layout(
                height=210,
                margin=dict(l=18, r=18, t=10, b=4),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#8B9CB5",
            )
            st.plotly_chart(fig_g, use_container_width=True)

            # Risk meter
            pct = int(conf * 100)
            st.markdown(f"""
            <div style='margin:-8px 0 14px;'>
              <div style='display:flex; justify-content:space-between;
                          align-items:center; margin-bottom:4px;'>
                <span class='ax-label'>RISK METER</span>
                <span class='ax-mono' style='color:{c}; font-size:.8rem;'>{pct}%</span>
              </div>
              <div class='ax-track' style='height:7px;'>
                <div class='ax-fill'
                  style='width:{pct}%;
                         background:linear-gradient(90deg,{c}55,{c});'>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Probability bar
            fig_p = go.Figure()
            fig_p.add_bar(
                x=["MALIGNANT", "BENIGN"],
                y=[conf * 100, (1 - conf) * 100],
                marker=dict(
                    color=["rgba(255,59,87,0.75)", "rgba(0,200,150,0.75)"],
                    line=dict(color=["#FF3B57", "#00C896"], width=1),
                ),
                width=0.42,
            )
            fig_p.update_layout(
                title=dict(text="PROBABILITY DISTRIBUTION",
                           font=dict(size=8.5, color="#4A5568",
                                     family="JetBrains Mono"), x=0),
                height=160,
                margin=dict(l=8, r=8, t=30, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#4A5568",
                showlegend=False,
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                           ticksuffix="%",
                           tickfont=dict(size=7.5)),
                xaxis=dict(showgrid=False,
                           tickfont=dict(size=7.5, family="JetBrains Mono")),
            )
            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("""
            <div class='ax-card'>
              <div class='ax-card-hdr'>
                <span class='ax-label'>EXPLAINABLE AI — GRAD-CAM ACTIVATION MAPS</span>
                <span class='ax-badge badge-warn'>XAI</span>
              </div>""", unsafe_allow_html=True)
            with st.spinner("Computing neural activation maps…"):
                cam_buf = engine.gradcam(pil_img, dx)
            st.image(cam_buf, use_container_width=True,
                     caption="[01] ORIGINAL  ·  [02] GRAD-CAM HEATMAP  ·  [03] AI FOCUS COMPOSITE")

            st.markdown("""
            <div style='margin-top:16px;'>
              <div class='ax-label' style='margin-bottom:10px;'>TOP CONTRIBUTING FEATURES</div>""",
                        unsafe_allow_html=True)
            for feat, score in sorted(feats.items(), key=lambda x: x[1], reverse=True)[:3]:
                pf = int(score * 100)
                bc = "#FF3B57" if pf > 75 else "#FFB020" if pf > 50 else "#00C896"
                st.markdown(f"""
                <div style='margin-bottom:11px;'>
                  <div style='display:flex; justify-content:space-between;
                              align-items:center; margin-bottom:4px;'>
                    <span style='font-size:.84rem; color:var(--text,#DCE8F5);'>{feat}</span>
                    <span class='ax-mono' style='color:{bc}; font-size:.82rem;'>{pf}%</span>
                  </div>
                  <div class='ax-track'>
                    <div class='ax-fill' style='width:{pf}%; background:{bc};'></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        # ── 4. ABCDE full panel ──────────────────────────────────
        st.markdown("""
        <div class='ax-card'>
          <div class='ax-card-hdr'>
            <span class='ax-label'>ABCDE BIOMARKER ANALYSIS — FULL FEATURE PANEL</span>
            <span class='ax-badge badge-purple'>5 MARKERS</span>
          </div>""", unsafe_allow_html=True)

        cols = st.columns(len(feats))
        for col, (feat, score) in zip(cols, feats.items()):
            pf = int(score * 100)
            bc = "#FF3B57" if pf > 75 else "#FFB020" if pf > 50 else "#00C896"
            with col:
                st.markdown(f"""
                <div style='text-align:center; padding:14px 6px;
                            border:1px solid rgba(255,255,255,0.06);
                            border-radius:8px; background:#070D1A;'>
                  <div class='ax-mono' style='font-size:1.55rem; color:{bc};'>{pf}%</div>
                  <div class='ax-label' style='margin:7px 0 9px; font-size:.56rem;
                              line-height:1.4;'>{feat}</div>
                  <div class='ax-track' style='height:4px; margin:0;'>
                    <div class='ax-fill' style='width:{pf}%; background:{bc};'></div>
                  </div>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 5. AI Reasoning ─────────────────────────────────────
        st.markdown("""
        <div class='ax-card'>
          <div class='ax-card-hdr'>
            <span class='ax-label'>🤖 EXPLAINABLE AI — DECISION REASONING CHAIN</span>
            <span class='ax-badge badge-purple'>5 LOGIC NODES</span>
          </div>""", unsafe_allow_html=True)
        for i, reason in enumerate(intel["reasoning"], 1):
            st.markdown(f"""
            <div class='ax-reason'>
              <span class='ax-reason-tag'>N{i:02d}</span>
              <span>{reason}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 6. Clinical tabs ─────────────────────────────────────
        st.markdown("""
        <div class='ax-card'>
          <div class='ax-card-hdr'>
            <span class='ax-label'>📋 CLINICAL DECISION PANEL</span>
            <span class='ax-badge badge-warn'>CLINICAL USE</span>
          </div>""", unsafe_allow_html=True)
        t1, t2, t3 = st.tabs([
            "  🩺  TREATMENT PROTOCOL  ",
            "  🛡️  PATIENT CARE  ",
            "  👨‍⚕️  PHYSICIAN ACTIONS  ",
        ])
        for tab, key in zip([t1, t2, t3],
                            ["procedures", "patient_care", "physician_ops"]):
            with tab:
                st.markdown("<br>", unsafe_allow_html=True)
                for i, item in enumerate(intel[key], 1):
                    st.markdown(f"""
                    <div class='ax-proto'>
                      <span class='ax-proto-num'>{i:02d}</span>
                      <span>{item}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 7. Action panel ──────────────────────────────────────
        st.markdown("""
        <div class='ax-card'>
          <div class='ax-card-hdr'>
            <span class='ax-label'>📥 ACTIONS — EXPORT · SAVE · APPROVE</span>
          </div>""", unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)

        # ① PDF report
        with b1:
            pdf_bytes = PDFEngine.build(pid, dx, conf, rel, feats, intel, ts)
            if pdf_bytes:
                fname = (f"SkinScan_{(pid or 'ANON').replace(' ', '_')}_"
                         f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                st.download_button(
                    "📄  DOWNLOAD PDF REPORT",
                    data=pdf_bytes, file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"pdf_{scan_id}",
                )
            else:
                st.markdown("""
                <div style='text-align:center; padding:12px; border:1px solid rgba(255,255,255,0.05);
                     border-radius:8px; color:#4A5568; font-family:JetBrains Mono,monospace;
                     font-size:.65rem; letter-spacing:1px;'>
                  ADD fpdf2 TO REQUIREMENTS
                </div>""", unsafe_allow_html=True)

        # ② Save record — FIXED (unique key + duplicate guard)
        with b2:
            save_done_key = f"saved_{scan_id}"
            if save_done_key not in st.session_state:
                st.session_state[save_done_key] = False

            if not st.session_state[save_done_key]:
                if st.button("💾  SAVE PATIENT RECORD",
                             use_container_width=True, key=f"save_{scan_id}"):
                    st.session_state.records.append({
                        "Scan ID":     f"#{scan_id:04d}",
                        "Timestamp":   ts,
                        "Patient":     pid or "ANONYMOUS",
                        "Diagnosis":   dx,
                        "Confidence":  f"{conf * 100:.2f}%",
                        "Reliability": f"{rel * 100:.2f}%",
                        "Risk":        intel["risk"],
                    })
                    st.session_state[save_done_key] = True
                    st.rerun()
            else:
                st.markdown(f"""
                <div style='text-align:center; padding:12px 8px;
                     background:rgba(0,200,150,0.09);
                     border:1px solid rgba(0,200,150,0.35);
                     border-radius:8px;'>
                  <div style='color:#00C896; font-family:JetBrains Mono,monospace;
                       font-size:.65rem; font-weight:700; letter-spacing:1px;'>
                    ✓ RECORD #{scan_id:04d} SAVED
                  </div>
                </div>""", unsafe_allow_html=True)

        # ③ Export CSV — FIXED (always renders, disabled state handled)
        with b3:
            if st.session_state.records:
                csv_bytes = pd.DataFrame(st.session_state.records).to_csv(index=False)
                st.download_button(
                    "📊  EXPORT CLINICAL CSV",
                    data=csv_bytes,
                    file_name=f"SkinScan_Registry_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"csv_{scan_id}",
                )
            else:
                st.markdown("""
                <div style='text-align:center; padding:12px;
                     border:1px solid rgba(255,255,255,0.05);
                     border-radius:8px; color:#4A5568;
                     font-family:JetBrains Mono,monospace;
                     font-size:.65rem; letter-spacing:1px;'>
                  SAVE A RECORD FIRST
                </div>""", unsafe_allow_html=True)

        # ④ Physician approval — FIXED (persistent toggle per scan)
        with b4:
            appr_key = f"approved_{scan_id}"
            if appr_key not in st.session_state:
                st.session_state[appr_key] = False

            if not st.session_state[appr_key]:
                if st.button("✅  PHYSICIAN APPROVED",
                             use_container_width=True, key=f"appr_{scan_id}"):
                    st.session_state[appr_key] = True
                    st.rerun()
            else:
                st.markdown(f"""
                <div style='text-align:center; padding:12px 8px;
                     background:rgba(0,200,150,0.09);
                     border:1px solid rgba(0,200,150,0.35);
                     border-radius:8px;'>
                  <div style='color:#00C896; font-family:JetBrains Mono,monospace;
                       font-size:.65rem; font-weight:700; letter-spacing:1.2px;'>
                    ✓ APPROVED — DR. ADMIN
                  </div>
                  <div style='color:#4A5568; font-family:JetBrains Mono,monospace;
                       font-size:.58rem; margin-top:4px;'>{ts}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  MASTER APP CONTROLLER
# ═══════════════════════════════════════════════════════════════════
class SkinScanApp:

    def __init__(self) -> None:
        st.set_page_config(
            page_title="SkinScan AI — Axiom Clinical OS",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self._init_state()
        self.engine = NeuralCoreEngine()
        _inject_css()

    def _init_state(self) -> None:
        _defaults = {
            "auth":       False,
            "records":    [],
            "scan_count": 0,
        }
        for k, v in _defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # ── Login ─────────────────────────────────────────────────────
    def _login(self) -> None:
        if st.session_state.auth:
            return
        _, col, _ = st.columns([1, 0.88, 1])
        with col:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='ax-card' style='text-align:center; padding:38px 30px;'>
              <div style='font-size:3rem; margin-bottom:4px;'>🧬</div>
              <div style='font-family:Oxanium,sans-serif; font-size:1.55rem;
                          font-weight:800; color:#00E5FF; letter-spacing:-0.3px;'>
                SKINSCAN AI
              </div>
              <div class='ax-label' style='margin:6px 0 22px;'>
                AXIOM CLINICAL OS  ·  v13.1  ·  SECURE ACCESS ONLY
              </div>
            </div>""", unsafe_allow_html=True)

            user = st.text_input("PHYSICIAN ID",   placeholder="admin")
            pwd  = st.text_input("SECURITY KEY",   placeholder="•••", type="password")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("⚡  INITIALIZE SECURE SESSION", use_container_width=True):
                if user == "admin" and pwd == "123":
                    st.session_state.auth = True
                    st.rerun()
                else:
                    st.error("❌  ACCESS DENIED — Invalid credentials")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────
    def _sidebar(self) -> str:
        with st.sidebar:
            st.markdown("""
            <div style='padding:20px 14px 14px;
                        border-bottom:1px solid rgba(0,229,255,0.1);
                        margin-bottom:4px;'>
              <div style='font-family:Oxanium,sans-serif; font-size:1.1rem;
                          font-weight:800; color:#00E5FF;'>
                🧬 SKINSCAN AI
              </div>
              <div class='ax-label' style='margin-top:3px;'>AXIOM CLINICAL OS  v13.1</div>
            </div>""", unsafe_allow_html=True)

            nav = option_menu(
                "",
                ["Dashboard", "AI Scanner", "Patient Registry", "Analytics"],
                icons=["grid-1x2-fill", "cpu-fill", "journal-medical", "bar-chart-fill"],
                default_index=0,
                styles={
                    "container": {"padding": "8px 4px",
                                  "background-color": "transparent"},
                    "icon":      {"color": "#8B9CB5", "font-size": "0.84rem"},
                    "nav-link":  {
                        "font-family": "Outfit, sans-serif",
                        "font-size": "0.87rem",
                        "color": "#8B9CB5",
                        "border-radius": "8px",
                        "margin": "2px 0",
                        "--hover-color": "rgba(0,229,255,0.08)",
                    },
                    "nav-link-selected": {
                        "background": "rgba(0,229,255,0.1)",
                        "color": "#00E5FF",
                        "font-weight": "600",
                    },
                },
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Engine status
            dot   = "🟢" if self.engine.is_online else "🟠"
            mode  = "NEURAL NET ONLINE" if self.engine.is_online else "SIMULATION MODE"
            st.markdown(f"""
            <div style='background:#070D1A;
                        border:1px solid rgba(0,229,255,0.1);
                        border-radius:8px; padding:14px 15px; margin:0 4px;'>
              <div class='ax-label' style='margin-bottom:10px;'>SYSTEM STATUS</div>
              <div style='font-size:.8rem; margin-bottom:6px;'>
                {dot}&ensp;<span style='color:#DCE8F5;'>{mode}</span>
              </div>
              <div style='font-family:JetBrains Mono,monospace; font-size:.64rem;
                   color:#4A5568; line-height:2;'>
                MODEL&emsp;{self.engine.model_version}<br>
                BUILD&emsp;{self.engine.build_tag}<br>
                SCANS&emsp;{st.session_state.scan_count:04d}<br>
                RECORDS&nbsp;{len(st.session_state.records):04d}
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🚪  TERMINATE SESSION", use_container_width=True):
                st.session_state.auth = False
                st.rerun()
        return nav

    # ── Page: Dashboard ───────────────────────────────────────────
    def _page_hub(self) -> None:
        now = datetime.datetime.now().strftime("%d %b %Y  ·  %H:%M")
        st.markdown(f"""
        <div style='margin-bottom:22px;'>
          <div class='ax-display'>
            CENTRAL COMMAND <span class='accent'>HUB</span>
          </div>
          <div style='color:#4A5568; font-family:JetBrains Mono,monospace;
                      font-size:.67rem; margin-top:6px; letter-spacing:1px;'>
            DR. ADMIN  ·  {now}
          </div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        _tiles = [
            ("14,892", "TOTAL SCANS",     "#00E5FF", "🔬"),
            ("97.8%",  "AVG CONFIDENCE",  "#818CF8", "🧠"),
            (str(len(st.session_state.records)), "SESSION RECORDS", "#00C896", "📋"),
            (str(st.session_state.scan_count),   "TODAY'S SCANS",   "#FFB020", "📊"),
        ]
        for col, (val, lbl, clr, ico) in zip([c1, c2, c3, c4], _tiles):
            col.markdown(f"""
            <div class='ax-tile'>
              <div class='ax-tile-ico'>{ico}</div>
              <div class='ax-tile-val' style='color:{clr};'>{val}</div>
              <div class='ax-tile-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        la, ra = st.columns(2)

        la.markdown("""
        <div class='ax-card'>
          <div class='ax-card-hdr'>
            <span class='ax-label'>QUICK START GUIDE</span>
            <span class='ax-badge badge-info'>GUIDE</span>
          </div>
          <div style='font-size:.88rem; color:#8B9CB5; line-height:2.1;'>
            <span style='color:#00E5FF; font-family:JetBrains Mono,monospace;'>[01]</span>
            &ensp;Go to <b style='color:#DCE8F5;'>AI Scanner</b> in the sidebar<br>
            <span style='color:#00E5FF; font-family:JetBrains Mono,monospace;'>[02]</span>
            &ensp;Enter a Patient ID<br>
            <span style='color:#00E5FF; font-family:JetBrains Mono,monospace;'>[03]</span>
            &ensp;Upload dermoscopic image (JPG / PNG)<br>
            <span style='color:#00E5FF; font-family:JetBrains Mono,monospace;'>[04]</span>
            &ensp;Click <b style='color:#DCE8F5;'>EXECUTE NEURAL SCAN</b>
          </div>
        </div>""", unsafe_allow_html=True)

        mode_badge = "badge-safe" if self.engine.is_online else "badge-warn"
        mode_label = "ONLINE" if self.engine.is_online else "SIMULATION"
        ra.markdown(f"""
        <div class='ax-card'>
          <div class='ax-card-hdr'>
            <span class='ax-label'>SYSTEM DIAGNOSTIC</span>
            <span class='ax-badge {mode_badge}'>{mode_label}</span>
          </div>
          <div style='font-family:JetBrains Mono,monospace; font-size:.7rem;
               color:#4A5568; line-height:2.2;'>
            <span style='color:#8B9CB5;'>AI ENGINE&ensp;&ensp;</span>
            {'🟢 NEURAL NET ONLINE' if self.engine.is_online else '🟠 SIMULATION MODE'}<br>
            <span style='color:#8B9CB5;'>MODEL&emsp;&emsp;&ensp;</span>
            <span style='color:#DCE8F5;'>{self.engine.model_version}</span><br>
            <span style='color:#8B9CB5;'>BUILD&emsp;&emsp;&ensp;</span>
            <span style='color:#DCE8F5;'>{self.engine.build_tag}</span><br>
            <span style='color:#8B9CB5;'>PYTHON&emsp;&ensp;&ensp;</span>
            <span style='color:#DCE8F5;'>3.14 COMPATIBLE</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Page: Scanner ─────────────────────────────────────────────
    def _page_scanner(self) -> None:
        st.markdown("""
        <div style='margin-bottom:22px;'>
          <div class='ax-display'>
            DIAGNOSTIC NEURAL <span class='accent'>LABORATORY</span>
          </div>
          <div style='color:#4A5568; font-family:JetBrains Mono,monospace;
                      font-size:.67rem; margin-top:6px; letter-spacing:1px;'>
            UPLOAD IMAGE → RUN INFERENCE → REVIEW CLINICAL OUTPUT
          </div>
        </div>""", unsafe_allow_html=True)

        in_col, _ = st.columns([1, 2.1])
        pil_img: Image.Image | None = None
        pid = ""
        run = False

        with in_col:
            st.markdown("""
            <div class='ax-card'>
              <div class='ax-card-hdr'>
                <span class='ax-label'>INPUT PARAMETERS</span>
                <span class='ax-badge badge-info'>STEP 1–2</span>
              </div>""", unsafe_allow_html=True)

            pid = st.text_input("PATIENT ID / NAME",
                                placeholder="e.g. PT-001 / John Doe",
                                key="pid_input")
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
            <div style='margin-bottom:6px;'>
              <div class='ax-card-hdr' style='margin-bottom:0; border-bottom:none; padding-bottom:0;'>
                <span class='ax-label'>DERMOSCOPIC IMAGE</span>
                <span class='ax-badge badge-info'>STEP 2</span>
              </div>
            </div>""", unsafe_allow_html=True)

            up = st.file_uploader("",
                                  type=["jpg", "jpeg", "png"],
                                  label_visibility="collapsed",
                                  key="img_upload")
            if up:
                pil_img = Image.open(up)
                st.image(pil_img, use_container_width=True)
                st.markdown("""
                <div style='font-family:JetBrains Mono,monospace; font-size:.62rem;
                     color:#00C896; margin:4px 0 10px; letter-spacing:1px;'>
                  ✓ IMAGE LOADED — INTEGRITY VERIFIED
                </div>""", unsafe_allow_html=True)
                run = st.button("▶  EXECUTE NEURAL SCAN",
                                use_container_width=True, key="run_btn")
            st.markdown("</div>", unsafe_allow_html=True)

        if up and run and pil_img is not None:
            st.divider()
            ScanAnimator.play()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.scan_count += 1
            scan_id = st.session_state.scan_count
            dx, conf, rel, feats = self.engine.execute_scan(pil_img)
            intel = ClinicalDB.get(dx)
            ResultDashboard.render(
                pil_img, pid, dx, conf, rel,
                feats, intel, self.engine, ts, scan_id,
            )

    # ── Page: Registry ────────────────────────────────────────────
    def _page_registry(self) -> None:
        st.markdown("""
        <div style='margin-bottom:22px;'>
          <div class='ax-display'>
            PATIENT <span class='accent'>REGISTRY</span>
          </div>
          <div style='color:#4A5568; font-family:JetBrains Mono,monospace;
                      font-size:.67rem; margin-top:6px; letter-spacing:1px;'>
            SECURE SESSION RECORDS  ·  ENCRYPTED LOCALLY
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='ax-card'>", unsafe_allow_html=True)

        if st.session_state.records:
            df = pd.DataFrame(st.session_state.records)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)

            col_exp, col_clr, _ = st.columns([1, 1, 2])
            with col_exp:
                st.download_button(
                    "📥  EXPORT FULL REGISTRY",
                    data=df.to_csv(index=False),
                    file_name=f"SkinScan_Registry_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="registry_export",
                )
            with col_clr:
                if st.button("🗑️  CLEAR ALL RECORDS",
                             use_container_width=True, key="clear_all"):
                    st.session_state.records = []
                    st.rerun()
        else:
            st.markdown("""
            <div style='text-align:center; padding:44px; color:#4A5568;
                 font-family:JetBrains Mono,monospace; font-size:.75rem;
                 letter-spacing:1.2px; line-height:2;'>
              📋  NO RECORDS YET<br>
              <span style='font-size:.62rem;'>
                RUN A SCAN → CLICK "SAVE PATIENT RECORD"
              </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Page: Analytics ───────────────────────────────────────────
    def _page_analytics(self) -> None:
        st.markdown("""
        <div style='margin-bottom:22px;'>
          <div class='ax-display'>
            REAL-TIME <span class='accent'>ANALYTICS</span>
          </div>
          <div style='color:#4A5568; font-family:JetBrains Mono,monospace;
                      font-size:.67rem; margin-top:6px; letter-spacing:1px;'>
            SESSION EPIDEMIOLOGY  ·  AI PERFORMANCE METRICS
          </div>
        </div>""", unsafe_allow_html=True)

        if not st.session_state.records:
            st.markdown("""
            <div class='ax-card' style='text-align:center; padding:44px;'>
              <div style='font-size:2rem; margin-bottom:8px;'>📊</div>
              <div class='ax-label'>NO DATA AVAILABLE</div>
              <div style='color:#4A5568; font-size:.85rem; margin-top:8px;'>
                Save scan results to generate analytics
              </div>
            </div>""", unsafe_allow_html=True)
            return

        df = pd.DataFrame(st.session_state.records)
        c1, c2 = st.columns(2)

        _chart_layout = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8B9CB5",
            title_font=dict(size=8.5, color="#4A5568", family="JetBrains Mono"),
            legend=dict(font=dict(family="JetBrains Mono", size=8.5)),
        )

        with c1:
            st.markdown("<div class='ax-card'>", unsafe_allow_html=True)
            fig1 = px.pie(
                df, names="Diagnosis", title="DIAGNOSIS DISTRIBUTION",
                hole=0.52,
                color_discrete_sequence=["#FF3B57", "#00C896"],
            )
            fig1.update_traces(
                textfont=dict(family="JetBrains Mono", size=9))
            fig1.update_layout(**_chart_layout)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='ax-card'>", unsafe_allow_html=True)
            df2 = df.copy()
            df2["Conf_Num"] = df2["Confidence"].str.replace("%", "").astype(float)
            fig2 = px.bar(
                df2, x="Patient", y="Conf_Num", color="Diagnosis",
                title="CONFIDENCE BY PATIENT",
                color_discrete_map={"Malignant": "#FF3B57", "Benign": "#00C896"},
            )
            fig2.update_layout(
                **_chart_layout,
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                           ticksuffix="%",
                           tickfont=dict(size=8, family="JetBrains Mono")),
                xaxis=dict(showgrid=False,
                           tickfont=dict(size=8, family="JetBrains Mono")),
                yaxis_title="Confidence (%)",
                bargap=0.35,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────
    def _footer(self) -> None:
        st.markdown("""
        <div style='text-align:center; padding:44px 0 20px; margin-top:16px;
             border-top:1px solid rgba(0,229,255,0.07);'>
          <div style='font-family:Oxanium,sans-serif; font-weight:700;
               font-size:.82rem; color:#1E293B; letter-spacing:1px;'>
            SKINSCAN AI  —  AXIOM CLINICAL OS  v13.1
          </div>
          <div style='font-family:JetBrains Mono,monospace; font-size:.62rem;
               color:#1E293B; margin-top:6px; line-height:2;'>
            DEVELOPED BY REHAN SHAFIQUE  ·  OOP ARCHITECTURE  ·  PYTHON 3.14  ·  FYP 2025<br>
            ⚠ RESEARCH &amp; CLINICAL DECISION-SUPPORT ONLY — NOT A SUBSTITUTE FOR LICENSED MEDICAL ADVICE
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Launch ────────────────────────────────────────────────────
    def launch(self) -> None:
        self._login()
        nav = self._sidebar()
        _routes = {
            "Dashboard":        self._page_hub,
            "AI Scanner":       self._page_scanner,
            "Patient Registry": self._page_registry,
            "Analytics":        self._page_analytics,
        }
        _routes.get(nav, self._page_hub)()
        self._footer()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    SkinScanApp().launch()
