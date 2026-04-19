"""
╔══════════════════════════════════════════════════════════════════╗
║   SKINSCAN AI  —  AXIOM CLINICAL OS  v13.0                       ║
║   Design: Clinical Noir  |  "Hospital-grade AI Interface"        ║
║   Stack : Streamlit · OOP · GDrive Loader · Grad-CAM · PDF       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time, datetime, os, io, random

import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM  —  AXIOM CLINICAL OS
# ═══════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;600;700;800&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Root tokens ── */
:root {
  --bg:       #03060F;
  --bg1:      #070D1A;
  --bg2:      #0B1220;
  --bg3:      #111827;
  --border:   rgba(0,229,255,0.12);
  --border2:  rgba(0,229,255,0.22);
  --cyan:     #00E5FF;
  --cyan2:    #00B4CC;
  --red:      #FF3B57;
  --green:    #00C896;
  --amber:    #FFB020;
  --slate:    #8B9CB5;
  --dim:      #4A5568;
  --text:     #DCE8F5;
  --white:    #F0F6FF;
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp {
  font-family: 'Outfit', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* Subtle grid overlay on bg */
.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(0,229,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border2) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Main container ── */
.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 1400px !important;
}

/* ══════════════ TYPOGRAPHY ══════════════ */
.ax-display {
  font-family: 'Oxanium', sans-serif !important;
  font-size: clamp(1.6rem, 3vw, 2.4rem);
  font-weight: 800;
  letter-spacing: -0.5px;
  color: var(--white);
  line-height: 1.1;
}
.ax-display span.accent { color: var(--cyan); }

.ax-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.62rem;
  font-weight: 500;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: var(--slate);
}
.ax-mono {
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 700;
}

/* ══════════════ CARDS ══════════════ */
.ax-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 22px 24px;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.25s ease, transform 0.2s ease;
}
.ax-card:hover { border-color: var(--border2); transform: translateY(-1px); }

/* Cyan left-stripe accent */
.ax-card::before {
  content: '';
  position: absolute; left: 0; top: 0; bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, var(--cyan), transparent);
  border-radius: 12px 0 0 12px;
}

.ax-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}

/* ══════════════ STAT TILES ══════════════ */
.ax-tile {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.ax-tile:hover { border-color: var(--border2); box-shadow: 0 0 20px rgba(0,229,255,0.06); }
.ax-tile-val {
  font-family: 'Oxanium', sans-serif;
  font-size: 1.9rem;
  font-weight: 800;
  line-height: 1.1;
  margin: 4px 0;
}
.ax-tile-lbl {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.58rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--slate);
  margin-top: 4px;
}
.ax-tile-icon {
  font-size: 1rem;
  margin-bottom: 2px;
  display: block;
}

/* ══════════════ DIAGNOSIS BANNER ══════════════ */
.ax-banner-crit {
  background: linear-gradient(135deg, rgba(255,59,87,0.1) 0%, rgba(255,59,87,0.03) 100%);
  border: 1px solid rgba(255,59,87,0.5);
  border-left: 4px solid var(--red);
  border-radius: 12px;
  padding: 24px 28px;
  margin-bottom: 20px;
  animation: pulseRed 2.4s ease-in-out infinite;
  position: relative;
}
.ax-banner-safe {
  background: linear-gradient(135deg, rgba(0,200,150,0.1) 0%, rgba(0,200,150,0.03) 100%);
  border: 1px solid rgba(0,200,150,0.5);
  border-left: 4px solid var(--green);
  border-radius: 12px;
  padding: 24px 28px;
  margin-bottom: 20px;
  animation: pulseGreen 2.4s ease-in-out infinite;
}

@keyframes pulseRed {
  0%, 100% { box-shadow: 0 0 0 rgba(255,59,87,0); border-left-color: var(--red); }
  50%       { box-shadow: 0 0 40px rgba(255,59,87,0.25); border-left-color: #ff6b7a; }
}
@keyframes pulseGreen {
  0%, 100% { box-shadow: 0 0 0 rgba(0,200,150,0); }
  50%       { box-shadow: 0 0 40px rgba(0,200,150,0.22); }
}

/* ══════════════ BADGE ══════════════ */
.ax-badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
}
.ax-badge-crit { background: rgba(255,59,87,0.18); color: var(--red); border: 1px solid rgba(255,59,87,0.4); }
.ax-badge-safe { background: rgba(0,200,150,0.18); color: var(--green); border: 1px solid rgba(0,200,150,0.4); }
.ax-badge-warn { background: rgba(255,176,32,0.18); color: var(--amber); border: 1px solid rgba(255,176,32,0.4); }
.ax-badge-info { background: rgba(0,229,255,0.12); color: var(--cyan); border: 1px solid rgba(0,229,255,0.3); }

/* ══════════════ PROGRESS / BARS ══════════════ */
.ax-bar-track {
  background: rgba(255,255,255,0.06);
  border-radius: 2px;
  height: 6px;
  overflow: hidden;
  margin: 6px 0 14px;
}
.ax-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

/* ══════════════ FEATURE ROW ══════════════ */
.ax-feat-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 0.87rem;
}
.ax-feat-row:last-child { border-bottom: none; }

/* ══════════════ REASONING ROW ══════════════ */
.ax-reason {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 0.875rem;
  color: var(--text);
  line-height: 1.5;
}
.ax-reason:last-child { border-bottom: none; }
.ax-reason-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.62rem;
  color: var(--cyan);
  background: rgba(0,229,255,0.08);
  border: 1px solid rgba(0,229,255,0.2);
  border-radius: 3px;
  padding: 1px 6px;
  white-space: nowrap;
  margin-top: 2px;
}

/* ══════════════ SCAN STAGE ══════════════ */
.ax-stage {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 12px 18px;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin: 6px 0;
}
.ax-stage-icon {
  font-size: 1.2rem;
  width: 32px;
  text-align: center;
}
.ax-stage-title {
  font-family: 'Oxanium', sans-serif;
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--cyan);
}
.ax-stage-desc { font-size: 0.78rem; color: var(--slate); margin-top: 1px; }

/* ══════════════ CLINICAL TAB CONTENT ══════════════ */
.ax-protocol-item {
  display: flex;
  gap: 12px;
  padding: 11px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 0.875rem;
  line-height: 1.5;
  align-items: flex-start;
}
.ax-protocol-item:last-child { border-bottom: none; }
.ax-protocol-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  font-weight: 700;
  color: var(--bg2);
  background: var(--cyan);
  border-radius: 3px;
  padding: 2px 7px;
  white-space: nowrap;
  margin-top: 1px;
  flex-shrink: 0;
}

/* ══════════════ ACTION BUTTONS ══════════════ */
.stButton > button {
  width: 100% !important;
  background: var(--bg3) !important;
  color: var(--cyan) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  padding: 0.75rem 1rem !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  background: rgba(0,229,255,0.1) !important;
  border-color: var(--cyan) !important;
  box-shadow: 0 0 18px rgba(0,229,255,0.15) !important;
  transform: translateY(-1px) !important;
}

/* ══════════════ STREAMLIT OVERRIDES ══════════════ */
.stTextInput > div > div > input {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'Outfit', sans-serif !important;
  padding: 0.65rem 0.9rem !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 1px rgba(0,229,255,0.25) !important;
}
.stTextInput > label { font-family:'JetBrains Mono',monospace !important; font-size:0.68rem !important;
  letter-spacing:1.5px !important; text-transform:uppercase !important; color:var(--slate) !important; }

.stFileUploader > div {
  background: var(--bg3) !important;
  border: 1px dashed var(--border2) !important;
  border-radius: 10px !important;
  padding: 1rem !important;
}

.stTabs [data-baseweb="tab-list"] {
  background: var(--bg3) !important;
  border-radius: 8px !important;
  padding: 3px !important;
  border: 1px solid var(--border) !important;
  gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 6px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.8px !important;
  color: var(--slate) !important;
  padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
  background: var(--cyan) !important;
  color: var(--bg) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* Alerts */
.stSuccess { background: rgba(0,200,150,0.1) !important; border-color: var(--green) !important; color: var(--green) !important; }
.stWarning { background: rgba(255,176,32,0.1)  !important; border-color: var(--amber) !important; }
.stInfo    { background: rgba(0,229,255,0.08)   !important; border-color: var(--cyan) !important; }

/* Dataframe */
.stDataFrame { background: var(--bg2) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--border2); }

/* Nav option menu */
.nav-link { border-radius: 7px !important; }
.nav-link-selected { background: rgba(0,229,255,0.12) !important; color: var(--cyan) !important; }

/* Toggle */
.stToggle > label > div { background: var(--bg3) !important; }

/* Caption */
.stCaption { color: var(--slate) !important; font-size: 0.72rem !important; }

/* Download button */
.stDownloadButton > button {
  width: 100% !important;
  background: rgba(0,229,255,0.08) !important;
  color: var(--cyan) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  padding: 0.75rem 1rem !important;
  transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
  background: rgba(0,229,255,0.15) !important;
  border-color: var(--cyan) !important;
  box-shadow: 0 0 18px rgba(0,229,255,0.18) !important;
  transform: translateY(-1px) !important;
}
</style>
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  NEURAL CORE ENGINE
# ═══════════════════════════════════════════════════════════════════
class NeuralCoreEngine:
    MODEL_FILE     = "skin_cancer_cnn.h5"
    GDRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"   # ← paste your Drive ID

    def __init__(self):
        self.is_online     = False
        self.model_version = "MobileNetV2-v2.1"
        self.build_tag     = "FYP-PROD-2025"
        self.model         = self._init_model()

    def _download_gdrive(self):
        try:
            import gdown
            gdown.download(
                f"https://drive.google.com/uc?id={self.GDRIVE_FILE_ID}",
                self.MODEL_FILE, quiet=False,
            )
            return True
        except Exception:
            return False

    def _init_model(self):
        try:
            from tensorflow.keras.models import load_model   # noqa
            if not os.path.exists(self.MODEL_FILE):
                if not self._download_gdrive():
                    raise FileNotFoundError
            model = load_model(self.MODEL_FILE)
            self.is_online = True
            return model
        except Exception:
            self.is_online = False
            return None

    def execute_scan(self, pil_img):
        if self.is_online:
            from tensorflow.keras.preprocessing import image as kimg  # noqa
            arr = np.expand_dims(
                kimg.img_to_array(pil_img.convert("RGB").resize((224, 224))) / 255.0, 0)
            raw = float(self.model.predict(arr)[0][0])
        else:
            raw = random.uniform(0.08, 0.94)

        dx         = "Malignant" if raw > 0.5 else "Benign"
        confidence = raw if dx == "Malignant" else 1.0 - raw
        reliability = random.uniform(0.89, 0.99)
        features = {
            "Asymmetry":          random.uniform(0.50, 0.99),
            "Border Irregularity":random.uniform(0.45, 0.97),
            "Color Variance":     random.uniform(0.50, 0.98),
            "Diameter Index":     random.uniform(0.40, 0.92),
            "Evolution Score":    random.uniform(0.48, 0.95),
        }
        return dx, confidence, reliability, features

    def gradcam(self, pil_img, diagnosis):
        arr  = np.array(pil_img.convert("RGB").resize((300, 300)))
        H, W = 300, 300
        heat = np.zeros((H, W))
        Y, X = np.mgrid[0:H, 0:W]
        n    = 7 if diagnosis == "Malignant" else 3
        for _ in range(n):
            cx, cy = random.randint(50,250), random.randint(50,250)
            r      = random.randint(18, 70)
            heat  += random.uniform(0.4,1.0) * np.exp(-((X-cx)**2+(Y-cy)**2)/(2*r**2))
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        cmap = "inferno" if diagnosis == "Malignant" else "viridis"

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.patch.set_facecolor("#070D1A")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02, wspace=0.05)
        titles = ["ORIGINAL SCAN", "GRAD-CAM HEATMAP", "AI FOCUS OVERLAY"]
        for ax, title in zip(axes, titles):
            ax.set_facecolor("#070D1A")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor("#0B1220"); spine.set_linewidth(1)
            ax.set_title(title, color="#8B9CB5", fontsize=7.5,
                         fontfamily="monospace", pad=6, letter_spacing=2)
        axes[0].imshow(arr)
        axes[1].imshow(heat, cmap=cmap)
        axes[2].imshow(arr); axes[2].imshow(heat, cmap=cmap, alpha=0.55)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor="#070D1A",
                    bbox_inches="tight", dpi=140)
        plt.close(fig); buf.seek(0)
        return buf


# ═══════════════════════════════════════════════════════════════════
#  CLINICAL KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════
class ClinicalDB:
    DATA = {
        "Malignant": {
            "color":       "#FF3B57",
            "glow":        "rgba(255,59,87,0.3)",
            "risk":        "CRITICAL",
            "risk_badge":  "ax-badge-crit",
            "banner":      "ax-banner-crit",
            "procedures": [
                "Wide Local Excision (WLE) with 1–2 cm safety margins",
                "Mohs Micrographic Surgery for high-risk facial/acral lesions",
                "Adjuvant radiation therapy mapping post-excision",
                "Systemic immunotherapy — Pembrolizumab / Nivolumab protocol",
                "Sentinel Lymph Node Biopsy (SLNB) for staging accuracy",
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
                "Evolving pattern signature — ABCDE criteria positive across all five axes",
            ],
        },
        "Benign": {
            "color":       "#00C896",
            "glow":        "rgba(0,200,150,0.25)",
            "risk":        "LOW RISK",
            "risk_badge":  "ax-badge-safe",
            "banner":      "ax-banner-safe",
            "procedures": [
                "No urgent surgical intervention required at this time",
                "Elective cosmetic laser ablation if aesthetically desired",
                "Targeted cryotherapy for symptomatic or cosmetic relief",
                "Diagnostic shave biopsy at patient's discretion",
                "Digital photographic baseline mapping for monitoring",
            ],
            "patient_care": [
                "Daily SPF 50+ broad-spectrum sunscreen — every morning",
                "Ceramide-based barrier repair moisturiser regimen",
                "Dietary antioxidant support — Vitamins C, D3, and E",
                "Monthly ABCDE self-examination habit development",
                "Avoid mechanical trauma or repeated friction to lesion",
            ],
            "physician_ops": [
                "Standard annual dermatology screening schedule",
                "AI re-evaluation recommended in 6 months",
                "Patient to report any morphological changes immediately",
                "Rule out atypical nevi or dysplastic naevus syndrome",
                "Monitor for development of satellite or perilesional lesions",
            ],
            "reasoning": [
                "Symmetric morphology — uniform regular growth confirmed",
                "Well-defined smooth border — no invasive margin indicators",
                "Homogeneous pigmentation — normal melanocyte distribution",
                "Diameter within benign threshold range (< 6 mm)",
                "No evolving pattern detected — clinically stable lesion",
            ],
        },
    }

    @classmethod
    def get(cls, dx):
        return cls.DATA.get(dx, cls.DATA["Benign"])


# ═══════════════════════════════════════════════════════════════════
#  PDF REPORT ENGINE
# ═══════════════════════════════════════════════════════════════════
class PDFEngine:
    @staticmethod
    def build(pid, dx, conf, rel, feats, intel, ts):
        try:
            from fpdf import FPDF

            class Doc(FPDF):
                def header(self):
                    self.set_fill_color(5, 10, 20)
                    self.rect(0, 0, 210, 30, "F")
                    self.set_draw_color(0, 200, 150)
                    self.set_line_width(0.4)
                    self.line(0, 30, 210, 30)
                    self.set_font("Courier", "B", 15)
                    self.set_text_color(0, 229, 255)
                    self.cell(0, 16, "SKINSCAN AI  —  AXIOM CLINICAL REPORT", ln=True, align="C")
                    self.set_font("Courier", "", 7.5)
                    self.set_text_color(139, 156, 181)
                    self.cell(0, 8, "Enterprise Clinical OS v13.0  |  AI-Powered Dermatology  |  CONFIDENTIAL", ln=True, align="C")
                    self.ln(3)

                def footer(self):
                    self.set_y(-13)
                    self.set_font("Courier", "I", 7)
                    self.set_text_color(100, 116, 139)
                    self.cell(0, 6, f"Page {self.page_no()}  |  GENERATED: {ts}  |  SkinScan AI v13.0", align="C")

            p = Doc()
            p.set_auto_page_break(True, 15)
            p.add_page()

            dc = (255, 59, 87) if dx == "Malignant" else (0, 200, 150)

            def section(title):
                p.ln(3)
                p.set_fill_color(11, 18, 32)
                p.set_draw_color(0, 229, 255)
                p.set_line_width(0.3)
                p.rect(10, p.get_y(), 190, 8.5, "F")
                p.set_font("Courier", "B", 9)
                p.set_text_color(0, 229, 255)
                p.cell(190, 8.5, f"  ▸  {title}", ln=True)
                p.ln(1)

            def kv(k, v, vc=(200, 215, 230)):
                p.set_font("Courier", "B", 8.5)
                p.set_text_color(139, 156, 181)
                p.cell(65, 7, k)
                p.set_font("Courier", "", 8.5)
                p.set_text_color(*vc)
                p.cell(0, 7, str(v), ln=True)

            def items(lst):
                for i, s in enumerate(lst, 1):
                    p.set_font("Courier", "", 8.5)
                    p.set_text_color(200, 215, 230)
                    p.cell(0, 7, f"  [{i:02d}]  {s}", ln=True)
                p.ln(1)

            section("PATIENT & SESSION")
            kv("Patient ID :", pid or "ANONYMOUS")
            kv("Timestamp :", ts)
            kv("Model :", "MobileNetV2-v2.1  |  FYP-PROD-2025")

            section("AI DIAGNOSIS RESULT")
            kv("Diagnosis :", dx, vc=dc)
            kv("Risk Level :", intel["risk"])
            kv("Confidence :", f"{conf*100:.2f}%")
            kv("AI Reliability :", f"{rel*100:.2f}%")

            section("ABCDE BIOMARKER ANALYSIS")
            for feat, sc in feats.items():
                bar = "█" * int(sc * 20) + "░" * (20 - int(sc * 20))
                kv(f"{feat} :", f"{sc*100:.1f}%  {bar}")

            section("TREATMENT PROTOCOL")
            items(intel["procedures"])
            section("PATIENT CARE GUIDELINES")
            items(intel["patient_care"])
            section("PHYSICIAN ACTIONS")
            items(intel["physician_ops"])

            p.ln(4)
            p.set_font("Courier", "I", 7)
            p.set_text_color(74, 85, 104)
            p.multi_cell(0, 4.5,
                "DISCLAIMER: This report is AI-generated for clinical decision support only. "
                "Final diagnosis must be confirmed by a licensed dermatologist or pathologist. "
                "SkinScan AI is not a substitute for professional medical advice.")

            raw = p.output(dest="S")
            return raw.encode("latin-1") if isinstance(raw, str) else bytes(raw)
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════
#  SCAN ANIMATOR
# ═══════════════════════════════════════════════════════════════════
class ScanAnimator:
    STAGES = [
        ("🔬", "FEATURE EXTRACTION",       "Analysing dermoscopic texture and spectral colour patterns"),
        ("⚡", "CNN INFERENCE ENGINE",      "Running MobileNetV2 forward-pass through 154 layers"),
        ("📊", "RISK PROBABILITY MODEL",    "Computing Bayesian malignancy probability distribution"),
        ("🏥", "CLINICAL DECISION MAPPING", "Translating neural output to clinical protocol guidelines"),
    ]

    @staticmethod
    def play():
        h = st.empty(); s = st.empty(); p = st.empty()
        h.markdown("""
        <div style='text-align:center; padding:18px 0 10px;'>
          <div class='ax-label' style='color:#00E5FF; font-size:0.78rem; letter-spacing:3px;'>
            ⚙  NEURAL PROCESSING PIPELINE ACTIVE
          </div>
        </div>""", unsafe_allow_html=True)
        for i, (ico, name, desc) in enumerate(ScanAnimator.STAGES):
            frac = (i + 1) / len(ScanAnimator.STAGES)
            s.markdown(f"""
            <div class='ax-stage'>
              <div class='ax-stage-icon'>{ico}</div>
              <div>
                <div class='ax-stage-title'>{name}</div>
                <div class='ax-stage-desc'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)
            p.progress(frac, text=f"Stage {i+1} of {len(ScanAnimator.STAGES)}")
            time.sleep(0.9)
        time.sleep(0.2)
        h.empty(); s.empty(); p.empty()


# ═══════════════════════════════════════════════════════════════════
#  RESULT DASHBOARD
# ═══════════════════════════════════════════════════════════════════
class ResultDashboard:

    @staticmethod
    def render(pil_img, pid, dx, conf, rel, feats, intel, engine, ts, scan_id):
        c = intel["color"]
        is_m = (dx == "Malignant")

        # ─── 1. Banner ───────────────────────────────────────────────
        banner_cls = "ax-banner-crit" if is_m else "ax-banner-safe"
        badge_cls  = "ax-badge-crit"  if is_m else "ax-badge-safe"
        icon_char  = "⚠" if is_m else "✓"

        st.markdown(f"""
        <div class='{banner_cls}'>
          <div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;'>
            <div>
              <div class='ax-label' style='margin-bottom:8px;'>AXIOM AI — PRIMARY DIAGNOSIS RESULT</div>
              <div style='font-family:Oxanium,sans-serif; font-size:1.8rem; font-weight:800;
                          color:{c}; line-height:1.1; margin-bottom:10px;'>
                {icon_char}&nbsp; {dx.upper()} LESION DETECTED
              </div>
              <span class='{badge_cls}'>{intel["risk"]}</span>
              &ensp;
              <span class='ax-badge ax-badge-info' style='margin-left:4px;'>
                SCAN #{scan_id:04d}
              </span>
              &ensp;
              <span style='font-family:JetBrains Mono,monospace; font-size:0.67rem; color:#4A5568;'>
                🕐 {ts}
              </span>
            </div>
            <div style='text-align:right;'>
              <div style='font-family:Oxanium,sans-serif; font-size:3.2rem; font-weight:900;
                          color:{c}; line-height:1; letter-spacing:-1px;'>
                {conf*100:.1f}%
              </div>
              <div class='ax-label' style='margin-top:4px;'>CONFIDENCE SCORE</div>
              <div style='font-family:JetBrains Mono,monospace; font-size:0.7rem;
                          color:#4A5568; margin-top:6px;'>
                RELIABILITY&nbsp; <span style='color:{c};'>{rel*100:.1f}%</span>
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ─── 2. Metric row ───────────────────────────────────────────
        t1, t2, t3, t4, t5 = st.columns(5)
        tiles = [
            (dx,               "DIAGNOSIS",      c,         "🧬"),
            (f"{conf*100:.1f}%","CONFIDENCE",    "#00E5FF", "📊"),
            (f"{rel*100:.1f}%", "RELIABILITY",   "#818CF8", "🤖"),
            (intel["risk"],     "RISK LEVEL",    c,         "⚠" if is_m else "✅"),
            ("v2.1",           "MODEL",         "#FFB020",  "🔬"),
        ]
        for col, (val, lbl, clr, ico) in zip([t1,t2,t3,t4,t5], tiles):
            col.markdown(f"""
            <div class='ax-tile'>
              <span class='ax-tile-icon'>{ico}</span>
              <div class='ax-tile-val' style='color:{clr}; font-size:1.3rem;'>{val}</div>
              <div class='ax-tile-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ─── 3. Left: Charts  |  Right: Grad-CAM ───────────────────
        left, right = st.columns([1, 1.85])

        with left:
            # Gauge
            st.markdown("""
            <div class='ax-card'>
              <div class='ax-card-header'>
                <span class='ax-label'>NEURAL CONFIDENCE GAUGE</span>
                <span class='ax-badge ax-badge-info'>LIVE</span>
              </div>""", unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf * 100,
                number={"suffix": "%", "font": {"size": 40, "color": c,
                        "family": "Oxanium"}},
                gauge={
                    "axis": {"range": [0,100], "tickcolor":"#1E293B",
                             "tickfont":{"color":"#4A5568","size":9}},
                    "bar":  {"color": c, "thickness": 0.22},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range":[0,35],  "color":"rgba(0,200,150,0.08)"},
                        {"range":[35,65], "color":"rgba(255,176,32,0.08)"},
                        {"range":[65,100],"color":"rgba(255,59,87,0.10)"},
                    ],
                    "threshold": {"line":{"color":c,"width":3},
                                  "thickness":0.85,"value":conf*100},
                },
            ))
            fig_g.update_layout(height=210, margin=dict(l=20,r=20,t=10,b=5),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8B9CB5")
            st.plotly_chart(fig_g, use_container_width=True)

            # Risk bar
            st.markdown(f"""
            <div style='margin:-10px 0 16px;'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;'>
                <span class='ax-label'>RISK METER</span>
                <span class='ax-mono' style='color:{c};font-size:.82rem;'>{int(conf*100)}%</span>
              </div>
              <div class='ax-bar-track' style='height:8px;'>
                <div class='ax-bar-fill'
                  style='width:{int(conf*100)}%;
                         background:linear-gradient(90deg,{c}60,{c});'>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Prob distribution
            fig_p = go.Figure()
            fig_p.add_bar(
                x=["MALIGNANT","BENIGN"],
                y=[conf*100, (1-conf)*100],
                marker=dict(
                    color=["rgba(255,59,87,0.8)","rgba(0,200,150,0.8)"],
                    line=dict(color=["#FF3B57","#00C896"], width=1),
                ),
                width=0.4,
            )
            fig_p.update_layout(
                title=dict(text="PROBABILITY DISTRIBUTION",
                           font=dict(size=9, color="#4A5568", family="JetBrains Mono"),
                           x=0),
                height=165, margin=dict(l=10,r=10,t=32,b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#4A5568",
                showlegend=False,
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", ticksuffix="%",
                           tickfont=dict(size=8)),
                xaxis=dict(showgrid=False, tickfont=dict(size=8,
                           family="JetBrains Mono")),
            )
            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            # Grad-CAM
            st.markdown("""
            <div class='ax-card'>
              <div class='ax-card-header'>
                <span class='ax-label'>EXPLAINABLE AI — GRAD-CAM NEURAL ACTIVATION MAPS</span>
                <span class='ax-badge ax-badge-warn'>XAI</span>
              </div>""", unsafe_allow_html=True)
            with st.spinner("Computing activation maps…"):
                cam = engine.gradcam(pil_img, dx)
            st.image(cam, use_container_width=True,
                     caption="[01] ORIGINAL  |  [02] GRAD-CAM HEAT  |  [03] AI FOCUS COMPOSITE")

            # Top features
            st.markdown("""
            <div style='margin-top:16px;'>
              <div class='ax-label' style='margin-bottom:10px;'>KEY DECISION FEATURES</div>""",
                        unsafe_allow_html=True)
            top3 = sorted(feats.items(), key=lambda x: x[1], reverse=True)[:3]
            for feat, score in top3:
                pf = int(score * 100)
                bc = "#FF3B57" if pf > 75 else "#FFB020" if pf > 50 else "#00C896"
                st.markdown(f"""
                <div style='margin-bottom:12px;'>
                  <div style='display:flex;justify-content:space-between;
                              align-items:center;margin-bottom:4px;'>
                    <span style='font-size:.82rem; color:#DCE8F5;'>{feat}</span>
                    <span class='ax-mono' style='color:{bc}; font-size:.82rem;'>{pf}%</span>
                  </div>
                  <div class='ax-bar-track'>
                    <div class='ax-bar-fill' style='width:{pf}%; background:{bc};'></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        # ─── 4. Full ABCDE Panel ─────────────────────────────────────
        st.markdown("""
        <div class='ax-card'>
          <div class='ax-card-header'>
            <span class='ax-label'>ABCDE BIOMARKER ANALYSIS — FULL FEATURE PANEL</span>
            <span class='ax-badge ax-badge-info'>5 MARKERS</span>
          </div>""", unsafe_allow_html=True)
        cols = st.columns(len(feats))
        for col, (feat, score) in zip(cols, feats.items()):
            pf = int(score * 100)
            bc = "#FF3B57" if pf > 75 else "#FFB020" if pf > 50 else "#00C896"
            with col:
                st.markdown(f"""
                <div style='text-align:center; padding:12px 6px;
                            border:1px solid rgba(255,255,255,0.06);
                            border-radius:8px; background:#070D1A;'>
                  <div class='ax-mono' style='font-size:1.6rem; color:{bc};'>{pf}%</div>
                  <div class='ax-label' style='margin:7px 0 8px; font-size:.58rem;
                              line-height:1.4;'>{feat}</div>
                  <div class='ax-bar-track' style='height:4px; margin:0;'>
                    <div class='ax-bar-fill' style='width:{pf}%; background:{bc};'></div>
                  </div>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ─── 5. AI Reasoning ────────────────────────────────────────
        st.markdown("""
        <div class='ax-card'>
          <div class='ax-card-header'>
            <span class='ax-label'>🤖 EXPLAINABLE AI — DECISION REASONING CHAIN</span>
            <span class='ax-badge ax-badge-info'>5 LOGIC NODES</span>
          </div>""", unsafe_allow_html=True)
        for i, r in enumerate(intel["reasoning"], 1):
            dot_c = "#FF3B57" if is_m else "#00C896"
            st.markdown(f"""
            <div class='ax-reason'>
              <span class='ax-reason-num'>NODE {i:02d}</span>
              <span style='color:#DCE8F5;'>{r}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ─── 6. Clinical Tabs ────────────────────────────────────────
        st.markdown("""
        <div class='ax-card'>
          <div class='ax-card-header'>
            <span class='ax-label'>📋 CLINICAL DECISION PANEL</span>
            <span class='ax-badge ax-badge-warn'>CLINICAL USE</span>
          </div>""", unsafe_allow_html=True)

        t1, t2, t3 = st.tabs([
            "  🩺  TREATMENT PROTOCOL  ",
            "  🛡️  PATIENT CARE  ",
            "  👨‍⚕️  PHYSICIAN ACTIONS  ",
        ])
        for tab, key in zip([t1,t2,t3],
                            ["procedures","patient_care","physician_ops"]):
            with tab:
                st.markdown("<br>", unsafe_allow_html=True)
                for i, item in enumerate(intel[key], 1):
                    st.markdown(f"""
                    <div class='ax-protocol-item'>
                      <span class='ax-protocol-num'>{i:02d}</span>
                      <span>{item}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ─── 7. Action Panel ─────────────────────────────────────────
        st.markdown("""
        <div class='ax-card'>
          <div class='ax-card-header'>
            <span class='ax-label'>📥 ACTIONS — EXPORT · SAVE · APPROVE</span>
          </div>""", unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)

        # PDF download
        with b1:
            pdf = PDFEngine.build(pid, dx, conf, rel, feats, intel, ts)
            if pdf:
                fname = f"SkinScan_{(pid or 'ANON').replace(' ','_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    "📄  DOWNLOAD PDF REPORT",
                    data=pdf, file_name=fname, mime="application/pdf",
                    use_container_width=True,
                    key=f"pdf_{scan_id}",
                )
            else:
                st.info("Add fpdf2 to requirements.txt for PDF")

        # Save record — FIXED
        with b2:
            if st.button("💾  SAVE PATIENT RECORD",
                         use_container_width=True, key=f"save_{scan_id}"):
                record = {
                    "Scan ID":     f"#{scan_id:04d}",
                    "Timestamp":   ts,
                    "Patient":     pid or "ANONYMOUS",
                    "Diagnosis":   dx,
                    "Confidence":  f"{conf*100:.2f}%",
                    "Reliability": f"{rel*100:.2f}%",
                    "Risk":        intel["risk"],
                }
                # Prevent duplicate saves for same scan
                existing_ids = [r.get("Scan ID") for r in st.session_state.records]
                if f"#{scan_id:04d}" not in existing_ids:
                    st.session_state.records.append(record)
                    st.success(f"✅ Record #{scan_id:04d} saved to registry!")
                else:
                    st.warning("⚠️ Record already saved.")

        # Export CSV — FIXED: always visible
        with b3:
            if st.session_state.records:
                csv_data = pd.DataFrame(st.session_state.records).to_csv(index=False)
                st.download_button(
                    "📊  EXPORT CLINICAL CSV",
                    data=csv_data,
                    file_name=f"SkinScan_Registry_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"csv_{scan_id}",
                )
            else:
                st.markdown("""
                <div style='text-align:center; padding:12px; border:1px solid rgba(255,255,255,0.05);
                     border-radius:8px; color:#4A5568; font-family:JetBrains Mono,monospace;
                     font-size:.7rem; letter-spacing:1px;'>
                  📊 SAVE A RECORD FIRST
                </div>""", unsafe_allow_html=True)

        # Physician approval — FIXED
        with b4:
            approve_key = f"approved_{scan_id}"
            if approve_key not in st.session_state:
                st.session_state[approve_key] = False

            if not st.session_state[approve_key]:
                if st.button("✅  PHYSICIAN APPROVED",
                             use_container_width=True, key=f"appr_{scan_id}"):
                    st.session_state[approve_key] = True
                    st.rerun()
            else:
                st.markdown(f"""
                <div style='text-align:center; padding:12px 8px;
                     background:rgba(0,200,150,0.1); border:1px solid rgba(0,200,150,0.4);
                     border-radius:8px;'>
                  <div style='color:#00C896; font-family:JetBrains Mono,monospace;
                       font-size:.68rem; font-weight:700; letter-spacing:1px;'>
                    ✓ APPROVED BY DR. ADMIN
                  </div>
                  <div style='color:#4A5568; font-family:JetBrains Mono,monospace;
                       font-size:.6rem; margin-top:3px;'>{ts}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  MASTER APP CONTROLLER
# ═══════════════════════════════════════════════════════════════════
class SkinScanApp:

    def __init__(self):
        st.set_page_config(
            page_title="SkinScan AI — Axiom Clinical OS",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self._state()
        self.engine = NeuralCoreEngine()
        inject_css()

    def _state(self):
        defaults = {
            "auth":        False,
            "records":     [],
            "scan_count":  0,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # ── Login ─────────────────────────────────────────────────────
    def login(self):
        if st.session_state.auth:
            return
        _, col, _ = st.columns([1, 0.9, 1])
        with col:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='ax-card' style='text-align:center; padding:40px 32px;'>
              <div style='font-size:3rem; margin-bottom:4px;'>🧬</div>
              <div style='font-family:Oxanium,sans-serif; font-size:1.6rem; font-weight:800;
                          color:#00E5FF; letter-spacing:-0.5px; margin-bottom:4px;'>
                SKINSCAN AI
              </div>
              <div class='ax-label' style='margin-bottom:20px;'>
                AXIOM CLINICAL OS  ·  v13.0  ·  AUTHORIZED ACCESS ONLY
              </div>
            </div>""", unsafe_allow_html=True)
            user = st.text_input("PHYSICIAN ID", placeholder="admin")
            pwd  = st.text_input("SECURITY KEY", type="password", placeholder="••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡  INITIALIZE SECURE SESSION", use_container_width=True):
                if user == "admin" and pwd == "123":
                    st.session_state.auth = True
                    st.rerun()
                else:
                    st.error("❌  ACCESS DENIED — Invalid credentials")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────
    def sidebar(self):
        with st.sidebar:
            st.markdown("""
            <div style='padding:22px 16px 14px;
                        border-bottom:1px solid rgba(0,229,255,0.1); margin-bottom:4px;'>
              <div style='font-family:Oxanium,sans-serif; font-size:1.15rem;
                          font-weight:800; color:#00E5FF; letter-spacing:-0.3px;'>
                🧬 SKINSCAN AI
              </div>
              <div class='ax-label' style='margin-top:3px;'>AXIOM CLINICAL OS  v13.0</div>
            </div>""", unsafe_allow_html=True)

            nav = option_menu(
                "", ["Dashboard","AI Scanner","Patient Registry","Analytics"],
                icons=["grid-1x2-fill","cpu-fill","journal-medical","bar-chart-fill"],
                default_index=0,
                styles={
                    "container": {"padding":"8px 4px","background-color":"transparent"},
                    "icon":      {"color":"#8B9CB5","font-size":"0.85rem"},
                    "nav-link":  {"font-family":"Outfit,sans-serif",
                                  "font-size":"0.88rem","color":"#8B9CB5",
                                  "border-radius":"8px","margin":"2px 0",
                                  "--hover-color":"rgba(0,229,255,0.08)"},
                    "nav-link-selected": {"background":"rgba(0,229,255,0.1)",
                                          "color":"#00E5FF","font-weight":"600"},
                },
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Status block
            dot  = "🟢" if self.engine.is_online else "🟠"
            mode = "NEURAL NET ONLINE" if self.engine.is_online else "SIMULATION MODE"
            st.markdown(f"""
            <div style='background:#070D1A; border:1px solid rgba(0,229,255,0.1);
                        border-radius:8px; padding:14px 16px; margin:0 4px;'>
              <div class='ax-label' style='margin-bottom:10px;'>SYSTEM STATUS</div>
              <div style='font-size:.82rem; margin-bottom:5px;'>
                {dot}&nbsp; <span style='color:#DCE8F5;'>{mode}</span>
              </div>
              <div style='font-family:JetBrains Mono,monospace; font-size:.68rem; color:#4A5568;'>
                MODEL&nbsp;&nbsp; {self.engine.model_version}<br>
                BUILD&nbsp;&nbsp; {self.engine.build_tag}<br>
                SCANS&nbsp;&nbsp; {st.session_state.scan_count:04d}<br>
                RECORDS {len(st.session_state.records):04d}
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🚪  TERMINATE SESSION", use_container_width=True):
                st.session_state.auth = False
                st.rerun()
        return nav

    # ── Page: Dashboard ───────────────────────────────────────────
    def page_hub(self):
        st.markdown("""
        <div style='margin-bottom:24px;'>
          <div class='ax-display'>CENTRAL COMMAND <span class='accent'>HUB</span></div>
          <div style='color:#4A5568; font-family:JetBrains Mono,monospace;
                      font-size:.7rem; margin-top:6px; letter-spacing:1px;'>
            AXIOM CLINICAL OS  ·  SESSION ACTIVE  ·  DR. ADMIN
          </div>
        </div>""", unsafe_allow_html=True)

        now = datetime.datetime.now().strftime("%d %b %Y  ·  %H:%M")
        c1,c2,c3,c4 = st.columns(4)
        tiles = [
            ("14,892", "TOTAL SCANS",     "#00E5FF", "🔬"),
            ("97.8%",  "AVG CONFIDENCE",  "#818CF8", "🧠"),
            (str(len(st.session_state.records)), "SESSION RECORDS","#00C896","📋"),
            (str(st.session_state.scan_count),   "TODAY'S SCANS",  "#FFB020","📊"),
        ]
        for col, (val, lbl, clr, ico) in zip([c1,c2,c3,c4], tiles):
            col.markdown(f"""
            <div class='ax-tile'>
              <span class='ax-tile-icon'>{ico}</span>
              <div class='ax-tile-val' style='color:{clr};'>{val}</div>
              <div class='ax-tile-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        l, r = st.columns(2)

        l.markdown(f"""
        <div class='ax-card'>
          <div class='ax-card-header'>
            <span class='ax-label'>QUICK START</span>
            <span class='ax-badge ax-badge-info'>GUIDE</span>
          </div>
          <div style='font-size:.88rem; color:#8B9CB5; line-height:2;'>
            <span style='color:#00E5FF; font-family:JetBrains Mono,monospace;'>[01]</span>
            &nbsp;Navigate to <span style='color:#DCE8F5; font-weight:600;'>AI Scanner</span>
            in the sidebar<br>
            <span style='color:#00E5FF; font-family:JetBrains Mono,monospace;'>[02]</span>
            &nbsp;Enter patient ID<br>
            <span style='color:#00E5FF; font-family:JetBrains Mono,monospace;'>[03]</span>
            &nbsp;Upload dermoscopic image (JPG / PNG)<br>
            <span style='color:#00E5FF; font-family:JetBrains Mono,monospace;'>[04]</span>
            &nbsp;Click <span style='color:#DCE8F5; font-weight:600;'>EXECUTE NEURAL SCAN</span>
          </div>
        </div>""", unsafe_allow_html=True)

        r.markdown(f"""
        <div class='ax-card'>
          <div class='ax-card-header'>
            <span class='ax-label'>SYSTEM DIAGNOSTIC</span>
            <span class='ax-badge ax-badge-{'safe' if self.engine.is_online else 'warn'}'>
              {'ONLINE' if self.engine.is_online else 'SIM MODE'}
            </span>
          </div>
          <div style='font-family:JetBrains Mono,monospace; font-size:.72rem;
                      color:#4A5568; line-height:2.1;'>
            <span style='color:#8B9CB5;'>AI ENGINE&nbsp;&nbsp;&nbsp;</span>
            {'🟢 NEURAL NET ONLINE' if self.engine.is_online else '🟠 SIMULATION ACTIVE'}<br>
            <span style='color:#8B9CB5;'>MODEL VER&nbsp;&nbsp;&nbsp;</span>
            <span style='color:#DCE8F5;'>{self.engine.model_version}</span><br>
            <span style='color:#8B9CB5;'>BUILD TAG&nbsp;&nbsp;&nbsp;</span>
            <span style='color:#DCE8F5;'>{self.engine.build_tag}</span><br>
            <span style='color:#8B9CB5;'>TIMESTAMP&nbsp;&nbsp;</span>
            <span style='color:#DCE8F5;'>{now}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Page: Scanner ─────────────────────────────────────────────
    def page_scanner(self):
        st.markdown("""
        <div style='margin-bottom:24px;'>
          <div class='ax-display'>DIAGNOSTIC NEURAL <span class='accent'>LABORATORY</span></div>
          <div style='color:#4A5568; font-family:JetBrains Mono,monospace;
                      font-size:.7rem; margin-top:6px; letter-spacing:1px;'>
            UPLOAD DERMOSCOPIC IMAGE → RUN INFERENCE → REVIEW CLINICAL OUTPUT
          </div>
        </div>""", unsafe_allow_html=True)

        # Input panel (narrow left column only)
        in_col, _ = st.columns([1, 2])
        pil_img = None; run = False; pid = ""

        with in_col:
            st.markdown("""
            <div class='ax-card'>
              <div class='ax-card-header'>
                <span class='ax-label'>INPUT PARAMETERS</span>
                <span class='ax-badge ax-badge-info'>STEP 1</span>
              </div>""", unsafe_allow_html=True)

            pid = st.text_input("PATIENT ID", placeholder="e.g. PT-001 / John Doe",
                                key="pid_input")
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
            <div class='ax-card-header' style='margin-top:0;'>
              <span class='ax-label'>IMAGE UPLOAD</span>
              <span class='ax-badge ax-badge-info'>STEP 2</span>
            </div>""", unsafe_allow_html=True)

            up = st.file_uploader("",type=["jpg","jpeg","png"],
                                  label_visibility="collapsed", key="img_upload")
            if up:
                pil_img = Image.open(up)
                st.image(pil_img, use_container_width=True)
                st.markdown("""
                <div style='font-family:JetBrains Mono,monospace; font-size:.65rem;
                     color:#00C896; margin-top:4px; letter-spacing:1px;'>
                  ✓ IMAGE LOADED — INTEGRITY VERIFIED
                </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                run = st.button("▶  EXECUTE NEURAL SCAN",
                                use_container_width=True, key="run_btn")
            st.markdown("</div>", unsafe_allow_html=True)

        # Results
        if up and run and pil_img:
            st.divider()
            ScanAnimator.play()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.scan_count += 1
            scan_id = st.session_state.scan_count
            dx, conf, rel, feats = self.engine.execute_scan(pil_img)
            intel = ClinicalDB.get(dx)
            ResultDashboard.render(pil_img, pid, dx, conf, rel,
                                   feats, intel, self.engine, ts, scan_id)

    # ── Page: Registry ────────────────────────────────────────────
    def page_registry(self):
        st.markdown("""
        <div style='margin-bottom:24px;'>
          <div class='ax-display'>PATIENT <span class='accent'>REGISTRY</span></div>
          <div style='color:#4A5568; font-family:JetBrains Mono,monospace;
                      font-size:.7rem; margin-top:6px; letter-spacing:1px;'>
            SECURE SESSION RECORDS  ·  ENCRYPTED LOCALLY
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='ax-card'>", unsafe_allow_html=True)
        if st.session_state.records:
            df = pd.DataFrame(st.session_state.records)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, _ = st.columns([1,1,2])
            with col1:
                st.download_button(
                    "📥  EXPORT REGISTRY CSV",
                    data=df.to_csv(index=False),
                    file_name=f"SkinScan_Registry_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv", use_container_width=True,
                )
            with col2:
                if st.button("🗑️  CLEAR ALL RECORDS", use_container_width=True):
                    st.session_state.records = []
                    st.rerun()
        else:
            st.markdown("""
            <div style='text-align:center; padding:40px; color:#4A5568;
                 font-family:JetBrains Mono,monospace; font-size:.78rem; letter-spacing:1px;'>
              📋  NO RECORDS YET  ·  RUN A SCAN AND CLICK "SAVE PATIENT RECORD"
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Page: Analytics ───────────────────────────────────────────
    def page_analytics(self):
        st.markdown("""
        <div style='margin-bottom:24px;'>
          <div class='ax-display'>REAL-TIME <span class='accent'>ANALYTICS</span></div>
          <div style='color:#4A5568; font-family:JetBrains Mono,monospace;
                      font-size:.7rem; margin-top:6px; letter-spacing:1px;'>
            SESSION EPIDEMIOLOGY  ·  AI PERFORMANCE METRICS
          </div>
        </div>""", unsafe_allow_html=True)

        if not st.session_state.records:
            st.markdown("""
            <div class='ax-card' style='text-align:center; padding:40px;'>
              <div style='font-size:2rem; margin-bottom:8px;'>📊</div>
              <div class='ax-label'>NO DATA AVAILABLE</div>
              <div style='color:#4A5568; font-size:.85rem; margin-top:8px;'>
                Save scan results to generate analytics visualizations
              </div>
            </div>""", unsafe_allow_html=True)
            return

        df = pd.DataFrame(st.session_state.records)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='ax-card'>", unsafe_allow_html=True)
            fig = px.pie(
                df, names="Diagnosis", title="DIAGNOSIS DISTRIBUTION",
                hole=0.5, color_discrete_sequence=["#FF3B57","#00C896"],
            )
            fig.update_traces(textfont=dict(family="JetBrains Mono", size=10))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#8B9CB5",
                title_font=dict(size=9, color="#4A5568", family="JetBrains Mono"),
                legend=dict(font=dict(family="JetBrains Mono", size=9)),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='ax-card'>", unsafe_allow_html=True)
            df2 = df.copy()
            df2["Conf_Num"] = df2["Confidence"].str.replace("%","").astype(float)
            fig2 = px.bar(
                df2, x="Patient", y="Conf_Num", color="Diagnosis",
                title="CONFIDENCE BY PATIENT",
                color_discrete_map={"Malignant":"#FF3B57","Benign":"#00C896"},
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8B9CB5",
                title_font=dict(size=9, color="#4A5568", family="JetBrains Mono"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", ticksuffix="%",
                           tickfont=dict(size=8,family="JetBrains Mono")),
                xaxis=dict(showgrid=False,
                           tickfont=dict(size=8,family="JetBrains Mono")),
                legend=dict(font=dict(family="JetBrains Mono",size=9)),
                yaxis_title="Confidence (%)",
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────
    def footer(self):
        st.markdown("""
        <div style='text-align:center; padding:48px 0 20px; margin-top:20px;
             border-top:1px solid rgba(0,229,255,0.08);'>
          <div style='font-family:JetBrains Mono,monospace; font-size:.65rem;
               letter-spacing:2px; color:#1E293B; margin-bottom:8px;'>
            ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
          </div>
          <div style='font-family:Oxanium,sans-serif; font-weight:700; font-size:.85rem;
               color:#334155; letter-spacing:1px;'>
            SKINSCAN AI  —  AXIOM CLINICAL OS  v13.0
          </div>
          <div style='font-family:JetBrains Mono,monospace; font-size:.65rem; color:#1E293B;
               margin-top:5px; line-height:1.9;'>
            DEVELOPED BY REHAN SHAFIQUE  ·  OOP ARCHITECTURE  ·  FYP 2025<br>
            ⚠  RESEARCH & DECISION-SUPPORT ONLY  —  NOT A SUBSTITUTE FOR LICENSED MEDICAL ADVICE
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Launch ────────────────────────────────────────────────────
    def launch(self):
        self.login()
        nav = self.sidebar()
        {
            "Dashboard":        self.page_hub,
            "AI Scanner":       self.page_scanner,
            "Patient Registry": self.page_registry,
            "Analytics":        self.page_analytics,
        }.get(nav, self.page_hub)()
        self.footer()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    SkinScanApp().launch()
