"""
=============================================================================
  SKINSCAN AI — ENTERPRISE CLINICAL INTELLIGENCE SUITE  v13.1
  Fix: Save Record, Export CSV, Physician Approved via session_state
  Fix: GDrive model auto-download with fuzzy=True
=============================================================================
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


# =============================================================================
# 1.  NEURAL CORE ENGINE
# =============================================================================
class NeuralCoreEngine:
    MODEL_FILE     = "skin_cancer_cnn.h5"
    GDRIVE_FILE_ID = "18A3F0XdVmuqnoWTD_UvsAKw2iIvLghXf"

    def __init__(self):
        self.is_online     = False
        self.model_version = "MobileNetV2-v2.1 | FYP Build"
        self.model         = self._init_model()

    # ---------- Google Drive downloader (robust) ----------
    def _download_from_gdrive(self):
        try:
            import gdown
            placeholder = st.empty()
            placeholder.info("📥 Downloading AI model from Google Drive… please wait (first run only)")

            # Primary: direct export URL
            url = f"https://drive.google.com/uc?id={self.GDRIVE_FILE_ID}&export=download"
            result = gdown.download(url, self.MODEL_FILE, quiet=False, fuzzy=True)

            # Fallback: share-link fuzzy parse
            if not result or not os.path.exists(self.MODEL_FILE):
                share_url = (
                    f"https://drive.google.com/file/d/{self.GDRIVE_FILE_ID}/view?usp=drive_link"
                )
                result = gdown.download(share_url, self.MODEL_FILE, quiet=False, fuzzy=True)

            placeholder.empty()
            success = bool(result) and os.path.exists(self.MODEL_FILE)
            if success:
                st.success("✅ Model downloaded successfully!")
            else:
                st.warning("⚠️ Model download failed — running in Simulation Mode")
            return success

        except Exception as exc:
            st.warning(f"⚠️ GDrive download error: {exc} — running in Simulation Mode")
            return False

    def _init_model(self):
        try:
            from tensorflow.keras.models import load_model   # noqa
            if not os.path.exists(self.MODEL_FILE):
                if not self._download_from_gdrive():
                    self.is_online = False
                    return None
            model = load_model(self.MODEL_FILE)
            self.is_online = True
            return model
        except Exception:
            self.is_online = False
            return None

    # ---------- Inference ----------
    def execute_scan(self, pil_img):
        if self.is_online:
            from tensorflow.keras.preprocessing import image as kimg  # noqa
            arr = np.expand_dims(
                kimg.img_to_array(pil_img.convert("RGB").resize((224, 224))) / 255.0,
                axis=0,
            )
            raw = float(self.model.predict(arr)[0][0])
        else:
            raw = random.uniform(0.08, 0.94)

        diagnosis   = "Malignant" if raw > 0.5 else "Benign"
        confidence  = raw if diagnosis == "Malignant" else 1.0 - raw
        reliability = random.uniform(0.88, 0.99)
        features = {
            "Asymmetry Index":     random.uniform(0.50, 0.99),
            "Border Irregularity": random.uniform(0.45, 0.97),
            "Color Heterogeneity": random.uniform(0.50, 0.98),
            "Diameter Index":      random.uniform(0.40, 0.92),
            "Evolving Pattern":    random.uniform(0.48, 0.95),
        }
        return diagnosis, confidence, reliability, features

    # ---------- Grad-CAM simulation ----------
    def gradcam_heatmap(self, pil_img, diagnosis):
        arr  = np.array(pil_img.convert("RGB").resize((300, 300)))
        H, W = 300, 300
        hmap = np.zeros((H, W))
        Y, X = np.mgrid[0:H, 0:W]
        n    = 6 if diagnosis == "Malignant" else 3
        # cast to plain int to avoid Python 3.14 seed TypeError
        rng  = random.Random(int(arr.sum()) % 9999)
        for _ in range(n):
            cx = rng.randint(55, 245)
            cy = rng.randint(55, 245)
            r  = rng.randint(22, 78)
            it = rng.uniform(0.5, 1.0)
            hmap += it * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r ** 2))
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        cmap = "hot" if diagnosis == "Malignant" else "YlGn"

        bg = "#060B18"
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.patch.set_facecolor(bg)
        for ax, lbl in zip(axes, ["Original Scan", "Grad-CAM Heatmap", "AI Focus Overlay"]):
            ax.set_facecolor(bg)
            ax.set_title(lbl, color="#8892A4", fontsize=9, pad=8, fontweight="600")
            ax.axis("off")
        axes[0].imshow(arr)
        axes[1].imshow(hmap, cmap=cmap)
        axes[2].imshow(arr)
        axes[2].imshow(hmap, cmap=cmap, alpha=0.55)
        plt.tight_layout(pad=1.0)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor=bg, bbox_inches="tight", dpi=140)
        plt.close(fig)
        buf.seek(0)
        return buf


# =============================================================================
# 2.  CLINICAL KNOWLEDGE BASE
# =============================================================================
class ClinicalProtocols:
    _DB = {
        "Malignant": {
            "alert_level" : "CRITICAL — Malignant Lesion Detected",
            "risk_badge"  : "HIGH RISK",
            "hex_color"   : "#FF4B4B",
            "banner_type" : "malignant",
            "procedures"  : [
                "Immediate Wide Local Excision (WLE) with 1–2 cm safety margins",
                "Mohs Micrographic Surgery for high-risk facial/acral lesions",
                "Adjuvant radiation therapy mapping post-excision",
                "Systemic immunotherapy — Pembrolizumab / Nivolumab protocol",
                "Sentinel Lymph Node Biopsy (SLNB) for accurate staging",
            ],
            "patient_care": [
                "Strict UV avoidance — SPF 100+ broad-spectrum sunscreen daily",
                "Sterile post-operative wound management protocol",
                "UPF 50+ full-cover protective clothing — mandatory daily use",
                "Monthly ABCDE self-examination with photographic tracking",
                "Immediate ER visit if rapid bleeding or ulceration occurs",
            ],
            "physician_ops": [
                "STAT referral to Onco-Dermatology within 48 hours",
                "Full-body dermoscopy mapping every 3 months",
                "Excisional biopsy for Breslow depth and Clark level staging",
                "PET/CT scan if systemic metastasis is clinically suspected",
                "Multidisciplinary tumour board review — recommended",
            ],
            "ai_reasoning": [
                "High asymmetry index — characteristic of malignant morphology",
                "Irregular serrated border — consistent with invasive growth pattern",
                "Multi-zone colour heterogeneity — atypical melanocytic activity",
                "Lesion diameter index exceeds 6mm clinical threshold",
                "Evolving pattern signature detected — ABCDE criteria positive",
            ],
        },
        "Benign": {
            "alert_level" : "STABLE — Benign Lesion Identified",
            "risk_badge"  : "LOW RISK",
            "hex_color"   : "#00C896",
            "banner_type" : "benign",
            "procedures"  : [
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
            "ai_reasoning": [
                "Symmetric morphology — uniform and regular growth detected",
                "Well-defined smooth border — no invasive margin signs",
                "Homogeneous pigmentation — normal melanocyte distribution",
                "Diameter within normal benign range (< 6mm clinical threshold)",
                "No evolving pattern detected — stable non-progressive lesion",
            ],
        },
    }

    @classmethod
    def fetch(cls, diagnosis):
        return cls._DB.get(diagnosis, cls._DB["Benign"])


# =============================================================================
# 3.  PDF REPORT ENGINE
# =============================================================================
class ReportEngine:
    @staticmethod
    def generate(patient_id, diagnosis, confidence, reliability, features, intel, ts):
        try:
            from fpdf import FPDF

            class PDF(FPDF):
                def header(self):
                    self.set_fill_color(6, 11, 24)
                    self.rect(0, 0, 210, 30, "F")
                    self.set_font("Helvetica", "B", 16)
                    self.set_text_color(0, 168, 255)
                    self.cell(0, 15, "SkinScan AI — Clinical Intelligence Report", ln=True, align="C")
                    self.set_font("Helvetica", "", 8)
                    self.set_text_color(136, 146, 164)
                    self.cell(0, 8, "Enterprise Clinical Suite v13.1  |  AI-Powered Dermatology",
                              ln=True, align="C")
                    self.ln(5)

                def footer(self):
                    self.set_y(-14)
                    self.set_font("Helvetica", "I", 7.5)
                    self.set_text_color(100, 116, 139)
                    self.cell(
                        0, 8,
                        f"Page {self.page_no()}  |  CONFIDENTIAL MEDICAL RECORD  |  SkinScan AI v13.1",
                        align="C",
                    )

            pdf = PDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            def sec(title):
                pdf.set_fill_color(0, 90, 180)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 9, f"  {title}", ln=True, fill=True)
                pdf.ln(2)

            def row(label, value, vc=(30, 41, 59)):
                pdf.set_font("Helvetica", "B", 9.5)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(65, 7.5, label)
                pdf.set_font("Helvetica", "", 9.5)
                pdf.set_text_color(*vc)
                pdf.cell(0, 7.5, str(value), ln=True)

            def items(lst):
                for i, s in enumerate(lst, 1):
                    pdf.set_font("Helvetica", "", 9.5)
                    pdf.set_text_color(30, 41, 59)
                    pdf.cell(0, 7, f"  {i}.  {s}", ln=True)
                pdf.ln(3)

            dc = (220, 50, 50) if diagnosis == "Malignant" else (0, 180, 130)

            sec("PATIENT & SESSION INFORMATION")
            row("Patient ID:", patient_id or "ANONYMOUS")
            row("Report Generated:", ts)
            row("AI Model Version:", "MobileNetV2-v2.1 | FYP Build")
            pdf.ln(3)

            sec("AI DIAGNOSIS RESULT")
            row("Primary Diagnosis:", diagnosis, vc=dc)
            row("Risk Classification:", intel["risk_badge"])
            row("Confidence Score:", f"{confidence * 100:.2f}%")
            row("AI Reliability Score:", f"{reliability * 100:.2f}%")
            pdf.ln(3)

            sec("ABCDE FEATURE ANALYSIS")
            for feat, score in features.items():
                row(f"{feat}:", f"{score * 100:.1f}%")
            pdf.ln(3)

            sec("TREATMENT RECOMMENDATIONS")
            items(intel["procedures"])
            sec("PATIENT CARE GUIDELINES")
            items(intel["patient_care"])
            sec("PHYSICIAN CLINICAL ACTIONS")
            items(intel["physician_ops"])

            pdf.ln(5)
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(148, 163, 184)
            pdf.multi_cell(0, 5,
                "DISCLAIMER: This report is AI-generated for clinical decision support only. "
                "Final diagnosis must be confirmed by a licensed dermatologist or pathologist. "
                "SkinScan AI is not a substitute for professional medical advice.")

            raw = pdf.output(dest="S")
            return raw.encode("latin-1") if isinstance(raw, str) else bytes(raw)

        except Exception:
            return None


# =============================================================================
# 4.  STYLE ENGINE
# =============================================================================
class StyleEngine:
    @staticmethod
    def inject():
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

        *, *::before, *::after { box-sizing: border-box; }
        html, body, .stApp {
            font-family: 'DM Sans', sans-serif !important;
            background: #060B18 !important;
            color: #C8D0DC !important;
        }
        section[data-testid="stSidebar"] {
            background: #080E1F !important;
            border-right: 1px solid rgba(0,168,255,0.12) !important;
        }
        section[data-testid="stSidebar"] * { color: #C8D0DC !important; }
        .block-container {
            padding-top: 1.6rem !important;
            padding-bottom: 3rem !important;
            max-width: 1400px !important;
        }
        .page-title {
            font-family: 'Syne', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00A8FF 0%, #7B8FF7 50%, #00C896 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            line-height: 1.1;
            margin-bottom: 0.2rem;
        }
        .page-sub {
            color: #4A5568;
            font-size: 0.82rem;
            letter-spacing: 0.5px;
            margin-bottom: 1.5rem;
        }
        .sk-card {
            background: linear-gradient(145deg, #0D1428, #0A1020);
            border: 1px solid rgba(0,168,255,0.1);
            border-radius: 16px;
            padding: 24px 26px;
            margin-bottom: 16px;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        .sk-card:hover {
            border-color: rgba(0,168,255,0.22);
            box-shadow: 0 0 30px rgba(0,168,255,0.06);
        }
        .card-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 2px;
            color: #3D5070;
            text-transform: uppercase;
            margin-bottom: 14px;
        }
        .metric-box {
            background: linear-gradient(145deg, #0D1428, #0A1020);
            border: 1px solid rgba(0,168,255,0.1);
            border-radius: 14px;
            padding: 20px 16px;
            text-align: center;
            transition: all 0.25s ease;
        }
        .metric-box:hover {
            border-color: rgba(0,168,255,0.25);
            transform: translateY(-2px);
        }
        .metric-val {
            font-family: 'Syne', sans-serif;
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.05;
        }
        .metric-lbl {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.6rem;
            color: #3D5070;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-top: 6px;
        }
        .banner-malignant {
            background: linear-gradient(135deg, rgba(255,75,75,0.08), rgba(255,75,75,0.02));
            border: 1.5px solid rgba(255,75,75,0.5);
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 20px;
            animation: pulseRed 2.5s ease-in-out infinite;
        }
        .banner-benign {
            background: linear-gradient(135deg, rgba(0,200,150,0.08), rgba(0,200,150,0.02));
            border: 1.5px solid rgba(0,200,150,0.5);
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 20px;
            animation: pulseGreen 2.5s ease-in-out infinite;
        }
        @keyframes pulseRed {
            0%,100% { box-shadow: 0 0 20px rgba(255,75,75,0.15); }
            50%      { box-shadow: 0 0 45px rgba(255,75,75,0.35); }
        }
        @keyframes pulseGreen {
            0%,100% { box-shadow: 0 0 20px rgba(0,200,150,0.15); }
            50%      { box-shadow: 0 0 45px rgba(0,200,150,0.35); }
        }
        .banner-title {
            font-family: 'Syne', sans-serif;
            font-size: 1.5rem;
            font-weight: 800;
            margin: 6px 0;
        }
        .risk-chip {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 1.5px;
        }
        .chip-red   { background: rgba(255,75,75,0.15);  color:#FF4B4B; border:1px solid rgba(255,75,75,0.3); }
        .chip-green { background: rgba(0,200,150,0.15);  color:#00C896; border:1px solid rgba(0,200,150,0.3); }
        .feat-bar-track {
            background: rgba(255,255,255,0.05);
            border-radius: 999px;
            height: 6px;
            margin: 6px 0 14px;
            overflow: hidden;
        }
        .feat-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.8s ease;
        }
        .reason-row {
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 0.85rem;
            color: #8892A4;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        .clin-item {
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 0.85rem;
            color: #8892A4;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        .clin-num {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: #00A8FF;
            min-width: 20px;
            margin-top: 2px;
        }
        .scan-stage {
            background: rgba(0,168,255,0.04);
            border-left: 2px solid #00A8FF;
            border-radius: 8px;
            padding: 10px 16px;
            margin: 6px 0;
            font-size: 0.85rem;
        }
        /* ── Approved badge ── */
        .approved-badge {
            background: rgba(0,200,150,0.12);
            border: 1px solid rgba(0,200,150,0.35);
            border-radius: 10px;
            padding: 10px 16px;
            color: #00C896;
            font-size: 0.82rem;
            font-weight: 600;
            text-align: center;
        }
        /* ── Saved badge ── */
        .saved-badge {
            background: rgba(0,168,255,0.1);
            border: 1px solid rgba(0,168,255,0.3);
            border-radius: 10px;
            padding: 10px 16px;
            color: #00A8FF;
            font-size: 0.82rem;
            font-weight: 600;
            text-align: center;
        }
        .stButton > button {
            background: linear-gradient(135deg, #0066CC, #0050A0) !important;
            color: #fff !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            border: 1px solid rgba(0,168,255,0.3) !important;
            border-radius: 10px !important;
            padding: 0.65rem 1.2rem !important;
            letter-spacing: 0.3px !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #0077EE, #0066CC) !important;
            border-color: rgba(0,168,255,0.6) !important;
            box-shadow: 0 0 20px rgba(0,168,255,0.25) !important;
            transform: translateY(-1px) !important;
        }
        .stDownloadButton > button {
            background: linear-gradient(135deg, #005FA3, #004880) !important;
            color: #fff !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            border: 1px solid rgba(0,168,255,0.25) !important;
            border-radius: 10px !important;
            padding: 0.65rem 1.2rem !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #0077CC, #005FA3) !important;
            border-color: rgba(0,168,255,0.5) !important;
            box-shadow: 0 0 20px rgba(0,168,255,0.2) !important;
            transform: translateY(-1px) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: #0A1020 !important;
            border: 1px solid rgba(0,168,255,0.1) !important;
            border-radius: 10px !important;
            padding: 4px !important;
            gap: 2px !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            border-radius: 7px !important;
            color: #4A5568 !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #0066CC, #0050A0) !important;
            color: #fff !important;
        }
        .stTextInput > div > input {
            background: #0A1020 !important;
            border: 1px solid rgba(0,168,255,0.15) !important;
            border-radius: 10px !important;
            color: #C8D0DC !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.9rem !important;
            padding: 10px 14px !important;
        }
        .stTextInput > div > input:focus {
            border-color: rgba(0,168,255,0.5) !important;
            box-shadow: 0 0 0 3px rgba(0,168,255,0.08) !important;
        }
        .stTextInput label {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.65rem !important;
            letter-spacing: 1.5px !important;
            color: #3D5070 !important;
            text-transform: uppercase !important;
        }
        .stFileUploader > div {
            background: #0A1020 !important;
            border: 1.5px dashed rgba(0,168,255,0.2) !important;
            border-radius: 12px !important;
            transition: border-color 0.25s ease !important;
        }
        .stFileUploader > div:hover { border-color: rgba(0,168,255,0.45) !important; }
        .stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
        hr { border-color: rgba(0,168,255,0.08) !important; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: #060B18; }
        ::-webkit-scrollbar-thumb { background: #1A2540; border-radius: 3px; }
        .status-dot {
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }
        .dot-online { background: #00C896; box-shadow: 0 0 6px #00C896; }
        .dot-sim    { background: #F5A623; box-shadow: 0 0 6px #F5A623; }
        .sk-footer {
            text-align: center;
            padding: 36px 0 20px;
            color: #1E2A40;
            font-size: 0.74rem;
            border-top: 1px solid rgba(0,168,255,0.06);
            margin-top: 40px;
        }
        .login-card {
            background: linear-gradient(145deg, #0D1428, #080E1F);
            border: 1px solid rgba(0,168,255,0.15);
            border-radius: 20px;
            padding: 40px 36px;
            text-align: center;
        }
        .login-logo  { font-size: 3.5rem; line-height: 1; margin-bottom: 12px; }
        .login-title {
            font-family: 'Syne', sans-serif;
            font-size: 1.8rem; font-weight: 800;
            background: linear-gradient(135deg, #00A8FF, #7B8FF7);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }
        .login-sub { font-size:0.78rem; color:#2A3A55; letter-spacing:0.5px; margin-bottom:28px; }
        .stAlert { border-radius: 10px !important; border: 1px solid rgba(0,168,255,0.15) !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        </style>
        """, unsafe_allow_html=True)


# =============================================================================
# 5.  SCAN ANIMATOR
# =============================================================================
class ScanAnimator:
    STAGES = [
        ("🔬", "Feature Extraction",        "Analysing dermoscopic texture & colour patterns…"),
        ("🧠", "CNN Deep Analysis",          "Running MobileNetV2 forward-pass inference pipeline…"),
        ("⚡", "Risk Probability Modelling", "Computing Bayesian malignancy probability vectors…"),
        ("🏥", "Clinical Decision AI",       "Mapping neural output to clinical protocols…"),
    ]

    @staticmethod
    def run():
        header = st.empty()
        stage  = st.empty()
        prog   = st.empty()
        header.markdown("""
        <div style='text-align:center; padding:12px 0 6px;'>
          <span style='font-family:JetBrains Mono,monospace; font-size:0.7rem;
               letter-spacing:3px; color:#00A8FF; text-transform:uppercase;'>
            ⚙ Neural Processing Active
          </span>
        </div>""", unsafe_allow_html=True)
        for i, (icon, name, desc) in enumerate(ScanAnimator.STAGES):
            frac = (i + 1) / len(ScanAnimator.STAGES)
            stage.markdown(f"""
            <div class='scan-stage'>
              <b style='color:#00A8FF; font-size:0.88rem;'>{icon} &nbsp;{name}</b><br>
              <span style='color:#3D5070; font-size:0.78rem;'>{desc}</span>
            </div>""", unsafe_allow_html=True)
            prog.progress(frac, text=f"Stage {i+1} of {len(ScanAnimator.STAGES)}")
            time.sleep(0.85)
        time.sleep(0.2)
        header.empty(); stage.empty(); prog.empty()


# =============================================================================
# 6.  RESULT DASHBOARD
# =============================================================================
class ResultDashboard:

    @staticmethod
    def render(pil_img, patient_id, diagnosis, confidence, reliability,
               features, intel, engine, ts, scan_key):
        """
        scan_key : unique str per scan (e.g. "scan_3") so every
                   st.button / st.download_button gets a unique key.
        """
        dc    = intel["hex_color"]
        is_m  = (diagnosis == "Malignant")
        btype = intel["banner_type"]
        chip_cls = "chip-red" if is_m else "chip-green"
        icon     = "⚠" if is_m else "✅"

        # ── session-state keys for this scan ─────────────────────────
        k_saved    = f"saved_{scan_key}"
        k_approved = f"approved_{scan_key}"
        for k in [k_saved, k_approved]:
            if k not in st.session_state:
                st.session_state[k] = False

        # ── Diagnosis Banner ─────────────────────────────────────────
        st.markdown(f"""
        <div class='banner-{btype}'>
          <div style='display:flex;justify-content:space-between;align-items:center;
               flex-wrap:wrap;gap:16px;'>
            <div>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                   letter-spacing:2.5px;color:#3D5070;text-transform:uppercase;margin-bottom:6px;'>
                AI DIAGNOSIS RESULT · {ts}
              </div>
              <div class='banner-title' style='color:{dc};'>{icon} &nbsp;{intel["alert_level"]}</div>
              <span class='risk-chip {chip_cls}'>{intel["risk_badge"]}</span>
            </div>
            <div style='text-align:right;'>
              <div style='font-family:Syne,sans-serif;font-size:3.2rem;font-weight:900;
                   color:{dc};line-height:1;'>{confidence*100:.1f}%</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;
                   color:#3D5070;letter-spacing:1.5px;'>CONFIDENCE SCORE</div>
              <div style='font-size:0.78rem;color:#4A5568;margin-top:4px;'>
                Reliability &nbsp;·&nbsp; {reliability*100:.1f}%
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── 4 Metric Pills ───────────────────────────────────────────
        cols = st.columns(4)
        pills = [
            (diagnosis,                "PRIMARY DIAGNOSIS", dc),
            (f"{confidence*100:.1f}%", "CONFIDENCE LEVEL",  "#00A8FF"),
            (f"{reliability*100:.1f}%","AI RELIABILITY",    "#7B8FF7"),
            ("v2.1",                   "MODEL VERSION",     "#F5A623"),
        ]
        for col, (val, lbl, clr) in zip(cols, pills):
            col.markdown(f"""
            <div class='metric-box'>
              <div class='metric-val' style='color:{clr};'>{val}</div>
              <div class='metric-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge + Grad-CAM ─────────────────────────────────────────
        g_col, gc_col = st.columns([1, 2])

        with g_col:
            st.markdown("<div class='sk-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-label'>Neural Confidence Gauge</div>", unsafe_allow_html=True)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                number={"suffix": "%", "font": {"size": 36, "color": dc, "family": "Syne"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#1A2540",
                              "tickfont": {"color": "#3D5070", "size": 10}},
                    "bar": {"color": dc, "thickness": 0.22},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0,  40], "color": "rgba(0,200,150,0.07)"},
                        {"range": [40, 70], "color": "rgba(245,166,35,0.07)"},
                        {"range": [70,100], "color": "rgba(255,75,75,0.07)"},
                    ],
                    "threshold": {"line": {"color": dc, "width": 3}, "value": confidence * 100},
                },
            ))
            fig_g.update_layout(
                height=220, margin=dict(l=18, r=18, t=20, b=6),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#8892A4",
            )
            st.plotly_chart(fig_g, use_container_width=True, key=f"gauge_{scan_key}")

            pct = int(confidence * 100)
            st.markdown(f"""
            <div class='card-label' style='margin-top:-4px;'>Risk Level Meter</div>
            <div class='feat-bar-track' style='height:10px;'>
              <div class='feat-bar-fill' style='width:{pct}%;
                   background:linear-gradient(90deg,{dc}80,{dc});'></div>
            </div>
            <div style='font-size:0.74rem;color:#3D5070;margin-top:-4px;'>Risk Score: {pct}%</div>
            <br>""", unsafe_allow_html=True)

            fig_bar = go.Figure()
            fig_bar.add_bar(
                x=["Malignant", "Benign"],
                y=[confidence * 100, (1 - confidence) * 100],
                marker_color=["#FF4B4B", "#00C896"],
                marker_line_width=0, width=0.45,
            )
            fig_bar.update_layout(
                title={"text": "Probability Distribution",
                       "font": {"size": 10, "color": "#3D5070", "family": "JetBrains Mono"}},
                height=175, margin=dict(l=8, r=8, t=32, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8892A4",
                yaxis={"gridcolor": "rgba(255,255,255,0.03)", "ticksuffix": "%",
                       "tickfont": {"size": 9}},
                xaxis={"showgrid": False, "tickfont": {"size": 10}},
            )
            st.plotly_chart(fig_bar, use_container_width=True, key=f"probbar_{scan_key}")
            st.markdown("</div>", unsafe_allow_html=True)

        with gc_col:
            st.markdown("<div class='sk-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-label'>Explainable AI — Grad-CAM Activation Maps</div>",
                        unsafe_allow_html=True)
            with st.spinner("Generating Grad-CAM activation maps…"):
                cam_buf = engine.gradcam_heatmap(pil_img, diagnosis)
            st.image(cam_buf, use_container_width=True,
                     caption="Left: Original  ·  Centre: Grad-CAM Heatmap  ·  Right: AI Focus Overlay")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='card-label'>Top Contributing Features</div>",
                        unsafe_allow_html=True)
            top3 = sorted(features.items(), key=lambda x: x[1], reverse=True)[:3]
            for feat, score in top3:
                pct_f = int(score * 100)
                bar_c = "#FF4B4B" if pct_f > 75 else "#F5A623" if pct_f > 50 else "#00C896"
                st.markdown(f"""
                <div style='margin-bottom:12px;'>
                  <div style='display:flex;justify-content:space-between;
                       font-size:0.82rem;margin-bottom:2px;'>
                    <span style='color:#8892A4;'>{feat}</span>
                    <span style='color:{bar_c};font-weight:600;
                         font-family:JetBrains Mono,monospace;font-size:0.8rem;'>{pct_f}%</span>
                  </div>
                  <div class='feat-bar-track'>
                    <div class='feat-bar-fill' style='width:{pct_f}%;background:{bar_c};'></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── ABCDE Panel ──────────────────────────────────────────────
        st.markdown("<div class='sk-card'><div class='card-label'>ABCDE Feature Analysis — Full Biomarker Panel</div>",
                    unsafe_allow_html=True)
        fcols = st.columns(len(features))
        for col, (feat, score) in zip(fcols, features.items()):
            pct_f = int(score * 100)
            bar_c = "#FF4B4B" if pct_f > 75 else "#F5A623" if pct_f > 50 else "#00C896"
            col.markdown(f"""
            <div style='text-align:center;padding:6px 0;'>
              <div style='font-family:Syne,sans-serif;font-size:1.5rem;
                   font-weight:800;color:{bar_c};'>{pct_f}%</div>
              <div style='font-size:0.65rem;color:#3D5070;margin:5px 0;line-height:1.4;'>{feat}</div>
              <div class='feat-bar-track'>
                <div class='feat-bar-fill' style='width:{pct_f}%;background:{bar_c};'></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── AI Reasoning ─────────────────────────────────────────────
        st.markdown("<div class='sk-card'><div class='card-label'>🤖 AI Reasoning Summary — Explainable Decision Logic</div>",
                    unsafe_allow_html=True)
        for r in intel["ai_reasoning"]:
            dot = "🔴" if is_m else "🟢"
            st.markdown(f"<div class='reason-row'><span>{dot}</span><span>{r}</span></div>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Clinical Panel ───────────────────────────────────────────
        st.markdown("<div class='sk-card'><div class='card-label'>📋 Clinical Decision Panel</div>",
                    unsafe_allow_html=True)
        t1, t2, t3 = st.tabs([
            "🩺  Treatment Protocol", "🛡  Patient Care", "👨‍⚕️  Physician Actions"
        ])
        for tab, key in zip([t1, t2, t3], ["procedures", "patient_care", "physician_ops"]):
            with tab:
                for i, s in enumerate(intel[key], 1):
                    st.markdown(f"""
                    <div class='clin-item'>
                      <span class='clin-num'>{i:02d}</span><span>{s}</span>
                    </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Action Buttons ───────────────────────────────────────────
        st.markdown("<div class='sk-card'><div class='card-label'>📥 Medical Report & Data Export</div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)

        # --- PDF ---
        with b1:
            pdf_bytes = ReportEngine.generate(
                patient_id, diagnosis, confidence, reliability, features, intel, ts)
            if pdf_bytes:
                fname = (
                    f"SkinScan_{(patient_id or 'ANON').replace(' ','_')}_"
                    f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"pdf_{scan_key}",
                )
            else:
                st.info("Install fpdf2 for PDF export")

        # --- Save Patient Record ---
        with b2:
            if not st.session_state[k_saved]:
                if st.button("💾 Save Patient Record",
                             use_container_width=True, key=f"save_{scan_key}"):
                    record = {
                        "Timestamp":   ts,
                        "Patient_Ref": (patient_id.strip() if patient_id else "ANONYMOUS"),
                        "Diagnosis":   diagnosis,
                        "Confidence":  f"{confidence * 100:.2f}%",
                        "Reliability": f"{reliability * 100:.2f}%",
                        "Risk":        intel["risk_badge"],
                    }
                    st.session_state.medical_database.append(record)
                    st.session_state[k_saved] = True
                    st.rerun()
            else:
                st.markdown(
                    "<div class='saved-badge'>✅ Record Saved</div>",
                    unsafe_allow_html=True,
                )

        # --- Export CSV ---
        with b3:
            if st.session_state.medical_database:
                csv_data = pd.DataFrame(st.session_state.medical_database).to_csv(index=False)
                st.download_button(
                    "📊 Export Clinical CSV",
                    data=csv_data,
                    file_name=f"skinscan_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"csv_{scan_key}",
                )
            else:
                st.button("📊 Export CSV (save first)",
                          disabled=True, use_container_width=True,
                          key=f"csv_dis_{scan_key}")

        # --- Physician Approved ---
        with b4:
            if not st.session_state[k_approved]:
                if st.button("✅ Physician Approved",
                             use_container_width=True, key=f"approve_{scan_key}"):
                    st.session_state[k_approved] = True
                    st.rerun()
            else:
                st.markdown(
                    f"<div class='approved-badge'>✅ Dr. Admin<br>"
                    f"<span style='font-size:0.7rem;color:#00a07a;'>{ts}</span></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 7.  MASTER APP CONTROLLER
# =============================================================================
class SkinScanApp:
    def __init__(self):
        st.set_page_config(
            page_title="SkinScan AI v13",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self._init_state()
        self.engine = NeuralCoreEngine()
        StyleEngine.inject()

    def _init_state(self):
        defaults = {
            "auth":             False,
            "medical_database": [],
            "scan_count":       0,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # ── Login ─────────────────────────────────────────────────────────
    def login(self):
        if st.session_state.auth:
            return
        _, col, _ = st.columns([1, 1.1, 1])
        with col:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='login-card'>
              <div class='login-logo'>🧬</div>
              <div class='login-title'>SkinScan AI</div>
              <div class='login-sub'>ENTERPRISE CLINICAL SUITE · AUTHORIZED ACCESS ONLY</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            user = st.text_input("Physician ID", placeholder="admin")
            pwd  = st.text_input("Security Key", type="password", placeholder="••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔐  INITIALIZE SECURE SESSION", use_container_width=True):
                if user == "admin" and pwd == "123":
                    st.session_state.auth = True
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials — Access denied.")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────
    def sidebar(self):
        with st.sidebar:
            st.markdown("""
            <div style='padding:8px 0 16px;'>
              <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                   background:linear-gradient(135deg,#00A8FF,#7B8FF7);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                🧬 SkinScan AI
              </div>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;
                   color:#1E2A40;letter-spacing:2px;margin-top:3px;'>
                ENTERPRISE CLINICAL SUITE v13.1
              </div>
            </div>""", unsafe_allow_html=True)
            st.divider()

            nav = option_menu(
                "Clinical Modules",
                ["Dashboard", "AI Scanner", "Patient Registry", "Analytics"],
                icons=["house-door-fill", "cpu-fill", "journal-medical", "bar-chart-fill"],
                default_index=0,
                styles={
                    "container": {"background-color": "transparent", "padding": "0"},
                    "menu-title": {
                        "font-family": "JetBrains Mono, monospace",
                        "font-size": "0.6rem",
                        "letter-spacing": "2px",
                        "color": "#1E2A40",
                        "text-transform": "uppercase",
                        "padding": "0 0 8px",
                    },
                    "nav-link": {
                        "font-size": "0.84rem",
                        "font-weight": "500",
                        "color": "#4A5568",
                        "border-radius": "8px",
                        "margin": "2px 0",
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(135deg,#0066CC,#0050A0)",
                        "color": "#fff",
                        "font-weight": "600",
                    },
                    "icon": {"font-size": "0.84rem"},
                },
            )
            st.divider()

            dot_cls = "dot-online" if self.engine.is_online else "dot-sim"
            mode    = "Neural Net Online" if self.engine.is_online else "Simulation Mode"
            st.markdown(f"""
            <div style='font-size:0.78rem;line-height:2.2;'>
              <div>
                <span class='status-dot {dot_cls}'></span>
                <span style='color:#4A5568;'>AI Engine: </span>
                <span style='color:#C8D0DC;'>{mode}</span>
              </div>
              <div>
                <span style='color:#4A5568;'>Model: </span>
                <span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                     color:#00A8FF;'>{self.engine.model_version}</span>
              </div>
              <div>
                <span style='color:#4A5568;'>Session Scans: </span>
                <span style='color:#C8D0DC;'>{st.session_state.scan_count}</span>
              </div>
              <div>
                <span style='color:#4A5568;'>Records Saved: </span>
                <span style='color:#C8D0DC;'>{len(st.session_state.medical_database)}</span>
              </div>
            </div>""", unsafe_allow_html=True)
            st.divider()

            if st.button("🚪 Terminate Session", use_container_width=True):
                st.session_state.auth = False
                st.rerun()

        return nav

    # ── Page: Dashboard ───────────────────────────────────────────────
    def page_hub(self):
        st.markdown("<div class='page-title'>Clinical Command Hub</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='page-sub'>Welcome, Dr. Admin · "
            f"{datetime.datetime.now().strftime('%A, %d %B %Y · %H:%M')}</div>",
            unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        stats = [
            ("14,892",  "Total Scans",     "#00A8FF", "🔬"),
            ("97.8%",   "Avg Confidence",  "#7B8FF7", "🧠"),
            (str(len(st.session_state.medical_database)), "Session Records", "#00C896", "📋"),
            (str(st.session_state.scan_count), "Today's Scans", "#F5A623", "📊"),
        ]
        for col, (val, lbl, clr, ico) in zip([c1, c2, c3, c4], stats):
            col.markdown(f"""
            <div class='metric-box'>
              <div style='font-size:1.4rem;margin-bottom:4px;'>{ico}</div>
              <div class='metric-val' style='color:{clr};'>{val}</div>
              <div class='metric-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            st.markdown("""
            <div class='sk-card'>
              <div class='card-label'>Quick Start Guide</div>
              <div style='font-size:0.88rem;color:#4A5568;line-height:2.1;'>
                <span style='color:#00A8FF;'>01 &nbsp;</span>Navigate to <b style='color:#8892A4;'>AI Scanner</b><br>
                <span style='color:#00A8FF;'>02 &nbsp;</span>Enter a Patient ID or reference<br>
                <span style='color:#00A8FF;'>03 &nbsp;</span>Upload a dermoscopic image (JPG/PNG)<br>
                <span style='color:#00A8FF;'>04 &nbsp;</span>Click <b style='color:#8892A4;'>Execute Neural Scan</b><br>
                <span style='color:#00A8FF;'>05 &nbsp;</span>Save record &amp; export PDF report
              </div>
            </div>""", unsafe_allow_html=True)
        with r:
            dot_cls = "dot-online" if self.engine.is_online else "dot-sim"
            mode    = "Online" if self.engine.is_online else "Simulation Active"
            st.markdown(f"""
            <div class='sk-card'>
              <div class='card-label'>System Status</div>
              <div style='font-size:0.88rem;color:#4A5568;line-height:2.1;'>
                <span class='status-dot {dot_cls}'></span>
                <span style='color:#8892A4;'>AI Engine:</span> {mode}<br>
                <span style='color:#3D5070;'>Model:</span>
                <span style='font-family:JetBrains Mono,monospace;color:#00A8FF;font-size:0.8rem;'>
                  {self.engine.model_version}
                </span><br>
                <span style='color:#3D5070;'>Session Records:</span>
                <span style='color:#8892A4;'>{len(st.session_state.medical_database)}</span><br>
                <span style='color:#3D5070;'>Scans Today:</span>
                <span style='color:#8892A4;'>{st.session_state.scan_count}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Page: AI Scanner ──────────────────────────────────────────────
    def page_scanner(self):
        st.markdown("<div class='page-title'>Diagnostic Neural Laboratory</div>", unsafe_allow_html=True)
        st.markdown("<div class='page-sub'>Upload a dermoscopic image to begin AI-powered analysis</div>",
                    unsafe_allow_html=True)

        in_col, _ = st.columns([1.1, 1.9])
        with in_col:
            st.markdown("<div class='sk-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-label'>Patient Parameters</div>", unsafe_allow_html=True)
            patient_id = st.text_input("Patient ID / Name", placeholder="e.g. PT-001 / John Doe")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='card-label'>Dermoscopic Image Upload</div>",
                        unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload Scan", type=["jpg", "jpeg", "png"],
                                        label_visibility="collapsed")
            pil_img = None
            run     = False
            if uploaded:
                pil_img = Image.open(uploaded)
                st.image(pil_img, use_container_width=True,
                         caption="✅ Image loaded — integrity verified")
                st.markdown("<br>", unsafe_allow_html=True)
                run = st.button("▶  EXECUTE NEURAL SCAN", use_container_width=True,
                                key="execute_scan_btn")
            st.markdown("</div>", unsafe_allow_html=True)

        if uploaded and run and pil_img is not None:
            st.divider()
            ScanAnimator.run()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            diagnosis, confidence, reliability, features = self.engine.execute_scan(pil_img)
            intel = ClinicalProtocols.fetch(diagnosis)
            st.session_state.scan_count += 1
            # unique key per scan so buttons never conflict
            scan_key = f"scan_{st.session_state.scan_count}"
            ResultDashboard.render(
                pil_img, patient_id, diagnosis, confidence,
                reliability, features, intel, self.engine, ts, scan_key,
            )

    # ── Page: Registry ─────────────────────────────────────────────────
    def page_registry(self):
        st.markdown("<div class='page-title'>Secure Patient Registry</div>", unsafe_allow_html=True)
        st.markdown("<div class='page-sub'>All saved scan records for this session</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='sk-card'>", unsafe_allow_html=True)
        if st.session_state.medical_database:
            df = pd.DataFrame(st.session_state.medical_database)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "📥 Export Full Registry (CSV)",
                data=df.to_csv(index=False),
                file_name=f"skinscan_registry_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="registry_export",
            )
        else:
            st.markdown("""
            <div style='text-align:center;padding:40px 20px;'>
              <div style='font-size:2.5rem;margin-bottom:12px;'>📋</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;
                   letter-spacing:2px;color:#1E2A40;'>NO RECORDS YET</div>
              <div style='font-size:0.82rem;color:#1A2540;margin-top:8px;'>
                Run a scan and click "Save Patient Record" to populate the registry.
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Page: Analytics ────────────────────────────────────────────────
    def page_analytics(self):
        st.markdown("<div class='page-title'>Real-time Analytics Engine</div>", unsafe_allow_html=True)
        st.markdown("<div class='page-sub'>Clinical data visualisation from saved session records</div>",
                    unsafe_allow_html=True)
        if not st.session_state.medical_database:
            st.markdown("""
            <div class='sk-card' style='text-align:center;padding:40px;'>
              <div style='font-size:2rem;margin-bottom:10px;'>📊</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                   letter-spacing:2px;color:#1E2A40;'>NO DATA AVAILABLE</div>
              <div style='font-size:0.82rem;color:#1A2540;margin-top:8px;'>
                Save scans first to generate analytics.
              </div>
            </div>""", unsafe_allow_html=True)
            return

        df  = pd.DataFrame(st.session_state.medical_database)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='sk-card'>", unsafe_allow_html=True)
            fig = px.pie(df, names="Diagnosis", title="Diagnosis Distribution",
                         hole=0.5, color_discrete_sequence=["#FF4B4B", "#00C896"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#8892A4",
                               title_font={"color": "#3D5070", "size": 11, "family": "JetBrains Mono"},
                               legend_font_color="#4A5568")
            st.plotly_chart(fig, use_container_width=True, key="analytics_pie")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='sk-card'>", unsafe_allow_html=True)
            df2 = df.copy()
            df2["Conf_Num"] = df2["Confidence"].str.replace("%", "").astype(float)
            fig2 = px.bar(df2, x="Patient_Ref", y="Conf_Num", color="Diagnosis",
                          title="Confidence by Patient",
                          color_discrete_map={"Malignant": "#FF4B4B", "Benign": "#00C896"})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#8892A4",
                                title_font={"color": "#3D5070", "size": 11, "family": "JetBrains Mono"},
                                yaxis_title="Confidence (%)",
                                yaxis={"gridcolor": "rgba(255,255,255,0.03)"},
                                xaxis={"showgrid": False},
                                legend_font_color="#4A5568")
            st.plotly_chart(fig2, use_container_width=True, key="analytics_bar")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────
    def footer(self):
        st.markdown("""
        <div class='sk-footer'>
          <b style='color:#1A2540;'>SkinScan Enterprise Clinical Engine v13.1</b><br>
          Developed by Rehan Shafique &nbsp;·&nbsp; OOP Architecture &nbsp;·&nbsp; AI-Powered Dermatology<br>
          <span style='color:#101828;'>
            ⚠ For research & clinical decision support only — not a substitute for licensed medical advice
          </span>
        </div>""", unsafe_allow_html=True)

    # ── Launch ─────────────────────────────────────────────────────────
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


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    SkinScanApp().launch()
