# SkinScan AI — Enterprise Clinical Suite v12.0

Professional skin lesion AI diagnostic system built with Streamlit + OOP architecture.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Login
- **Physician ID:** `admin`
- **Security Key:** `123`

## Features
| Module | Features |
|---|---|
| Command Hub | Live metrics, recent scans, system health |
| AI Diagnostic Lab | Image upload, scan animation, diagnosis card, gauge, probability bars, 5 protocol tabs, approval button |
| Medical Reports | Patient form, report preview, PDF/TXT download, CSV export |
| Patient Registry | Full scan history table, searchable, exportable |
| Analytics Engine | Donut chart, confidence bar chart, per-scan overview |

## With Real Model
Place `skin_cancer_cnn.h5` in the same folder — the engine auto-loads it.
Without it, simulation mode activates seamlessly (no crash).

## Architecture
- `NeuralCoreEngine` — model loading + inference with failsafe
- `ClinicalProtocols` — evidence-based medical knowledge base
- `SessionManager` — session state management
- All UI helpers — reusable card/badge/metric components
