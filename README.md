# 🧠 NeuroAds — Brain Response Predictor for Digital Marketing

**Predict how your ad video, audio, image, or copy activates the human brain.**

Built on [Meta FAIR's TRIBE v2](https://github.com/facebookresearch/tribev2) — a multimodal fMRI encoding model that maps video + audio + language onto the cortical surface.

---

## 🚀 What It Does

Upload your ad content → TRIBE v2 predicts brain responses → Get marketing insights:

| Metric | Description |
|---|---|
| **Attention Score** | 0–100 scale of overall brain engagement per moment |
| **Hook Strength** | How the first 3 seconds hit the brain (critical for ads) |
| **Peak Moment** | Exact timestamp of maximum brain activation — place your CTA here |
| **ROI Breakdown** | Visual / Auditory / Language / Attention cortex activation |
| **Top 5 Hooks** | Best attention-grabbing segments from your content |

---

## 💻 Environment Requirements

- **Python 3.12** (Google Colab default — fully supported)
- **FFmpeg** (required for video/audio processing)
- **HuggingFace Token** (required to download `facebook/tribev2` weights — get one free at [huggingface.co](https://huggingface.co/settings/tokens))

---

## 📁 Project Structure

```
NeuroAds/
├── app.py                  ← Flask backend (prediction pipeline + REST API)
├── requirements.txt        ← Python 3.12-compatible dependencies
├── colab_run.ipynb         ← 🟡 One-click Google Colab notebook
├── setup.bat               ← Windows one-click setup script
├── model/                  ← Model weights downloaded here at first run (gitignored)
├── templates/
│   └── index.html          ← Premium Dark UI
├── static/
│   ├── css/style.css       ← Glassmorphism design
│   └── js/main.js          ← Dashboard & Charts logic
├── uploads/                ← Temp file storage (gitignored)
├── cache/                  ← Feature cache (gitignored)
└── results/                ← Analysis output (gitignored)
```

---

## ☁️ Run on Google Colab (Recommended)

**Easiest way — no setup needed:**

1. Open [`colab_run.ipynb`](colab_run.ipynb) in Google Colab
2. Run all cells — it will:
   - Install FFmpeg and all Python dependencies
   - Install TRIBE v2 from GitHub
   - Download model weights to `./model/` (once, ~2 GB)
   - Launch Flask with a public `ngrok` URL you can open in any browser
3. Paste your HuggingFace token when prompted (or leave blank if weights already downloaded)

---

## ⚡ Quick Start (Local — Windows)

### Automatic Setup

Run `setup.bat`. It will create a virtual environment, install requirements, and launch the app.

### Manual Setup

```bash
# Python 3.12 virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install requirements
pip install -r requirements.txt
pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"

# Run
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 🔑 Model Download

On the **first run**, the model weights (`best.ckpt`, ~2 GB) are automatically downloaded from HuggingFace into the `./model/` folder.  
**Subsequent runs load directly from disk — no re-download needed.**

You need a free HuggingFace account token for the initial download:
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a **Read** token
3. Enter it in the app's HuggingFace Token field (or set `HF_TOKEN` environment variable)

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyse` | `POST` | Start analysis — accepts `video`, `audio`, `text`, `image` files or `text` form field |
| `/api/status/<job_id>` | `GET` | Poll job status and retrieve final JSON results |

---

## 🎯 Digital Marketing Use Cases

- **Ad Video Analysis** — Find "dead moments" where attention drops
- **Hook Testing** — Compare different intros to see which lights up the attention cortex faster
- **Copy Analysis** — See how specific voiceover lines or text overlays impact the brain
- **CTA Placement** — Identify the exact moment of peak activation to place your Call to Action
- **Image Ad Testing** — Analyse static banners and creatives for brain impact

---

## ⚠️ Important Notes

- **Initial Download**: First prediction downloads ~2 GB of weights into `./model/` (one time only)
- **Hemodynamic Lag**: The model automatically corrects for the 5-second delay in brain response (BOLD signal)
- **GPU Support**: If a CUDA-enabled GPU is detected (e.g. Colab T4), the model uses it automatically for ~10× faster predictions
- **Python Version**: Fully compatible with Python 3.12 (Colab default)

---

## 📜 License

This project uses Meta's TRIBE v2, released under the **CC-BY-NC-4.0** license (Non-Commercial Research Use Only).
