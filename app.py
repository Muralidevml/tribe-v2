# -*- coding: utf-8 -*-
"""
NeuroAds - Brain Response Predictor for Digital Marketing
Uses Meta's TRIBE v2 model to predict fMRI brain responses to ad content.
Compatible with Python 3.12 (Google Colab default).
"""

import os
import sys
import json
import uuid
import traceback
import threading
import pathlib
from pathlib import Path
import shutil
import torch

# ── CUDA Memory Management & Device Detection ─────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] System Device: {DEVICE}")

# ── PosixPath fix for checkpoint loading on Windows ───────────────────────────
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# ── Windows exca resilience patch (skipped on Linux/Colab) ───────────────────
if sys.platform == "win32":
    try:
        import exca.cachedict.inflight

        def _patched_is_pid_alive(pid: int) -> bool:
            try:
                if pid <= 0:
                    return False
                os.kill(pid, 0)
                return True
            except (ProcessLookupError, OSError, SystemError):
                return False
            except PermissionError:
                return True

        exca.cachedict.inflight._is_pid_alive = _patched_is_pid_alive
    except ImportError:
        pass

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Monkey-patch TRIBE v2 for CPU Stability ───────────────────────────────────
try:
    from tribev2.eventstransforms import ExtractWordsFromAudio
    import subprocess, tempfile, json, torch
    import pandas as pd
    
    @staticmethod
    def _patched_get_transcript(wav_filename, language):
        language_codes = dict(english="en", french="fr", spanish="es", dutch="nl", chinese="zh")
        if language not in language_codes: raise ValueError(f"Language {language} not supported")

        # FIX: Ensure we use compatible compute types
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # We force int8 on CPU and even on CUDA if it fails, we want a safe mode
        compute_type = "float16" if device == "cuda" else "int8"

        with tempfile.TemporaryDirectory() as output_dir:
            cmd = [
                "uvx", "whisperx", str(wav_filename),
                "--model", "large-v3",
                "--language", language_codes[language],
                "--device", device,
                "--compute_type", compute_type,
                "--batch_size", "16",
                "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                "--output_dir", output_dir,
                "--output_format", "json",
            ]
            cmd = [c for c in cmd if c]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # If float16 failed on CUDA, try int8 as a last resort
                if "float16" in result.stderr and compute_type == "float16":
                    print("[WARN] float16 failed, retrying with int8...")
                    cmd[cmd.index("--compute_type") + 1] = "int8"
                    result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"whisperx failed:\n{result.stderr}")

            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript_data = json.loads(json_path.read_text())

        words = []
        for i, segment in enumerate(transcript_data["segments"]):
            for word in segment.get("words", []):
                if "start" in word:
                    words.append({
                        "text": word["word"].strip(),
                        "start": word["start"],
                        "duration": word["end"] - word["start"],
                        "sequence_id": i,
                        "sentence": segment["text"].strip()
                    })
        return pd.DataFrame(words)

    # Apply the patch
    ExtractWordsFromAudio._get_transcript_from_audio = _patched_get_transcript
    print("[INFO] Applied stability patch to ExtractWordsFromAudio")
except Exception as e:
    print(f"[WARN] Could not patch ExtractWordsFromAudio: {e}")

# ── Job state & Global Model ───────────────────────────────────────────────────
JOBS: dict = {}
MODEL = None
MODEL_LOCK = threading.Lock()

# ── Flask setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Folder layout ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()
UPLOAD_FOLDER  = BASE_DIR / "uploads"
RESULTS_FOLDER = BASE_DIR / "results"
MODEL_FOLDER   = BASE_DIR / "model"

# In Colab, /tmp is often more stable for the high-frequency cache writes
if os.path.exists("/content") and not os.path.exists("/content/drive"):
    CACHE_FOLDER = Path("/tmp/neuroads_cache")
else:
    CACHE_FOLDER = BASE_DIR / "cache"

for d in [UPLOAD_FOLDER, CACHE_FOLDER, RESULTS_FOLDER, MODEL_FOLDER]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[CRITICAL] Could not create folder {d}: {e}")

# ── Startup Diagnostics ────────────────────────────────────────────────────────
def _check_system():
    print(f"[INFO] Working Directory: {BASE_DIR}")
    # Check disk space
    total, used, free = shutil.disk_usage(BASE_DIR)
    free_gb = free // (2**30)
    print(f"[INFO] Disk Space: {free_gb} GB free")
    if free_gb < 5:
        print("[WARN] Low disk space! Need at least 5GB for model and processing.")
    
    # Check for Drive mount
    if "/content/drive" in str(BASE_DIR):
        print("[CRITICAL] Running from Google Drive detected. This WILL cause [Errno 5] I/O errors.")
        print("[CRITICAL] Please move the repo to /content/neuroads for stable performance.")

_check_system()

ALLOWED_VIDEO  = {"mp4", "mov", "avi", "mkv", "webm"}
ALLOWED_AUDIO  = {"mp3", "wav", "ogg", "flac", "m4a"}
ALLOWED_TEXT   = {"txt"}
ALLOWED_IMAGE  = {"png", "jpg", "jpeg", "webp"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def allowed_file(filename: str, kinds: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in kinds


def _ensure_dummy_video() -> Path:
    """Create a 5-second black video in MODEL_FOLDER if it doesn't exist."""
    dummy = MODEL_FOLDER / "dummy_5s.mp4"
    if not dummy.exists():
        try:
            from moviepy import ColorClip
            v_codec = "h264_nvenc" if DEVICE == "cuda" else "libx264"

            clip = ColorClip(size=(64, 64), color=(0, 0, 0), duration=5.0)
            clip.write_videofile(str(dummy), fps=1, codec=v_codec, audio=False, logger=None)
        except Exception as e:
            print(f"[WARN] Could not create dummy video: {e}")
    return dummy


def _load_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL

        from tribev2.demo_utils import TribeModel
        config_path = MODEL_FOLDER / "config.yaml"
        ckpt_path   = MODEL_FOLDER / "best.ckpt"

        if config_path.exists() and ckpt_path.exists():
            print("[INFO] Loading model from local ./model folder ...")
            checkpoint_dir = MODEL_FOLDER
        else:
            print("[INFO] Downloading model weights ...")
            from huggingface_hub import hf_hub_download
            hf_hub_download("facebook/tribev2", "config.yaml",  local_dir=str(MODEL_FOLDER))
            hf_hub_download("facebook/tribev2", "best.ckpt",    local_dir=str(MODEL_FOLDER))
            checkpoint_dir = MODEL_FOLDER

        device = DEVICE
        print(f"[INFO] Initializing model on {device} ...")
        
        if device == "cuda":
            torch.cuda.empty_cache()

        # The correct config keys for TribeModel extractors:
        MODEL = TribeModel.from_pretrained(
            checkpoint_dir,
            cache_folder=str(CACHE_FOLDER),
            device=device,
            config_update={
                "data.num_workers": 0,
                "accelerator": device,
                "data.video_feature.image.device": device, # Use GPU if available
                "data.audio_feature.device": device,       # Use GPU if available
                "data.text_feature.device": device,        # Use GPU if available
            },
        )
        return MODEL


def _run_prediction(
    job_id: str,
    video_path,
    audio_path,
    text_path,
    image_path,
    hf_token,
):
    """Worker function executed in a background thread."""
    try:
        JOBS[job_id]["status"]   = "loading_model"
        JOBS[job_id]["progress"] = 5

        # ── Optional HuggingFace login ─────────────────────────────────────
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        # ── Imports ────────────────────────────────────────────────────────
        import pandas as pd
        import numpy as np
        from neuralset.events.utils import standardize_events
        from neuralset.events.transforms import (
            AddText,
            AddSentenceToWords,
            AddContextToWords,
        )

        JOBS[job_id]["progress"] = 10

        # ── Load / cache model ─────────────────────────────────────────────
        model = _load_model()

        JOBS[job_id]["progress"] = 40
        JOBS[job_id]["status"]   = "extracting_events"

        # ── Multimodal data extraction ─────────────────────────────────────

        # Case A: Image only
        if image_path and not video_path and not audio_path:
            try:
                import easyocr
                # Use GPU for OCR if available
                reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))
                result = reader.readtext(image_path, detail=0)
                extracted_text = " ".join(result).strip() or "Visual content analysis"
            except Exception:
                extracted_text = "Visual content analysis"

            from moviepy import ImageClip
            temp_video = CACHE_FOLDER / f"{job_id}_img_anchor.mp4"
            clip = ImageClip(str(image_path), duration=5.0)
            # Use GPU for encoding if available
            v_codec = "h264_nvenc" if DEVICE == "cuda" else "libx264"
            
            clip.write_videofile(
                str(temp_video), fps=10, codec=v_codec,
                audio=False, preset="ultrafast" if v_codec == "libx264" else None, 
                logger=None,
            )

            df = model.get_events_dataframe(video_path=str(temp_video))

            words = extracted_text.split()
            if words:
                dur = 5.0 / len(words)
                new_words = [
                    {
                        "type": "Word", "text": w,
                        "start": i * dur, "duration": dur,
                        "timeline": "default", "subject": "default",
                    }
                    for i, w in enumerate(words)
                ]
                df = df[df.type != "Word"]
                df = pd.concat([df, pd.DataFrame(new_words)], ignore_index=True)
                df = standardize_events(df)
                df = AddText()(df)
                df = AddSentenceToWords(max_unmatched_ratio=0.99)(df)
                df = AddContextToWords(sentence_only=False, max_context_len=1024, split_field="")(df)

        # Case B: Text only
        elif text_path and not video_path and not audio_path:
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    script = f.read().strip()
            except Exception:
                script = "Marketing script analysis"

            dummy = _ensure_dummy_video()

            df = model.get_events_dataframe(video_path=str(dummy))

            words = script.split() or ["(empty)"]
            dur   = 5.0 / len(words)
            new_words = [
                {
                    "type": "Word", "text": w,
                    "start": i * dur, "duration": dur,
                    "timeline": "default", "subject": "default",
                }
                for i, w in enumerate(words)
            ]
            df = df[df.type != "Word"]
            df = pd.concat([df, pd.DataFrame(new_words)], ignore_index=True)
            df = standardize_events(df)
            df = AddText()(df)
            df = AddSentenceToWords(max_unmatched_ratio=0.99)(df)
            df = AddContextToWords(sentence_only=False, max_context_len=1024, split_field="")(df)

        # Case C: Video / Audio / mixed
        else:
            df = model.get_events_dataframe(
                video_path=video_path,
                audio_path=audio_path,
                text_path=text_path,
            )

        JOBS[job_id]["progress"] = 60
        JOBS[job_id]["status"]   = "predicting"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        preds, segments = model.predict(events=df)

        JOBS[job_id]["progress"] = 80
        JOBS[job_id]["status"]   = "analysing"

        results = _analyse(preds, segments, df)

        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["status"]   = "done"
        JOBS[job_id]["results"]  = results

    except OSError as e:
        trace = traceback.format_exc()
        print(f"[ERROR] I/O Error during prediction: {e}")
        try:
            crash_log = BASE_DIR / "CRASH_LOG.txt"
            with open(crash_log, "w", encoding="utf-8") as f:
                f.write(f"I/O Error: {e}\n\n{trace}")
        except:
            pass
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"]  = f"Input/Output Error (Disk full or Drive issue): {e}"
    except Exception as exc:
        trace = traceback.format_exc()
        print(f"[ERROR] General Error: {exc}")


def _analyse(preds, segments, df) -> dict:
    """Convert raw TRIBE v2 predictions into marketing metrics."""
    import numpy as np

    T, V = preds.shape

    mean_activation_per_step = preds.mean(axis=1).tolist()
    peak_activation_per_step = preds.max(axis=1).tolist()

    overall_mean = float(np.mean(preds))
    overall_peak = float(np.max(preds))

    # Normalise 0-100 for Attention Score
    raw = np.array(mean_activation_per_step)
    lo, hi = raw.min(), raw.max()
    attention_scores = (((raw - lo) / (hi - lo + 1e-9)) * 100).tolist()

    # ROI proxy mapping
    visual_roi    = preds[:, :V // 4]
    audio_roi     = preds[:, V // 4 : V // 2]
    language_roi  = preds[:, V // 2 : 3 * V // 4]
    attention_roi = preds[:, 3 * V // 4 :]

    def norm01(arr):
        a = np.array(arr).mean(axis=1) if len(arr.shape) > 1 else np.array(arr)
        mn, mx = a.min(), a.max()
        return ((a - mn) / (mx - mn + 1e-9)).tolist()

    roi_norm = {
        "visual":    norm01(visual_roi),
        "auditory":  norm01(audio_roi),
        "language":  norm01(language_roi),
        "attention": norm01(attention_roi),
    }

    # Segments
    segment_scores = []
    for i, seg in enumerate(segments):
        if i < T:
            act = float(preds[i].mean())
            segment_scores.append(
                {
                    "index":      i,
                    "start":      float(getattr(seg, "start", i * 2)),
                    "end":        float(getattr(seg, "end", i * 2 + 2)),
                    "text":       str(getattr(seg, "text", "")),
                    "activation": act,
                    "attention":  float(attention_scores[i]),
                }
            )

    top_hooks = sorted(segment_scores, key=lambda x: x["attention"], reverse=True)[:5]

    def classify(score: float) -> str:
        if score >= 75:
            return "🔥 High Engagement"
        if score >= 50:
            return "✅ Moderate Engagement"
        if score >= 25:
            return "⚠️ Low Engagement"
        return "❌ Very Low"

    avg_attention  = float(np.mean(attention_scores))
    peak_attention = float(np.max(attention_scores))
    peak_timestep  = int(np.argmax(raw))

    hook_window = max(1, min(2, len(attention_scores)))
    hook_score  = float(np.mean(attention_scores[:hook_window]))

    win = max(1, min(3, len(attention_scores)))
    if len(attention_scores) >= win:
        retention_curve = np.convolve(
            attention_scores, np.ones(win) / win, mode="valid"
        ).tolist()
    else:
        retention_curve = attention_scores[:]

    events_preview = []
    for _, row in df.head(300).iterrows():
        events_preview.append(
            {
                "type":     str(row.get("type", "")),
                "start":    float(row.get("start", 0)),
                "duration": float(row.get("duration", 0)),
                "text":     str(row.get("text", ""))[:80],
                "context":  str(row.get("context", ""))[:80],
            }
        )

    return {
        "kpis": {
            "avg_attention":  round(avg_attention, 1),
            "peak_attention": round(peak_attention, 1),
            "hook_score":     round(hook_score, 1),
            "peak_timestep":  peak_timestep,
            "overall_mean":   round(overall_mean, 4),
            "overall_peak":   round(overall_peak, 4),
            "n_timesteps":    T,
            "n_vertices":     V,
            "engagement":     classify(avg_attention),
        },
        "timeseries": {
            "attention_scores": [round(x, 2) for x in attention_scores],
            "mean_activation":  [round(x, 4) for x in mean_activation_per_step],
            "peak_activation":  [round(x, 4) for x in peak_activation_per_step],
            "retention_curve":  [round(x, 2) for x in retention_curve],
        },
        "roi":      roi_norm,
        "segments": segment_scores,
        "top_hooks": top_hooks,
        "events_preview": events_preview,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyse", methods=["POST"])
def analyse():
    has_file = any(k in request.files for k in ["video", "audio", "text", "image"])
    has_text = bool(request.form.get("text"))
    if not has_file and not has_text:
        return jsonify({"error": "Please upload at least one file or enter script text."}), 400

    job_id   = str(uuid.uuid4())
    hf_token = request.form.get("hf_token", "").strip()

    def save(field, kinds):
        f = request.files.get(field)
        if f and f.filename and allowed_file(f.filename, kinds):
            name = secure_filename(f.filename)
            path = UPLOAD_FOLDER / f"{job_id}_{name}"
            f.save(path)
            return str(path)
        return None

    video_path = save("video", ALLOWED_VIDEO)
    audio_path = save("audio", ALLOWED_AUDIO)
    text_path  = save("text",  ALLOWED_TEXT)
    image_path = save("image", ALLOWED_IMAGE)

    # Handle direct text input from textarea
    if not text_path and has_text:
        text_path = str(UPLOAD_FOLDER / f"{job_id}_script.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(request.form.get("text"))

    JOBS[job_id] = {"status": "queued", "progress": 0, "results": None, "error": None}

    threading.Thread(
        target=_run_prediction,
        args=(job_id, video_path, audio_path, text_path, image_path, hf_token),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    print("NeuroAds — Brain Response Predictor")
    print(f"Python {sys.version}")
    print("Starting on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
