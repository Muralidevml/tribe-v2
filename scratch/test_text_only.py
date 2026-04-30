
import os
import sys
import pandas as pd
import numpy as np
import torch
import pathlib
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

# Add project root to path
sys.path.append(os.getcwd())

from moviepy import ColorClip
if not os.path.exists("dummy_5s.mp4"):
    clip = ColorClip(size=(640, 480), color=(0, 0, 0), duration=5.0)
    clip.write_videofile("dummy_5s.mp4", fps=1, codec="libx264", audio=False, logger=None)
if not os.path.exists("dummy.mp4"):
    clip = ColorClip(size=(640, 480), color=(0, 0, 0), duration=1.0)
    clip.write_videofile("dummy.mp4", fps=1, codec="libx264", audio=False, logger=None)

from tribev2.demo_utils import TribeModel
from neuralset.events.utils import standardize_events
from neuralset.events.transforms import AddText, AddSentenceToWords, AddContextToWords

def test_config(name, events):
    print(f"\n--- Testing Config: {name} ---")
    df = pd.DataFrame(events)
    df = standardize_events(df)
    df = AddText()(df)
    df = AddSentenceToWords(max_unmatched_ratio=0.99)(df)
    df = AddContextToWords(sentence_only=False, max_context_len=1024, split_field="")(df)
    
    try:
        model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache", device="cpu")
        preds, segments = model.predict(events=df, verbose=False)
        print(f"Result: {preds.shape[0]} segments, Mean activation: {preds.mean():.6f}")
        if preds.shape[0] > 0:
            print(f"Sample prediction slice: {preds[0, :5]}")
    except Exception as e:
        print(f"Error: {e}")

# Config D: Words + 5s Video Anchor (Testing for variance)
test_config("Text + 5s Video Anchor", [
    {"type": "Video", "filepath": "dummy_5s.mp4", "start": 0.0, "timeline": "default", "subject": "default"},
    {"type": "Word", "text": "hello", "start": 0.0, "duration": 0.5, "timeline": "default", "subject": "default"},
    {"type": "Word", "text": "world", "start": 0.5, "duration": 0.5, "timeline": "default", "subject": "default"}
])

# Config E: Words (Different) + 5s Video Anchor 
test_config("Text (Different) + 5s Video Anchor", [
    {"type": "Video", "filepath": "dummy_5s.mp4", "start": 0.0, "timeline": "default", "subject": "default"},
    {"type": "Word", "text": "catastrophic", "start": 0.0, "duration": 0.5, "timeline": "default", "subject": "default"},
    {"type": "Word", "text": "failure", "start": 0.5, "duration": 0.5, "timeline": "default", "subject": "default"}
])
