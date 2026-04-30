from tribev2.demo_utils import TribeModel
import traceback
import sys
from pathlib import Path

# Fix for paths
import pathlib
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

try:
    Path("test.txt").write_text("Hello, this is a test to check if the brain prediction model runs.")
        
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder="./cache"
    )
    
    # Try text to trigger audio/whisperx
    df = model.get_events_dataframe(text_path="test.txt")
    print("DataFrame generated successfully!")
    
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
