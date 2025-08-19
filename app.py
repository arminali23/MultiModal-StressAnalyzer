from flask import Flask, render_template, request, redirect, url_for
import threading
import subprocess
import os
import random
import time

# Import your two model runners (each must return {"source": "...", "ratio": float, "level": "low|medium|high"})
from real_time_emotion import run_emotion_detection
from realtime_voice_emotion import run_voice_emotion

app = Flask(__name__)

# ----------------------------- Activity pools ----------------------------- #
HIGH_STRESS_ACTIVITIES = [
    "Take a walk in nature", "Do some deep breathing", "Stretch for 10 minutes", "Write in a journal",
    "Practice mindfulness", "Drink a glass of water", "Do 10 jumping jacks", "Watch a funny video",
    "Light a candle", "Wash your face"
]

MEDIUM_STRESS_ACTIVITIES = [
    "Step outside", "Make a warm drink", "Call a friend", "Try a simple puzzle",
    "Dance to your favorite song", "Stretch arms and legs", "Do a short meditation", "Sketch something silly",
    "Listen to ambient sounds", "Write 3 things you’re grateful for"
]

# ----------------------------- Helpers ----------------------------- #
def read_stress_level_from_txt():
    """Read fused stress level written by /start (if present)."""
    if os.path.exists("stress_level.txt"):
        with open("stress_level.txt", "r") as f:
            return f.read().strip()
    return None

def write_stress_level_to_txt(level_str):
    """Persist fused stress level for compatibility with your existing recommend flow."""
    with open("stress_level.txt", "w") as f:
        f.write(level_str)

def fuse_stress(vision_ratio: float, audio_ratio: float):
    """
    Late fusion of two modalities.
    Default weighting: 0.6 vision, 0.4 audio (adjust if needed).
    Returns (fused_level:str, fused_ratio:float).
    """
    fused_ratio = 0.6 * float(vision_ratio) + 0.4 * float(audio_ratio)
    if fused_ratio < 32.5:
        fused_level = "low"
    elif fused_ratio < 62.5:
        fused_level = "medium"
    else:
        fused_level = "high"
    return fused_level, fused_ratio

def get_random_file(folder):
    """Pick a random file path from a given local folder (e.g., 'videos' or 'music')."""
    files = [f for f in os.listdir(folder) if not f.startswith(".")]
    if not files:
        return None
    return os.path.join(folder, random.choice(files))

def open_file_cross_platform(path):
    """Open a local file with the default OS application (video/music)."""
    if path is None:
        return
    if os.name == "posix":
        os.system(f'open "{path}"')       # macOS
    elif os.name == "nt":
        os.startfile(path)                # Windows
    else:
        # Fallback for Linux/BSD
        subprocess.Popen(["xdg-open", path])

# ----------------------------- Routes ----------------------------- #
@app.route("/")
def index():
    """
    Main page.
    The template should switch UI by 'stage':
      - 'start': show the Start button
      - 'result': show fused results and next-step buttons
      - 'video'/'music'/'activities': show what was opened/listed
    """
    return render_template("main.html", stage="start")

@app.route("/start", methods=["POST"])
def start():
    """
    1) Run camera and voice stress estimation in parallel for ~10 seconds.
    2) Fuse ratios -> final stress level and write to stress_level.txt.
    3) Render 'result' stage with per-modality and fused summaries.
    """
    results = {"vision": None, "audio": None}

    def run_vision():
        try:
            # display=True shows the webcam window like your standalone script
            results["vision"] = run_emotion_detection(duration=10, display=True)
        except Exception as e:
            results["vision"] = {"source": "vision", "ratio": 0.0, "level": "low", "error": str(e)}

    def run_audio():
        try:
            results["audio"] = run_voice_emotion(duration=10)
        except Exception as e:
            results["audio"] = {"source": "audio", "ratio": 0.0, "level": "low", "error": str(e)}

    t1 = threading.Thread(target=run_vision)
    t2 = threading.Thread(target=run_audio)
    t1.start(); t2.start()
    t1.join(); t2.join()

    # Safe defaults if any runner failed
    v = results["vision"] or {"ratio": 0.0, "level": "low"}
    a = results["audio"] or {"ratio": 0.0, "level": "low"}

    fused_level, fused_ratio = fuse_stress(v.get("ratio", 0.0), a.get("ratio", 0.0))
    write_stress_level_to_txt(fused_level)  # keep your recommend.py compatibility

    summary = {
        "vision": v,
        "audio": a,
        "fused": {"level": fused_level, "ratio": fused_ratio}
    }
    return render_template("main.html", stage="result", summary=summary)

@app.route("/video", methods=["POST"])
def video():
    """
    Open a random local video from ./videos and show its name on the page.
    The button 'Another video' in the UI should POST here again.
    """
    video_path = get_random_file("videos")
    open_file_cross_platform(video_path)
    return render_template("main.html", stage="video", file=os.path.basename(video_path) if video_path else "No file found")

@app.route("/music", methods=["POST"])
def music():
    """
    Open a random local track from ./music and show its name on the page.
    The button 'Another music' in the UI should POST here again.
    """
    music_path = get_random_file("music")
    open_file_cross_platform(music_path)
    return render_template("main.html", stage="music", file=os.path.basename(music_path) if music_path else "No file found")

@app.route("/activities", methods=["POST"])
def activities():
    """
    List 5 activities based on fused stress.
    The button 'New activities' in the UI should POST here again.
    """
    stress = read_stress_level_from_txt()
    if stress == "high":
        acts = random.sample(HIGH_STRESS_ACTIVITIES, k=min(5, len(HIGH_STRESS_ACTIVITIES)))
    else:
        acts = random.sample(MEDIUM_STRESS_ACTIVITIES, k=min(5, len(MEDIUM_STRESS_ACTIVITIES)))
    return render_template("main.html", stage="activities", activities=acts)

if __name__ == "__main__":
    app.run(debug=True)
