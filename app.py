from flask import Flask, render_template, request, redirect
import subprocess
import os
import random

app = Flask(__name__)

HIGH_STRESS_ACTIVITIES = [
    "Take a walk in nature", "Do some deep breathing", "Stretch for 10 minutes", "Write in a journal",
    "Practice mindfulness", "Drink a glass of water", "Do 10 jumping jacks", "Watch a funny video"
]

MEDIUM_STRESS_ACTIVITIES = [
    "Step outside", "Make a warm drink", "Call a friend", "Try a puzzle",
    "Dance to music", "Stretch arms", "Do a meditation", "Sketch something silly"
]

def get_stress_level():
    if os.path.exists("stress_level.txt"):
        with open("stress_level.txt", "r") as f:
            return f.read().strip()
    return None

def get_random_file(folder):
    files = os.listdir(folder)
    return os.path.join(folder, random.choice(files))

@app.route('/')
def index():
    return render_template('main.html', stage='start')

@app.route('/start', methods=['POST'])
def start():
    subprocess.call(['python3', 'real_time_emotion.py'])
    stress = get_stress_level()
    return render_template('main.html', stage='result', stress=stress)

@app.route('/video', methods=['POST'])
def video():
    video = get_random_file("videos")
    os.system(f"open \"{video}\"" if os.name == "posix" else f"start {video}")
    return render_template('main.html', stage='video', file=video)

@app.route('/music', methods=['POST'])
def music():
    music = get_random_file("music")
    os.system(f"open \"{music}\"" if os.name == "posix" else f"start {music}")
    return render_template('main.html', stage='music', file=music)

@app.route('/activities', methods=['POST'])
def activities():
    stress = get_stress_level()
    activities = random.sample(HIGH_STRESS_ACTIVITIES if stress == "high" else MEDIUM_STRESS_ACTIVITIES, 5)
    return render_template('main.html', stage='activities', activities=activities)

if __name__ == '__main__':
    app.run(debug=True)