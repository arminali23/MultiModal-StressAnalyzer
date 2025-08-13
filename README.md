# Real-Time Emotion and Stress Detection System

This project is a real-time emotion and stress detection system that uses webcam and microphone input to detect a user's emotional state and estimate their stress level. Based on the detected stress level, the system recommends personalized content such as videos, music, and suggested activities through a simple web interface.

## Features

- Emotion detection from facial expressions using webcam
- Stress detection based on both facial emotions and voice analysis
- Real-time analysis of emotional state (30-second capture)
- Web interface built with Flask for user interaction
- Personalized recommendations based on stress level:
  - Calming videos
  - Relaxing music
  - Stress-relief activities
- All data processed locally, no internet connection required

## Project Structure

final project/
├── app.py Flask web app
├── real_time_emotion.py Emotion detection via webcam
├── realtime_voice_emotion.py Voice emotion detection (optional)
├── recommend.py Recommendation system logic
├── emotion_modelv1.pt Pretrained CNN model for facial emotion
├── voice_emotion_model.pt Pretrained model for voice emotion (if used)
├── templates/
│ └── index.html Web interface HTML file
├── videos/
│ ├── video1.mp4
│ ├── video2.mp4
│ └── ...
├── music/
│ ├── music1.mp3
│ ├── music2.mp3
│ └── ...
└── README.md

## Installation

Make sure you have Python 3.x installed along with the necessary packages.

To install Flask and PyTorch:

pip install flask
pip install torch torchvision

If you use voice emotion detection:

pip install sounddevice
pip install numpy scipy

## How to Run

1. Navigate to the project folder in terminal.

cd "final project"

2. Start the web application.

python app.py

3. Open your browser and go to:

http://127.0.0.1:5000

## How It Works

- The user clicks "Start Webcam Analysis" to begin capturing emotions via camera.
- The system runs for approximately 30 seconds and records facial expressions.
- Optionally, voice input is also analyzed to determine emotional tone.
- Based on the emotion data, a stress level is calculated (Low, Medium, or High).
- The user then receives content suggestions in the following order:
  1. A calming video
  2. A music track
  3. A list of relaxing activities
- At each stage, the user can choose whether they want another recommendation of the same type or move to the next step.

## Notes

- Voice emotion detection is included in the system architecture but may be skipped during demo.
- All video and music files are stored locally under the "videos" and "music" folders.
- The system does not rely on external APIs or internet-based processing.

## Author

This project was developed by Armin Ali as part of a master's program on Artificial Intelligence for Media.
