import os
import random
import subprocess
import platform

# Activity suggestions
HIGH_STRESS_ACTIVITIES = [
    "Take a walk in nature", "Do some deep breathing", "Stretch for 10 minutes", "Write in a journal",
    "Practice mindfulness", "Try progressive muscle relaxation", "Drink a glass of water", "Listen to soothing music",
    "Do 10 jumping jacks", "Color or draw something", "Water your plants", "Cuddle a pet", "Read a short story",
    "Do 5 minutes of yoga", "Clean up a small area", "Hug someone you trust", "Light a candle", "Wash your face",
    "Massage your shoulders", "Watch a funny video"
]

MEDIUM_STRESS_ACTIVITIES = [
    "Step outside and get some fresh air", "Make yourself a warm drink", "Do a quick tidy-up",
    "Play a calm game", "Call a friend", "Try a simple puzzle", "Dance to your favorite song",
    "Stretch your arms and legs", "Do a short guided meditation", "Sketch something silly",
    "Do a breathing exercise", "Take a power nap", "Unplug from your phone for 10 minutes",
    "Write 3 things you're grateful for", "Pet a dog or cat", "Squeeze a stress ball",
    "Smell something pleasant", "Listen to ambient sounds", "Do finger painting", "Do 15 squats"
]

def open_file(filepath):
    """Open media files based on operating system."""
    if platform.system() == "Darwin":
        subprocess.call(["open", filepath])
    elif platform.system() == "Windows":
        os.startfile(filepath)
    else:
        subprocess.call(["xdg-open", filepath])

def ask_to_replace(item_name):
    """Prompt user to replace media or activity."""
    response = input(f"Would you like to try a different {item_name}? (y/n): ").strip().lower()
    return response == "y"

def recommend_content(stress_level):
    """Legacy interactive mode (asks user for each step)."""
    print(f"\nYour stress level is: {stress_level.upper()}")

    if stress_level == "low":
        print("Your stress level is low. No worries. Still, you can enjoy a relaxing video.")
        while True:
            video = random.choice(os.listdir("videos"))
            video_path = os.path.join("videos", video)
            print(f"Opening relaxing video: {video_path}")
            open_file(video_path)
            if not ask_to_replace("video"):
                break

    elif stress_level == "medium":
        print("Your stress level is moderate. Let's calm down with some activities.")

        while True:
            video = random.choice(os.listdir("videos"))
            video_path = os.path.join("videos", video)
            print(f"Opening relaxing video: {video_path}")
            open_file(video_path)
            if not ask_to_replace("video"):
                break

        while True:
            music = random.choice(os.listdir("music"))
            music_path = os.path.join("music", music)
            print(f"Playing relaxing music: {music_path}")
            open_file(music_path)
            if not ask_to_replace("music"):
                break

        while True:
            activities = random.sample(MEDIUM_STRESS_ACTIVITIES, 5)
            print("\nTry these 5 relaxing activities:")
            for act in activities:
                print(f"• {act}")
            if not ask_to_replace("activity list"):
                break

    elif stress_level == "high":
        print("High stress detected. Let's try to help you relax.")

        while True:
            video = random.choice(os.listdir("videos"))
            video_path = os.path.join("videos", video)
            print(f"Opening calming video: {video_path}")
            open_file(video_path)
            if not ask_to_replace("video"):
                break

        while True:
            music = random.choice(os.listdir("music"))
            music_path = os.path.join("music", music)
            print(f"Playing calming music: {music_path}")
            open_file(music_path)
            if not ask_to_replace("music"):
                break

        while True:
            activities = random.sample(HIGH_STRESS_ACTIVITIES, 5)
            print("\nTry these 5 calming activities:")
            for act in activities:
                print(f"• {act}")
            if not ask_to_replace("activity list"):
                break

        still_feeling_bad = input("\nAre you still feeling overwhelmed? (y/n): ").strip().lower()
        if still_feeling_bad == "y":
            print("\nPlease consider reaching out to a mental health professional. You're not alone.")

    else:
        print("Invalid stress level. Choose from: low, medium, high.")

def recommend_auto(stress_level):
    """Automatic recommendation mode used for real-time emotion detection integration."""
    print(f"\nYour stress level is: {stress_level.upper()}")

    if stress_level == "low":
        print("Low stress level detected. Showing one relaxing video.")
        video = random.choice(os.listdir("videos"))
        open_file(os.path.join("videos", video))

    elif stress_level == "medium":
        print("Medium stress level detected. Playing video, music, and showing activities.")

        video = random.choice(os.listdir("videos"))
        open_file(os.path.join("videos", video))

        music = random.choice(os.listdir("music"))
        open_file(os.path.join("music", music))

        activities = random.sample(MEDIUM_STRESS_ACTIVITIES, 5)
        print("\nTry these relaxing activities:")
        for act in activities:
            print(f"• {act}")

    elif stress_level == "high":
        print("High stress level detected. Playing video, music, and showing calming activities.")

        video = random.choice(os.listdir("videos"))
        open_file(os.path.join("videos", video))

        music = random.choice(os.listdir("music"))
        open_file(os.path.join("music", music))

        activities = random.sample(HIGH_STRESS_ACTIVITIES, 5)
        print("\nTry these calming activities:")
        for act in activities:
            print(f"• {act}")

    else:
        print("Unknown stress level.")