import os
import re
import io
import sys
import time
import json
import glob 
import ctypes
import queue
import platform
import threading
import subprocess
import urllib.parse
from io import BytesIO
from datetime import datetime
import requests
import speech_recognition as sr
import pyttsx3
from PIL import Image 
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from screen_brightness_control import set_brightness
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from dotenv import dotenv_values
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pygame
from groq import Groq
from openai import OpenAI 
import webbrowser
import cv2
import pollinations
from gtts import gTTS
import tempfile
import mediapipe as mp
from playsound import playsound
import winsound
from datetime import datetime, timedelta
import threading
from RealTimeSearchEngine import RealtimeSearchEngine
import dateparser
import re
import wikipedia

os.environ['GLOG_minloglevel'] = '2'

newsapi = "new_api_key_here"
weather_api_key = "weather_api_key_here"
openai_key = "your_openai_api_key_here"
MEMORY_FILE = "memory.json"
gemini_key="gemini_api_key_here"
PDF_DIR = os.path.join(os.getcwd(), "generated_pdfs")
ASSET_DIR = os.path.join(PDF_DIR, "assets")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)
load_dotenv()
env_vars = dotenv_values(".env")
GroqAPIkey = env_vars.get("GroqAPIkey") or os.getenv("GROQ_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
groq_client = Groq(api_key=GroqAPIkey) if GroqAPIkey else None
openai_client = OpenAI(api_key=openai_key) if openai_key else None
openai = OpenAI(api_key=openai_key)

if gemini_key:
    genai.configure(api_key=gemini_key)

recognizer = sr.Recognizer()

camera_running = False
camera_thread = None


def speak(text: str):
    print("Jarvis:", text)
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    print("Finished speaking:", text)

def word_to_num(word):
    numbers = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        "hundred": 100
    }
    if isinstance(word, (int, float)):
        return int(word)
    if str(word).isdigit():
        return int(word)
    parts = str(word).strip().lower().split()
    total = 0
    for p in parts:
        if p in numbers:
            total += numbers[p]
    return total


def fetch_and_speak_news():
    try:
        speak("Fetching the latest news.")
        response = requests.get(
            f"https://newsapi.org/v2/top-headlines?country=us&apiKey={newsapi}", timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            if not articles:
                speak("No news articles found.")
            else:
                for i, article in enumerate(articles, 1):
                    title = (article.get('title') or '').strip()
                    if title:
                        print(f"News {i}: {title}")
                        speak(title)
        else:
            speak("Sorry, I couldn't fetch the news right now.")
            print("News API Error:", response.text)
    except Exception as e:
        speak("An error occurred while fetching the news.")
        print("News Error:", e)


def get_weather(city):
    try:
        speak(f"Fetching weather for {city}.")
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            description = data['weather'][0]['description']
            speak(f"The temperature in {city} is {temp}°C with {description}.")
        else:
            speak("I couldn't find weather data for that location.")
    except Exception as e:
        speak("There was a problem getting the weather.")
        print("Weather Error:", e)


def set_volume(level):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevelScalar(max(0.0, min(1.0, level / 100.0)), None)


def change_volume(delta):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    current = volume.GetMasterVolumeLevelScalar()
    new_level = min(max(current + delta / 100, 0.0), 1.0)
    volume.SetMasterVolumeLevelScalar(new_level, None)


def mute_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMute(1, None)


def unmute_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMute(0, None)


def change_brightness(level):
    set_brightness(level)


def control_system(action, delay=0):
    if action == "shutdown":
        os.system(f"shutdown /s /t {delay}")
    elif action == "restart":
        os.system(f"shutdown /r /t {delay}")
    elif action == "lock":
        ctypes.windll.user32.LockWorkStation()



def load_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w") as f:
            json.dump({"notes": [], "reminders": [], "preferences": {}}, f)
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)


def remember(command):
    memory = load_memory()
    if "remember" in command:
        note = command.replace("remember", "").strip()
        memory["notes"].append(note)
        save_memory(memory)
        return f"Okay, I'll remember that: {note}"
    elif "remind me" in command:
        reminder = command.replace("remind me", "").strip()
        memory["reminders"].append({"task": reminder, "time": str(datetime.now())})
        save_memory(memory)
        return f"Reminder saved: {reminder}"
    elif "what do you remember" in command:
        notes = memory.get("notes", [])
        return "Here's what I remember: " + "; ".join(notes) if notes else "I don't remember anything yet."
    return None


def get_ai_response(prompt):

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini failed: {e}")

    try:
        model = "llama-3.3-70b-versatile"
        response = RealtimeSearchEngine
    except Exception as e:
        print(f"Groq Failed: {e}")

    try:
     
        if openai_client:
            response = openai_client.responses.create(
                model="gpt-4o-mini",
                input=prompt
            )
            return response.output_text
    except Exception as e:
        print(f"OpenAI failed: {e}")

    try:
        
        if groq_client:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"Groq failed: {e}")

    try:
        
        return wikipedia.summary(prompt, sentences=2)
    except Exception as e:
        print(f"Wikipedia failed: {e}")
        return "Sorry, I can't get that right now."


class AlarmManager:
    def __init__(self, speak_callback):
        self.alarms = []
        self.speak = speak_callback
        self.running = True
        thread = threading.Thread(target=self.check_alarms, daemon=True)
        thread.start()

    def parse_time(self, time_string):
        # Normalize input
        time_string = time_string.lower().strip()
        time_string = time_string.replace(".", "")  
        time_string = re.sub(r'\s+', ' ', time_string)  

    
        if "in" in time_string or "after" in time_string:
            minutes = re.search(r'(\d+)\s*minute', time_string)
            hours = re.search(r'(\d+)\s*hour', time_string)
            delta = timedelta()
            if hours:
                delta += timedelta(hours=int(hours.group(1)))
            if minutes:
                delta += timedelta(minutes=int(minutes.group(1)))
            return datetime.now() + delta

    
        match = re.match(r'(\d{1,2})(\d{2})?\s*(am|pm)', time_string)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            period = match.group(3)
            if period == "pm" and hour != 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
            alarm_time = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            if alarm_time < datetime.now():
                alarm_time += timedelta(days=1)
            return alarm_time

       
        match = re.match(r'(\d{1,2}):(\d{2})\s*(am|pm)?', time_string)
        if match:
            hour, minute, period = int(match.group(1)), int(match.group(2)), match.group(3)
            if period == "pm" and hour != 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
            alarm_time = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            if alarm_time < datetime.now():
                alarm_time += timedelta(days=1)
            return alarm_time

        return None

    def set_alarm(self, time_string):
        alarm_time = self.parse_time(time_string)
        if alarm_time:
            self.alarms.append(alarm_time)
            self.speak(f"Alarm set for {alarm_time.strftime('%I:%M %p')}")
        else:
            self.speak("Sorry, I could not understand the time format.")

    
    def play_alarm(self):
        pygame.mixer.init()
        pygame.mixer.music.load("assets/mixkit-alert-alarm-1005.wav")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
           pygame.time.Clock().tick(10)

    def check_alarms(self):
        while self.running:
            now = datetime.now()
            for alarm in self.alarms[:]:
                if now >= alarm:
                    print("Alarm condition met!")
                    self.speak("Alarm ringing now!")
                    try:
                        self.play_alarm() 
                    except Exception as e:
                        print(f"Alarm sound failed: {e}")
                    self.alarms.remove(alarm)
      
alarm_manager = AlarmManager(speak)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def start_camera():
    global camera_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    camera_running = True
    print("Camera thread started.")
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened() and camera_running:
            success, frame = cap.read()
            if not success:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            if results.detections:
                print("👀 Face detected!")
            cv2.imshow("Camera Awareness", frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                camera_running = False
                break
    cap.release()
    cv2.destroyAllWindows()
    print("Camera thread stopped.")



def generate_topic_content(topic: str) -> str:
    """Prefer Groq; fall back to OpenAI; finally a local template."""
    sys_prompt = (
        "You are an expert writer. Produce a clear, structured article for a PDF. "
        "Use short paragraphs, section headings (plain text), and bullet points. "
        "Return plain text only, no markdown symbols."
    )
    
    if groq_client:
        try:
            resp = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Topic: {topic}"},
                ],
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("[Groq] Error:", e)
    
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Topic: {topic}"},
                ],
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("[OpenAI] Error:", e)
    # Fallback template
    return (
        f"{topic}\n\nOverview\n- Definition\n- Why it matters\n\nKey Concepts\n- Concept 1\n- Concept 2\n\nExamples\n- Example A\n- Example B\n\nConclusion\n- Final thoughts"
    )


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "topic"


def fetch_topic_images(topic: str, count: int = 2) -> list:
    """Fetch royalty-free images via Unsplash source (no API key)."""
    paths = []
    q = urllib.parse.quote(topic)
    for i in range(count):
        url = f"https://source.unsplash.com/1024x768/?{q}&sig={i}"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                img = Image.open(BytesIO(r.content)).convert("RGB")
                fname = f"{sanitize_filename(topic)}_{i}.jpg"
                fpath = os.path.join(ASSET_DIR, fname)
                img.save(fpath, "JPEG", quality=85)
                paths.append(fpath)
        except Exception as e:
            print("[Images] Fetch error:", e)
    return paths


def _scaled_rl_image(path: str, max_width: float = 5.8*inch) -> RLImage:
    im = Image.open(path)
    w, h = im.size
    scale = min(1.0, max_width / float(w))
    return RLImage(path, width=w * scale, height=h * scale)


def create_pdf(topic: str, content: str, image_paths: list) -> str:
    safe = sanitize_filename(topic)
    fname = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{safe}.pdf"
    path = os.path.join(PDF_DIR, fname)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name="TitleCenter", parent=styles['Title'], alignment=TA_CENTER, spaceAfter=12)
    body = styles['BodyText']

    story = []
    story.append(Paragraph(topic, title_style))
    story.append(Spacer(1, 10))

  
    for img_path in image_paths:
        try:
            story.append(_scaled_rl_image(img_path))
            story.append(Spacer(1, 8))
        except Exception as e:
            print("[PDF] image add error:", e)

    for para in [p.strip() for p in content.split("\n\n") if p.strip()]:
        story.append(Paragraph(para.replace("\n", "<br/>"), body))
        story.append(Spacer(1, 6))

    doc = SimpleDocTemplate(path, pagesize=A4)
    doc.build(story)
    return path

def ImageGen(prompt):
     try: 
        url = f"https://image.pollinations.ai/prompt/{prompt}" 
        response = requests.get(url)
        if response.status_code == 200: 
            os.makedirs("Data", exist_ok=True) 
            file_path = os.path.join("Data", "Image.png") 
            img = Image.open(BytesIO(response.content)) 
            img.save(file_path) 
            return file_path 
        else: print("Error generating image:", response.text) 
        return None 
     except Exception as e:
         print("ImageGen Error:", e) 
         return None
         
def OpenImage(path):
     if path and os.path.exists(path): 
        img = Image.open(path) 
        img.show() 
     else:
        print("Image file does not exist.") 

def Main(newprompt): 
    file_path = ImageGen(newprompt) 
    if file_path: 
        OpenImage(file_path)


def open_anything(name):
    websites = {
        "google": "https://www.google.com",
        "bing": "https://www.bing.com",
        "yahoo": "https://www.yahoo.com",
        "wikipedia": "https://www.wikipedia.org",
        "youtube": "https://www.youtube.com",
        "spotify": "https://open.spotify.com",
        "netflix": "https://www.netflix.com",
        "amazon prime": "https://www.primevideo.com",
        "hotstar": "https://www.hotstar.com",
        "facebook": "https://www.facebook.com",
        "instagram": "https://www.instagram.com",
        "twitter": "https://twitter.com",
        "x": "https://twitter.com",
        "linkedin": "https://www.linkedin.com",
        "reddit": "https://www.reddit.com",
        "pinterest": "https://www.pinterest.com",
        "snapchat": "https://www.snapchat.com",
        "amazon": "https://www.amazon.com",
        "flipkart": "https://www.flipkart.com",
        "ebay": "https://www.ebay.com",
        "myntra": "https://www.myntra.com",
        "ajio": "https://www.ajio.com",
        "bbc news": "https://www.bbc.com/news",
        "cnn": "https://edition.cnn.com",
        "ndtv": "https://www.ndtv.com",
        "india today": "https://www.indiatoday.in",
        "times of india": "https://timesofindia.indiatimes.com",
        "github": "https://github.com",
        "stackoverflow": "https://stackoverflow.com",
        "w3schools": "https://www.w3schools.com",
        "geeksforgeeks": "https://www.geeksforgeeks.org",
        "royal enfield": "https://www.royalenfield.com",
        "tata motors": "https://www.tatamotors.com",
        "tesla": "https://www.tesla.com",
        "bmw": "https://www.bmw.com",
    }
    if name.lower() in websites:
        webbrowser.open(websites[name.lower()])
        speak(f"Opening {name}")
        return
    try:
        system_os = platform.system()
        if system_os == "Windows":
            os.startfile(name)
        elif system_os == "Darwin":
            subprocess.run(["open", name])
        elif system_os == "Linux":
            subprocess.run(["xdg-open", name])
        else:
            speak("Your OS is not supported for opening apps.")
            return
        speak(f"Opening {name}")
    except FileNotFoundError:
        speak(f"I couldn't find {name} on this system.")
    except Exception as e:
        speak(f"Error opening {name}: {e}")


def play_song(song_name, platform_name="youtube"):
    query = urllib.parse.quote(song_name)
    if platform_name.lower() == "spotify":
        url = f"https://open.spotify.com/search/{query}"
    else:
        url = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(url)
    print(f"Opening {platform_name.title()} search for: {song_name}")


PDF_CMD_PATTERN = re.compile(r"(?:create|make|generate)\s+(?:a\s+)?pdf\s+(?:about|on|regarding)\s+(.+)")


def handle_system_command(command):
    command = command.lower()
    if "set volume to" in command:
        level_str = command.split("set volume to")[-1].strip()
        level = word_to_num(level_str)
        set_volume(level)
        speak(f"Volume set to {level} percent.")
    elif "increase volume by" in command:
        level = word_to_num(command.split("increase volume by")[-1].strip())
        change_volume(level)
        speak(f"Increased volume by {level} percent.")
    elif "decrease volume by" in command:
        level = word_to_num(command.split("decrease volume by")[-1].strip())
        change_volume(-level)
        speak(f"Decreased volume by {level} percent.")
    elif "mute" in command:
        mute_volume()
        speak("Volume muted.")
    elif "unmute" in command:
        unmute_volume()
        speak("Volume unmuted.")
    elif "set brightness to" in command:
        level = word_to_num(command.split("set brightness to")[-1].strip())
        change_brightness(level)
        speak(f"Brightness set to {level} percent.")
    elif "increase brightness by" in command:
        level = word_to_num(command.split("increase brightness by")[-1].strip())
        change_brightness(min(100, level + 10))
        speak(f"Increased brightness by {level} percent.")
    elif "decrease brightness by" in command:
        level = word_to_num(command.split("decrease brightness by")[-1].strip())
        change_brightness(max(0, level - 10))
        speak(f"Decreased brightness by {level} percent.")
    elif "shutdown" in command:
        delay = 0
        if "in" in command and "minute" in command:
            delay_str = command.split("in")[-1].split("minute")[0].strip()
            delay = word_to_num(delay_str) * 60
        control_system("shutdown", delay)
        speak("Shutting down your computer.")
    elif "restart" in command:
        control_system("restart")
        speak("Restarting your computer.")
    elif "lock" in command:
        control_system("lock")
        speak("Locking your computer.")


def process_command(command):
    global camera_running, camera_thread
    command = command.lower().strip()
    print("Processed Command:", command)


    m = PDF_CMD_PATTERN.search(command)
    if m:
        topic = m.group(1).strip()
        speak(f"Generating PDF about {topic}. Please wait.")
        content = generate_topic_content(topic)
        images = fetch_topic_images(topic, count=3)
        pdf_path = create_pdf(topic, content, images)
        speak(f"PDF on {topic} is ready. Saved at {pdf_path}.")
        print(f"PDF created: {pdf_path}")
        return

    if "open google" in command:
        speak("Opening Google.")
        webbrowser.open("https://www.google.com")

    elif "open facebook" in command:
        speak("Opening Facebook.")
        webbrowser.open("https://www.facebook.com")

    elif "open youtube" in command:
        speak("Opening YouTube.")
        webbrowser.open("https://www.youtube.com")

    elif "amazon" in command:
        speak("Opening Amazon.")
        webbrowser.open_new_tab("https://www.amazon.com")

    elif "flipkart" in command:
        speak("Opening Flipkart.")
        webbrowser.open_new_tab("https://www.flipkart.com")

    elif "royal enfield" in command:
        speak("Opening Royal Enfield.")
        webbrowser.open_new_tab("https://www.royalenfield.com")

    elif "open linkedin" in command:
        speak("Opening LinkedIn.")
        webbrowser.open("https://www.linkedin.com")

    elif command.startswith("open "):
        thing_to_open = command.replace("open", "").strip()
        open_anything(thing_to_open)

    elif "news" in command:
        fetch_and_speak_news()

    elif "weather in" in command:
        city = command.split("weather in")[-1].strip()
        get_weather(city) if city else speak("Please specify a city name.")

    elif "remember" in command or "remind me" in command or "what do you remember" in command:
        response = remember(command)
        if response:
            speak(response)

    elif command.startswith("remind me"):
        # Simple alias to set_alarm; you could parse label from text if you like
        phrase = command.replace("remind me", "").strip()
        alarm_manager.set_alarm(phrase, label="Reminder")

    elif "set alarm" in command:
        time_string = command.replace("set alarm for", "").strip()
        alarm_manager.set_alarm(time_string)

    elif "play" in command:
        if "spotify" in command:
            song_name = command.replace("play", "").replace("on spotify", "").strip()
            play_song(song_name, "spotify")
        else:
            song_name = command.replace("play", "").replace("on youtube", "").strip()
            play_song(song_name, "youtube")

    elif "stop camera" in command:
        camera_running = False
        if camera_thread and camera_thread.is_alive():
            camera_thread.join()
        speak("Camera stopped.")

    elif "camera" in command:
        if not camera_thread or not camera_thread.is_alive():
            camera_thread = threading.Thread(target=start_camera, daemon=True)
            camera_thread.start()
            speak("Starting camera awareness.")
        else:
            speak("The camera is already running.")

    elif any(w in command for w in ["volume", "brightness", "shutdown", "restart", "lock", "mute"]):
        handle_system_command(command)

    elif "generate image of" in command: 
        prompt = command.replace("generate image of", " ").strip() 
        speak(f"Generating an image of {prompt}. Please wait.") 
        file_path = ImageGen(prompt) 
        if file_path: 
            OpenImage(file_path) 
            speak(f"Your image of {prompt} has been generated and opened.") 
        else: 
            speak("I couldn't generate the image. Please try again.")

    elif command in ["good bye", "goodbye", "bye", "see you", "exit", "quit"]:
        speak("Goodbye Daksh, see you soon!")
        sys.exit(0)

    elif command in ["hello", "hi", "hey"]:
        speak("Hello Daksh, how can I help you?")


    else:
        speak("Thinking....")
        try:
            answer = get_ai_response(command)
            speak(answer)
        except Exception as e:
            print("[Jarvis get_ai_response] error:", repr(e))
            speak("Sorry, I can't get that right now.")


if __name__ == "__main__":
    recognizer = sr.Recognizer()
    speak("Initializing Jarvis...")
    while True:
        try:
            with sr.Microphone() as source:
                print("\nListening for wake word 'Jarvis'...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            wake_word = recognizer.recognize_google(audio)
            print("Heard:", wake_word)
            if "jarvis" in wake_word.lower():
                speak("Yes Daksh?")
                with sr.Microphone() as source:
                    print("Listening for your command...")
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                command = recognizer.recognize_google(audio)
                print("Raw recognized command:", command)
                process_command(command)
            else:
                print("Wake word not detected.")
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Google Speech Recognition error: {e}")
        except KeyboardInterrupt:
            speak("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
