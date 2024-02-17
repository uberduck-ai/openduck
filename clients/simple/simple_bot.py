"""
Simple bot that records audio from the microphone, generates an audio response with uberduck or openai, and plays the response.

Usage:
    python uberduck_bot.py

Options:
    --openai    Use OpenAI for processing instead of Uberduck

Instructions:
    0. Set UBERDUCK_API and OPENAI_ORGANIZATION_ID environment variables 
    1. Press the space bar to start recording
    2. Press the space bar again to stop recording
    3. A response will be generated and played
    4. Wait for the response to finish playing before starting a new recording
    5. Press the space bar to start a new recording
"""

import argparse
import os
import requests
import threading
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
from sshkeyboard import listen_keyboard
import soundfile as sf
from openai import OpenAI
from pydub import AudioSegment

SAMPLE_RATE = 22050
CHANNELS = 1

IDLE = "IDLE"
RECORDING = "RECORDING"
PROCESSING = "PROCESSING"
PLAYBACK = "PLAYBACK"
RECORDING_FILE = "recording.wav"
RESPONSE_FILE = "response.wav"
UBERDUCK_API = os.environ["UBERDUCK_API"]

speech_file_path = Path(__file__).parent / "response.wav"
chat_history = [
    {
        "role": "system",
        "content": "You are a conversational small stuffed animal toy for children. You are a conversational partner that teaches the child. Your responses are never more than three sentences. You always end with a question so I can respond.",
    }
]

parser = argparse.ArgumentParser(description="Audio processing script.")
parser.add_argument(
    "--openai",
    action="store_true",
    help="Use OpenAI for processing instead of Uberduck",
)
args = parser.parse_args()
USE_UBERDUCK = not args.openai
if not USE_UBERDUCK:
    client = OpenAI(
        organization=os.environ["OPENAI_ORGANIZATION_ID"],
    )


def uberduck_response():
    with open(RECORDING_FILE, "rb") as file:
        print(f"[INFO] Sending audio to the server...")
        files = {"audio": (RECORDING_FILE, file, "audio/wav")}
        response = requests.post(UBERDUCK_API, files=files)
        print(f"[INFO] Response received from the server: {response.status_code}")
    if response.status_code == 200:
        data = np.frombuffer(response.content, dtype=np.int16)
        sf.write(RESPONSE_FILE, data, 24000)
    else:
        print(
            f"[ERROR] Failed to get audio from the server, status code: {response.status_code}"
        )


def openai_response():
    transcript = ""
    with open(RECORDING_FILE, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="text"
        )

    print(f"[PROCESSING] Transcript received: {transcript}")
    chat_history.append({"role": "user", "content": transcript})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=chat_history
    )
    text = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": text})

    print(f"[PROCESSING] Text response: {text}")
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="alloy",
        input=text,
    )
    print("[PROCESSING] TTS response received.")
    response.stream_to_file(speech_file_path)

    sound = AudioSegment.from_mp3(speech_file_path)
    sound.export("response.wav", format="wav")
    print("[PROCESSING] Audio saved to file.")


class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.duration = 10  # Maximum recording duration in seconds
        self.audio_queue = queue.Queue()

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        print("[INFO] Recording started...")
        threading.Thread(target=self.record, daemon=True).start()

    def stop_recording(self):
        self.recording = False
        print("[INFO] Recording stopped.")
        self.save_recording()

    def record(self):
        with sd.InputStream(
            callback=self.audio_callback, samplerate=SAMPLE_RATE, channels=CHANNELS
        ):
            while self.recording:
                sd.sleep(100)

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_queue.put(indata.copy())

    def save_recording(self):
        while not self.audio_queue.empty():
            data = self.audio_queue.get()
            self.audio_data.append(data)
        if self.audio_data:
            audio_array = np.concatenate(self.audio_data, axis=0)
            sf.write(RECORDING_FILE, audio_array, SAMPLE_RATE)

    def play(self):
        print("[INFO] Playing...")
        data, fs = sf.read("response.wav")
        sd.play(data, 24000)
        sd.wait()
        print("[INFO] Playback finished. Press space to start recording.")

    def start_processing(self):
        print("[INFO] Processing...")
        if USE_UBERDUCK:
            uberduck_response()
        else:
            openai_response()
        print("[INFO] Processing finished.")


class StateMachine:
    def __init__(self):
        self.recorder = AudioRecorder()
        self.state = IDLE

    def set_state(self, state):
        self.state = state

    def __str__(self) -> str:
        return f"State: {self.state}"

    def on_press(self, key):
        print("key: ", key)
        if key == "space":
            if self.state == IDLE:
                self.set_state(RECORDING)
                self.recorder.start_recording()

            elif self.state == RECORDING:
                self.recorder.stop_recording()
                self.set_state(PROCESSING)
                self.recorder.start_processing()
                self.set_state(PLAYBACK)
                self.recorder.play()
                self.set_state(IDLE)

    def run(self):
        listen_keyboard(on_press=self.on_press)


if __name__ == "__main__":
    print("Press space to start recording.")
    sm = StateMachine()
    sm.run()
