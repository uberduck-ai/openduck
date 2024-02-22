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
import queue
from pathlib import Path
import asyncio
import websockets

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from uuid import uuid4
from asgiref.sync import async_to_sync

SAMPLE_RATE = 24000
CHANNELS = 1

UBERDUCK_API_HOST = os.environ["UBERDUCK_API_HOST"]

SILENCE_THRESHOLD = 1.0

session = str(uuid4())


async def send_audio(websocket, audio_queue):
    while True:
        print("sending audio...")
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break
        await websocket.send(audio_chunk)
        audio_queue.task_done()
        await asyncio.sleep(0.01)


async def receive_audio(websocket):
    print("We're receiving audio!")
    async for message in websocket:
        data = np.frombuffer(message, dtype=np.int16)
        sd.play(data, 24000)
        sd.wait()
        print("[INFO] Playing received audio.")


class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.duration = 10  # Maximum recording duration in seconds
        self.audio_queue = queue.Queue()
        self.stream = None

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        print("[INFO] Recording started...")
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=SAMPLE_RATE,  # 1 block = 1 second (SAMPLE_RATE frames)
        )
        self.stream.start()  # Manually start the stream
        sd.sleep(100)
        # while self.recording:
        #     sd.sleep(100)
        # self.stream.stop()  # Manually stop the stream when done recording
        # print("[INFO] Recording stopped.")

    def audio_callback(self, indata, frames, time, status):
        audio_bytes = indata.tobytes()
        if self.recording:
            self.audio_queue.put(audio_bytes)


def play_startup_sound():
    startup_sound, fs = sf.read("startup.wav")
    sd.play(startup_sound, fs)
    sd.wait()


class SharedState:
    pass


async def run_websocket(uri, audio_queue):
    state = SharedState()
    async with websockets.connect(uri) as websocket:
        send_task = asyncio.create_task(send_audio(websocket, audio_queue))
        receive_task = asyncio.create_task(receive_audio(websocket))
        await asyncio.gather(send_task, receive_task)
        while True:
            print("I'm still running!")
            await asyncio.sleep(1)


if __name__ == "__main__":
    uri = f"ws://{UBERDUCK_API_HOST}?session_id={session}"
    play_startup_sound()

    recorder = AudioRecorder()
    recorder.start_recording()
    asyncio.run(run_websocket(uri, recorder.audio_queue))
