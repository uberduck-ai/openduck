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

import os
import queue
import asyncio
import threading
import wave
import websockets

import click
import numpy as np
import sounddevice as sd
import pyaudio
from uuid import uuid4

# Audio configuration
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1

UBERDUCK_API_HOST = os.environ["UBERDUCK_API_HOST"]

session = str(uuid4())

play_queue = queue.Queue()

shutdown_flag = threading.Event()


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
        print("Received audio!")
        # Assuming `message` is audio data. For simplicity, we're not handling the case where
        # the message needs decoding or conversion. This will depend on how your audio data
        # is being sent (e.g., raw bytes, encoded, etc.).

        # Convert bytes to numpy array (assuming the incoming data is compatible with numpy)
        # This step may need adjustment based on your data format.
        audio_data = np.frombuffer(message, dtype=np.int16)
        play_queue.put(message)


def play_audio():
    """Continuously checks the queue and plays audio chunks."""
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=OUTPUT_SAMPLE_RATE, output=True
    )
    while True:
        if shutdown_flag.is_set():
            break
        if not play_queue.empty():
            data = play_queue.get()
            stream.write(data)  # Play the audio chunk


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

    def audio_callback(self, indata, frames, time, status):
        audio_bytes = indata.tobytes()
        if self.recording:
            self.audio_queue.put(audio_bytes)


def play_startup_sound():
    p = pyaudio.PyAudio()
    CHUNK = 1024
    # Open stream
    with wave.open("startup.wav", "rb") as wf:
        # Instantiate PyAudio and initialize PortAudio system resources (1)
        p = pyaudio.PyAudio()
        # Open stream (2)
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
        # Play samples from the wave file (3)
        while len(data := wf.readframes(CHUNK)):  # Requires Python 3.8+ for :=
            stream.write(data)
    # Close stream (4)
    stream.close()
    # Release PortAudio system resources (5)
    p.terminate()


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


@click.command()
@click.option("--record", is_flag=True, help="Enable recording.")
def main(record):
    ws_proto = "ws" if "localhost" in UBERDUCK_API_HOST else "wss"
    record_param = "true" if record else "false"
    uri = f"{ws_proto}://{UBERDUCK_API_HOST}?session_id={session}&record={record_param}"
    play_startup_sound()

    recorder = AudioRecorder()
    recorder.start_recording()
    play_thread = threading.Thread(target=play_audio)
    play_thread.start()
    try:
        asyncio.run(run_websocket(uri, recorder.audio_queue))
    except KeyboardInterrupt:
        shutdown_flag.set()
        print("Shutting down.")


if __name__ == "__main__":
    main()
