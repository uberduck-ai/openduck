import threading
import time
import queue

from daily import *

import pyaudio

SPEAKER_SAMPLE_RATE = 24000
MIC_SAMPLE_RATE = 16000
NUM_CHANNELS = 1
MEETING = "https://matthewkennedy5.daily.co/Od7ecHzUW4knP6hS5bug"

stop_event = threading.Event()


class PyAudioApp:

    def __init__(self):
        self.__app_quit = False

        # We configure the microphone as non-blocking so we don't block PyAudio
        # when we write the frames.
        self.__virtual_mic = Daily.create_microphone_device(
            "my-mic",
            sample_rate=MIC_SAMPLE_RATE,
            channels=NUM_CHANNELS,
            non_blocking=True,
        )

        # In contrast, we configure the speaker as blocking. In this case,
        # PyAudio's output stream callback will wait until we get the data from
        # Daily's speaker.
        self.__virtual_speaker = Daily.create_speaker_device(
            "my-speaker",
            sample_rate=SPEAKER_SAMPLE_RATE,
            channels=NUM_CHANNELS,
        )
        Daily.select_speaker_device("my-speaker")

        self.__pyaudio = pyaudio.PyAudio()
        self.__input_stream = self.__pyaudio.open(
            format=pyaudio.paInt16,
            channels=NUM_CHANNELS,
            rate=MIC_SAMPLE_RATE,
            input=True,
            stream_callback=self.on_input_stream,
            frames_per_buffer=MIC_SAMPLE_RATE,
        )

        self.__client = CallClient()

        self.__client.update_subscription_profiles(
            {
                "base": {
                    "camera": "unsubscribed",
                    "microphone": "subscribed",
                }
            }
        )

        self.__thread = threading.Thread(target=self.send_audio_stream)
        self.__thread.start()

    def on_joined(self, data, error):
        if error:
            print(f"Unable to join meeting: {error}")
            self.__app_quit = True

    def run(self, meeting_url):
        self.__client.join(
            meeting_url,
            client_settings={
                "inputs": {
                    "camera": False,
                    "microphone": {
                        "isEnabled": True,
                        "settings": {
                            "deviceId": "my-mic",
                            "customConstraints": {
                                "autoGainControl": {"exact": True},
                                "noiseSuppression": {"exact": True},
                                "echoCancellation": {"exact": True},
                            },
                        },
                    },
                },
            },
            completion=self.on_joined,
        )
        self.__thread.join()

    def leave(self):
        self.__app_quit = True
        self.__client.leave()
        # This is not very pretty (taken from PyAudio docs).
        while self.__input_stream.is_active():
            time.sleep(0.1)
            print("sleeing")
        self.__input_stream.close()
        self.__pyaudio.terminate()

    def on_input_stream(self, in_data, frame_count, time_info, status):
        if self.__app_quit:
            return None, pyaudio.paAbort

        self.__virtual_mic.write_frames(in_data)

        return None, pyaudio.paContinue

    def send_audio_stream(self):
        while not self.__app_quit:
            frames = self.__virtual_speaker.read_frames(SPEAKER_SAMPLE_RATE // 10)
            play_queue.put(frames)
            print(f"Received frames: {len(frames)}")


play_queue = queue.Queue()


def play_audio():
    """Continuously checks the queue and plays audio chunks."""
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=NUM_CHANNELS,
        rate=SPEAKER_SAMPLE_RATE,
        output=True,
    )
    while True:
        if stop_event.is_set():
            break
        if not play_queue.empty():
            data = play_queue.get()
            stream.write(data)


def main():
    Daily.init()
    app = PyAudioApp()
    play_thread = threading.Thread(target=play_audio)
    play_thread.start()
    try:
        app.run(MEETING)
    except KeyboardInterrupt:
        print("Ctrl-C detected. Exiting!")
        app.leave()
        stop_event.set()
        raise


if __name__ == "__main__":
    main()
