#
# This demo will join a Daily meeting and it will capture audio from the default
# system microphone and send it to the meeting. It will also play the audio
# received from the meeting via the default system speaker.
#
# Usage: python3 record_and_play.py -m MEETING_URL
#

import argparse
import threading
import time

from daily import *

import pyaudio

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1
MEETING = "https://matthewkennedy5.daily.co/Od7ecHzUW4knP6hS5bug"


class PyAudioApp:

    def __init__(self):
        self.__app_quit = False

        # We configure the microphone as non-blocking so we don't block PyAudio
        # when we write the frames.
        self.__virtual_mic = Daily.create_microphone_device(
            "my-mic",
            sample_rate=INPUT_SAMPLE_RATE,
            channels=NUM_CHANNELS,
            non_blocking=True,
        )

        # In contrast, we configure the speaker as blocking. In this case,
        # PyAudio's output stream callback will wait until we get the data from
        # Daily's speaker.
        self.__virtual_speaker = Daily.create_speaker_device(
            "my-speaker",
            sample_rate=INPUT_SAMPLE_RATE,
            channels=NUM_CHANNELS,
            non_blocking=True,
        )
        Daily.select_speaker_device("my-speaker")

        self.__pyaudio = pyaudio.PyAudio()
        self.__input_stream = self.__pyaudio.open(
            format=pyaudio.paInt16,
            channels=NUM_CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            stream_callback=self.on_input_stream,
        )
        self.__output_stream = self.__pyaudio.open(
            format=pyaudio.paInt16,
            channels=NUM_CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
        )

        self.__client = CallClient()

        self.__client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "microphone": "subscribed"}}
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
                "publishing": {
                    "microphone": {
                        "isPublishing": True,
                        "sendSettings": {
                            "channelConfig": "mono",
                        },
                    }
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
        self.__input_stream.close()
        self.__pyaudio.terminate()

    def on_input_stream(self, in_data, frame_count, time_info, status):
        if self.__app_quit:
            return None, pyaudio.paAbort

        print("Input stream!", len(in_data))
        # If the microphone hasn't started yet `write_frames` this will return
        # 0. In that case, we just tell PyAudio to continue.
        self.__virtual_mic.write_frames(in_data)

        return None, pyaudio.paContinue

    def on_speaker_frames(self, buffer):
        if not self.__app_quit:
            self.__output_stream.write(buffer)
            self.__virtual_speaker.read_frames(4400, completion=self.on_speaker_frames)

    def send_audio_stream(self):
        self.__virtual_speaker.read_frames(4400, completion=self.on_speaker_frames)
        while not self.__app_quit:
            time.sleep(1)
            # pass
        self.__output_stream.close()


def main():
    Daily.init()
    app = PyAudioApp()
    try:
        app.run(MEETING)
    except KeyboardInterrupt:
        print("Ctrl-C detected. Exiting!")
    finally:
        app.leave()


if __name__ == "__main__":
    main()
