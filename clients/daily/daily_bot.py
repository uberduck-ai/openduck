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
import queue

from daily import *

import pyaudio
import wave
import sounddevice as sd
import numpy as np

SPEAKER_SAMPLE_RATE = 24000
MIC_SAMPLE_RATE = 16000
NUM_CHANNELS = 1
MEETING = "https://matthewkennedy5.daily.co/Od7ecHzUW4knP6hS5bug"


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
        self.__output_stream = self.__pyaudio.open(
            format=pyaudio.paInt16,
            channels=NUM_CHANNELS,
            rate=SPEAKER_SAMPLE_RATE,
            output=True,
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

        # self.__recv_thread = threading.Thread(target=self.manage_audio_stream)
        # self.__recv_thread.start()

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
                            # "customConstraints": {
                            #     "autoGainControl": {"exact": True},
                            #     "noiseSuppression": {"exact": True},
                            #     "echoCancellation": {"exact": True},
                            # },
                        },
                    },
                },
                # "publishing": {
                #     "microphone": {
                #         "isPublishing": True,
                #         "sendSettings": {
                #             "channelConfig": "mono",
                #         },
                #     }
                # },
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

        self.__virtual_mic.write_frames(in_data)

        return None, pyaudio.paContinue

    def handle_audio(self, audio):
        self.logger.info("!!! Starting speaking")
        start = time.time()
        b = bytearray()
        final = False
        smallest_write_size = 3200
        try:
            for chunk in audio:
                b.extend(chunk)
                l = len(b) - (len(b) % smallest_write_size)
                if l:
                    self.microphone.write_frames(bytes(b[:l]))
                    b = b[l:]

            if len(b):
                self.microphone.write_frames(bytes(b))
        except Exception as e:
            self.logger.error(f"Exception in handle_audio: {e}")
        finally:
            self.logger.info(f"!!! Finished speaking in {time.time() - start} seconds")

    # def on_speaker_frames(self, buffer):
    #     if not self.__app_quit:
    #         self.__output_stream.write(buffer)
    #         self.__virtual_speaker.read_frames(
    #             SPEAKER_SAMPLE_RATE // 10, completion=self.on_speaker_frames
    #         )

    # def manage_audio_stream(self):
    #     self.__stream.start()
    #     while not self.__app_quit:
    #         time.sleep(0.1)
    #     self.__stream.stop()

    def send_audio_stream(self):
        # output_file = wave.open("received_audio.wav", "wb")
        # output_file.setnchannels(1)
        # output_file.setsampwidth(2)
        # output_file.setframerate(SPEAKER_SAMPLE_RATE)
        while not self.__app_quit:
            frames = self.__virtual_speaker.read_frames(SPEAKER_SAMPLE_RATE // 10)
            # audio_data = np.frombuffer(frames, dtype=np.int16)
            play_queue.put(frames)
            print(f"Received frames: {len(frames)}")

            # audio_16k_np = audio_16k_np.astype(np.float32) / np.iinfo(np.int16).max
            # audio_16k_np = audio_16k_np.astype(np.float32)
            # sd.play(audio_16k_np, SPEAKER_SAMPLE_RATE)
            # sd.wait()
            # print(f"Received audio stream: {len(frames)}")
            # if frames:
            #     output_file.writeframes(frames)
            # self.__output_stream.write(message)
            # time.sleep(1)
        # self.__output_stream.close()
        # output_file.close()

    # def send_audio_stream(self):
    #     # Define a callback function for the OutputStream
    #     def callback(outdata, frames, time, status):
    #         if status:
    #             print(status)
    #         assert frames == len(self.buffer)
    #         outdata[:] = self.buffer.reshape(-1, 1)

    #     # Initialize an empty buffer
    #     self.buffer = np.zeros(
    #         int(SPEAKER_SAMPLE_RATE * 0.5), dtype=np.int16
    #     )  # 0.5 seconds buffer

    #     # Create and start an OutputStream
    #     with sd.OutputStream(
    #         samplerate=SPEAKER_SAMPLE_RATE,
    #         channels=NUM_CHANNELS,
    #         dtype="int16",
    #         callback=callback,
    #     ):
    #         while not self.__app_quit:
    #             frames = self.__virtual_speaker.read_frames(SPEAKER_SAMPLE_RATE // 10)
    #             audio_16k_np = np.frombuffer(frames, dtype=np.int16)
    #             # Here, instead of directly playing back the audio, we update the buffer
    #             # Ensure the buffer size and new audio size match, or implement a more complex buffering strategy
    #             self.buffer = np.concatenate(
    #                 (self.buffer[len(audio_16k_np) :], audio_16k_np)
    #             )
    #             time.sleep(0.1)  # Adjust sleep time as needed for buffer management


play_queue = queue.Queue()


def play_audio():
    """Continuously checks the queue and plays audio chunks."""
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=NUM_CHANNELS,
        rate=SPEAKER_SAMPLE_RATE,
        output=True,
    )
    while True:
        # if shutdown_flag.is_set():
        #     break
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
    finally:
        app.leave()


if __name__ == "__main__":
    main()
