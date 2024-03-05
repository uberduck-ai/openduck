import argparse
import threading
import time

import numpy as np
from daily import *

import sounddevice as sd

SAMPLE_RATE = 16000
NUM_CHANNELS = 1


class SoundDeviceApp:

    def __init__(self, sample_rate, num_channels):
        self.__app_quit = False
        self.__num_channels = num_channels

        # Configure the microphone and speaker devices with Daily
        self.__virtual_mic = Daily.create_microphone_device(
            "my-mic", sample_rate=sample_rate, channels=num_channels, non_blocking=True
        )

        self.__virtual_speaker = Daily.create_speaker_device(
            "my-speaker",
            sample_rate=sample_rate,
            channels=num_channels,
            non_blocking=True,
        )
        Daily.select_speaker_device("my-speaker")

        # Set up the audio stream callback for sounddevice
        self.__stream = sd.Stream(
            samplerate=sample_rate,
            channels=num_channels,
            dtype="int16",
            callback=self.audio_callback,
            finished_callback=self.stream_finished_callback,
        )

        self.__client = CallClient()

        self.__client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "microphone": "subscribed"}}
        )

        self.__thread = threading.Thread(target=self.manage_audio_stream)
        self.__thread.start()

    def audio_callback(self, indata, outdata, frames, time, status):
        if self.__app_quit:
            raise sd.CallbackAbort
        else:
            # Convert indata (NumPy array) to bytes
            indata_bytes = indata.tobytes()

            # Write the converted bytes to the virtual microphone
            self.__virtual_mic.write_frames(indata_bytes)

            # Read frames from the virtual speaker
            buffer = self.__virtual_speaker.read_frames(frames)
            if buffer is not None and len(buffer) >= outdata.nbytes:
                # Convert bytes back to a NumPy array of the appropriate shape and dtype
                buffer_array = np.frombuffer(buffer, dtype=np.int16).reshape(
                    -1, self.__num_channels
                )
                if buffer_array.shape == outdata.shape:
                    outdata[:] = buffer_array
                else:
                    # This might occur if the buffer does not contain enough data
                    outdata.fill(0)  # Fill the rest with zeros
            else:
                outdata.fill(0)  # Fill outdata with zeros if no data is available

    def stream_finished_callback(self):
        self.leave()

    def manage_audio_stream(self):
        self.__stream.start()
        while not self.__app_quit:
            time.sleep(0.1)
        self.__stream.stop()

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
                            "channelConfig": (
                                "stereo" if self.__num_channels == 2 else "mono"
                            ),
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

    def on_speaker_frames(self, buffer):
        # This method may be adjusted or integrated within the callback as needed.
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--meeting",
        required=False,
        help="Meeting URL",
        default="https://wrl.daily.co/test-room",
    )
    parser.add_argument(
        "-c", "--channels", type=int, default=NUM_CHANNELS, help="Number of channels"
    )
    parser.add_argument(
        "-r", "--rate", type=int, default=SAMPLE_RATE, help="Sample rate"
    )
    args = parser.parse_args()

    Daily.init()

    app = SoundDeviceApp(args.rate, args.channels)

    try:
        app.run(args.meeting)
    except KeyboardInterrupt:
        print("Ctrl-C detected. Exiting!")
    finally:
        app.leave()

    # Let leave finish
    time.sleep(2)


if __name__ == "__main__":
    main()
