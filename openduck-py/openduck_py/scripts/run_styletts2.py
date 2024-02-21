import soundfile as sf
from openduck_py.voices import styletts2
from openduck_py.voices.settings import SAMPLE_RATE


audio = styletts2.styletts2_inference(
    text="Hey, I'm the Uberduck! What do you want to learn about today?"
)

sf.write("startup.wav", audio, SAMPLE_RATE)  # Assuming the sample rate is 22050 Hz
