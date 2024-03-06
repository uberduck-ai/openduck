from piper.voice import PiperVoice
import torch
from torchaudio.functional import resample

from .settings import DEVICE

model = PiperVoice.load("models/cartoon-boy-upbeat-piper.onnx",
                        config_path="models/cartoon-boy-upbeat-piper.onnx.json",
                        use_cuda=(DEVICE == "cuda"))

def inference(text: str, language: str = "english"):
    audio = model.synthesize_stream_raw(
        text,
        speaker_id = 0,
    )
    audio = b"".join(audio)
    audio = torch.frombuffer(audio, dtype=torch.int16).float() / 32767 # TODO silly
    audio = resample(audio, model.config.sample_rate, 24000)
    return audio
