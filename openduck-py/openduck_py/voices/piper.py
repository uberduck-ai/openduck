import numpy as np
import torch
from torchaudio.functional import resample

from piper.voice import PiperVoice

from .settings import DEVICE

model = PiperVoice.load("models/cartoon-boy-upbeat-piper.onnx",
                        config_path="models/cartoon-boy-upbeat-piper.onnx.json",
                        use_cuda=(DEVICE == "cuda"))

def inference(text: str, output_sample_rate: int, language: str = "english") -> np.ndarray:
    audio = model.synthesize_stream_raw(
        text,
        speaker_id = 0,
    )
    audio = b"".join(audio)
    audio = torch.frombuffer(audio, dtype=torch.int16).float() / 32767 # TODO silly
    audio = resample(audio, model.config.sample_rate, output_sample_rate)
    return audio
