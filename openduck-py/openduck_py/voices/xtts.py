import numpy as np
from torchaudio.functional import resample

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from .settings import DEVICE

config = XttsConfig()
config.load_json("models/XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="models/XTTS-v2") # use_deepspeed=True
if DEVICE == "cuda":
    model.cuda()

gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["models/cartoon-boy-upbeat.wav"])

def inference(text: str, output_sample_rate: int, language: str = "english") -> np.ndarray:
    audio = model.inference(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
    )
    audio = audio["wav"]
    audio = resample(audio, 24000, output_sample_rate)
    return audio

def streaming_inference(text: str, output_sample_rate: int, language: str = "english") -> np.ndarray:
    stream = model.inference_stream(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
    )
    return (resample(chunk, 24000, output_sample_rate) for chunk in stream)
