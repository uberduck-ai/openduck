import os
import tempfile
import re
import yaml
import pylru

import phonemizer
import torch
import librosa
import torchaudio
from nltk.tokenize import word_tokenize
import numpy as np
from torchaudio.functional import resample

from .api.modules.diffusion.sampler import (
    DiffusionSampler,
    ADPM2Sampler,
    KarrasSchedule,
)
from .settings import (
    DEVICE,
    ESPEAK_LANGUAGES,
    LATIN_CHARACTERS,
    PUNCTUATION_CHARACTERS,
)

from .api import (
    inference,
    recursive_munch,
)
from .api.models import (
    build_model,
    load_asr_models,
    load_f0_models,
)
from .api.utils.plbert.util import load_plbert
from openduck_py.utils.s3 import download_file


STYLETTS2_SAMPLE_RATE = 24000

_to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)

mean, std = -4, 4


def load_config(path):
    config = yaml.safe_load(open(path))
    return config


def load_object_from_s3(s3_key, s3_bucket, loader=lambda x: x):
    with tempfile.NamedTemporaryFile() as temp_file:
        download_file(
            s3_key,
            s3_bucket,
            temp_file.name,
        )
        attribute = loader(temp_file.name)

    return attribute


def load_params(path):
    params_whole = torch.load(path, map_location=DEVICE)
    return params_whole["net"]


def load_model(model_path, text_aligner, pitch_extractor, plbert, model_params):
    # NOTE (Sam): building the model prior to loading makes using the "model_args" key to store the config not work.
    model = build_model(
        recursive_munch(model_params),
        text_aligner,
        pitch_extractor,
        plbert,
    )

    _ = [model[key].to(DEVICE) for key in model]

    # state_dict = load_object_from_s3(
    #     s3_key=model_path, s3_bucket=model_bucket, loader=load_params
    # )

    state_dict = load_params(model_path)

    for key in model:
        if key in state_dict:
            print("%s loaded" % key)
            try:
                model[key].load_state_dict(state_dict[key])
            except:
                from collections import OrderedDict

                state_subdict = state_dict[key]
                new_state_dict = OrderedDict()
                for k, v in state_subdict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)

    _ = [model[key].eval() for key in model]

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(
            sigma_min=0.0001, sigma_max=3.0, rho=9.0
        ),  # empirical parameters
        clamp=False,
    )
    return model, sampler


def load_phonemizer(language, cache):
    if language not in cache:
        cache[language] = phonemizer.backend.EspeakBackend(
            language=ESPEAK_LANGUAGES[language],
            preserve_punctuation=True,
            with_stress=True,
            words_mismatch="ignore",
            language_switch="remove-flags",
        )
    return cache[language]


def _preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = _to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def _compute_style(model, path):
    wave, sr = librosa.load(path, sr=STYLETTS2_SAMPLE_RATE)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != STYLETTS2_SAMPLE_RATE:
        audio = librosa.resample(audio, sr, STYLETTS2_SAMPLE_RATE)
    mel_tensor = _preprocess(audio).to(DEVICE)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


def _split_by_language(text):
    """Takes a block of text and returns a list of blocks and a list of booleans indicating whether each block is in latin characters or not.
    This can be used for basic language splitting."""
    blocks = []
    is_english = []
    current_block = ""
    previous_is_english = None
    for character in text:
        current_is_english = bool(re.findall(LATIN_CHARACTERS, character))
        current_is_punctuation = bool(re.findall(PUNCTUATION_CHARACTERS, character))
        assert not (current_is_english and current_is_punctuation)
        if previous_is_english is None:
            current_block = current_block + character
            if not current_is_punctuation:
                previous_is_english = current_is_english
        elif current_is_punctuation:
            current_block = current_block + character

        elif current_is_english == previous_is_english:
            current_block = current_block + character
        else:
            blocks.append(current_block)
            is_english.append(previous_is_english)
            current_block = character
            previous_is_english = current_is_english

    blocks.append(current_block)
    if previous_is_english is not None:
        is_english.append(previous_is_english)
    else:
        is_english.append(True)

    return blocks, is_english


def _phonemize(text, phonemizers, language):
    text = text.strip()
    phonemizer = load_phonemizer(language, phonemizers)
    ps = phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])  # reorganizes spaces and periods
    ps = " ".join(ps)

    return ps


def resize_array(input_array, new_size):
    """
    Resize a NumPy array to a new size.
    If the array is longer than new_size, it is truncated.
    If the array is shorter, it is right-padded with zeros.

    :param input_array: NumPy array to be resized.
    :param new_size: The desired new size of the array.
    :return: The resized NumPy array.
    """
    current_size = input_array.shape[0]

    if current_size > new_size:
        # Truncate the array if it's longer than new_size
        return input_array[:new_size]
    elif current_size < new_size:
        # Pad the array with zeros if it's shorter than new_size
        padding = np.zeros(new_size - current_size)
        return np.concatenate((input_array, padding))
    else:
        # If the array is already of the desired size, return it as is
        return input_array


config_path = "styletts2/rap_v1_config.yml"
config_bucket = "uberduck-models-us-west-2"
model_params = load_object_from_s3(
    s3_key=config_path, s3_bucket=config_bucket, loader=load_config
)["model_params"]
cache = pylru.lrucache(1)


models_prefix = os.path.join(os.path.dirname(__file__), "../..")

asr_config = load_config(os.path.join(models_prefix, "models/styletts2/asr_config.yml"))
plbert_config = load_config(
    os.path.join(models_prefix, "models/styletts2/plbert_config.yml")
)

text_aligner = load_asr_models(
    os.path.join(models_prefix, "models/styletts2/text_aligner.pth"), asr_config
)
pitch_extractor = load_f0_models(
    os.path.join(models_prefix, "models/styletts2/pitch_extractor.t7")
)
plbert = load_plbert(
    plbert_config, os.path.join(models_prefix, "models/styletts2/plbert.t7")
)
model, sampler = load_model(
    model_path=os.path.join(models_prefix, "models/styletts2/cartoon-boy-upbeat.pth"),
    text_aligner=text_aligner,
    pitch_extractor=pitch_extractor,
    plbert=plbert,
    model_params=model_params,
)


ref_s = _compute_style(model, "models/styletts2/cartoon-boy-upbeat.wav")


def styletts2_inference(
    text: str, language: str = "english", output_sample_rate=24000
) -> np.ndarray:
    print("styletts2.run started")

    # NOTE (Sam): to deal with short inference issue https://github.com/yl4579/StyleTTS2/issues/46.
    warm_start_required = (
        len(text) < 40 and language == "english"
    )  # only supported for english due to uncertainty on number of minimum number of graphemes for other languages.
    if warm_start_required:
        text = "warming up some hot bars for your listening pleasure: " + text
        warm_start_index = 58  # w.r.t the phonemization of the warm start text.
    else:
        warm_start_index = 0

    phonemizers = pylru.lrucache(5)
    phonemes = _phonemize(text, phonemizers, language)
    audio_array = inference(
        phonemes,
        model,
        sampler,
        ref_s,
        warm_start_index=warm_start_index,
        output_sample_rate=output_sample_rate,
    )
    return audio_array
