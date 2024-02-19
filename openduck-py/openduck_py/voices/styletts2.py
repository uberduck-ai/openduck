import tempfile
import re
import yaml
import pylru

import phonemizer
import torch
from scipy.io.wavfile import write as write_wav
import librosa
import torchaudio
from fugashi import Tagger
from nltk.tokenize import word_tokenize
import numpy as np

from .api.modules.diffusion.sampler import (
    DiffusionSampler,
    ADPM2Sampler,
    KarrasSchedule,
)
from .settings import (
    DEVICE,
    SAMPLE_RATE,
    ESPEAK_LANGUAGES,
    LATIN_CHARACTERS,
    PUNCTUATION_CHARACTERS,
    MODEL_BUCKET,
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
from openduck_py.utils.s3 import download_file, upload_file


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


def _load_model(
    model_bucket, model_path, text_aligner, pitch_extractor, plbert, model_params
):
    # NOTE (Sam): building the model prior to loading makes using the "model_args" key to store the config not work.
    model = build_model(
        recursive_munch(model_params),
        text_aligner,
        pitch_extractor,
        plbert,
    )

    _ = [model[key].to(DEVICE) for key in model]

    state_dict = load_object_from_s3(
        s3_key=model_path, s3_bucket=model_bucket, loader=load_params
    )
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


# NOTE (Sam): this is a version of the loading code from celery-bark that caches models.
def load_model(
    cache, model_bucket, model_path, text_aligner, pitch_extractor, plbert, model_params
):
    key = model_path.replace("/", "_")
    if key not in cache:
        cache[key] = _load_model(
            model_bucket,
            model_path,
            text_aligner,
            pitch_extractor,
            plbert,
            model_params,
        )

    return cache[key]


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
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = _preprocess(audio).to(DEVICE)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


def _kanji_to_hiragana(text):
    tagger = Tagger()
    words = tagger.parseToNodeList(text)
    return " ".join([word.feature.kana or word.surface for word in words])


def _japanese_to_ipa(text, phonemizer):
    text = _kanji_to_hiragana(text)
    text = text.replace("ィ", "イ")
    text = text.replace("ェ", "エ")
    text = text.replace("ェ", "ー")
    text = text.replace("  ", " ")

    ps = phonemizer.phonemize([text])

    return ps


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
    # NOTE (Sam): if voice is japanese, phonemize japanese and english separately to avoid issue where japanese phonemizer doesn't handle english well.
    # We should probably extend this for other non-English langauges, but more subtle cases may be tougher.
    if language == "japanese":
        text_blocks, is_english = _split_by_language(text)
        phonemes = []
        for block, is_english in zip(text_blocks, is_english):
            if is_english:
                phonemizer = load_phonemizer("english", phonemizers)
                ps = phonemizer.phonemize([block])
                ps = word_tokenize(ps[0])  # reorganizes spaces and periods
                ps = " ".join(ps)
            else:
                phonemizer = load_phonemizer("japanese", phonemizers)
                ps = _japanese_to_ipa(block, phonemizer)
                ps = word_tokenize(ps[0])
                ps = " ".join(ps)
            phonemes.append(ps)
        ps = " ".join(phonemes)
    else:
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
asr_config = load_object_from_s3(
    s3_key="styletts2/asr/config.yml", s3_bucket=MODEL_BUCKET, loader=load_config
)
text_aligner = load_object_from_s3(
    s3_key="styletts2/asr/epoch_00080.pth",
    s3_bucket=MODEL_BUCKET,
    loader=lambda path: load_asr_models(path, asr_config),
)
pitch_extractor = load_object_from_s3(
    s3_key="styletts2/jdc/bst.t7", s3_bucket=MODEL_BUCKET, loader=load_f0_models
)
plbert_config = load_object_from_s3(
    s3_key="styletts2/plbert/config.yml", s3_bucket=MODEL_BUCKET, loader=load_config
)
plbert = load_object_from_s3(
    s3_key="styletts2/plbert/step_1000000.t7",
    s3_bucket=MODEL_BUCKET,
    loader=lambda x: load_plbert(plbert_config, x),
)

model_path = "styletts2/prototype_voice.pth"
model_bucket = "uberduck-models-us-west-2"
model, sampler = load_model(
    cache=cache,
    model_bucket=model_bucket,
    model_path=model_path,
    text_aligner=text_aligner,
    pitch_extractor=pitch_extractor,
    plbert=plbert,
    model_params=model_params,
)

style_prompt_path = "bertie-chipper.wav"
style_prompt_bucket = "uberduck-audio-files"

ref_s = load_object_from_s3(
    s3_key=style_prompt_path,
    s3_bucket=style_prompt_bucket,
    loader=lambda x: _compute_style(model, x),
)


def styletts2_inference(text: str, language: str = "english"):
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
    )
    return audio_array
