import time
import os
from typing import Dict, List

import torch
import numpy as np
from scipy.spatial.distance import cdist
from pyannote.core import Segment
from pyannote.audio import Pipeline, Inference, Model

from openduck_py.settings import HF_AUTH_TOKEN, EMB_MATCH_THRESHOLD, WS_SAMPLE_RATE


def load_pipelines() -> tuple:
    """
    Load the diarization and embedding pipelines.
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_AUTH_TOKEN,
    )
    pipeline.to(torch.device("cuda"))

    embedding_pipeline = Model.from_pretrained(
        "pyannote/embedding", use_auth_token=HF_AUTH_TOKEN
    )
    inference = Inference(embedding_pipeline, window="whole")
    inference.to(torch.device("cuda"))

    return pipeline, inference


def identify_speakers(
    audio_data: torch.Tensor,
    sample_rate: int,
    speaker_embedding: np.array,
    pipeline,
    inference,
) -> Dict:
    """
    Segment the speaker in the input file and return the segments and their distances from the speaker embedding.

    Args:
        audio_data (torch.Tensor): The waveform of the input audio.
        sample_rate (int): The sample rate of the input audio.
        speaker_embedding (np.array): The speaker embedding to compare against.
        pipeline: The diarization pipeline.
        inference: The embedding inference pipeline.

    Returns:
        Dict: The segments and their distances from the speaker embedding.
    """
    speaker_segments = {}
    start = time.time()
    pyannote_input = {"waveform": audio_data, "sample_rate": sample_rate}
    output = pipeline(pyannote_input)

    audio_length_seconds = audio_data.shape[1] / sample_rate

    for segment, _, speaker in output.itertracks(yield_label=True):
        segment_end = min(segment.end, audio_length_seconds)
        if segment.start >= segment_end or segment.end - segment.start < 0.3:
            continue

        start = time.time()
        adjusted_segment = Segment(segment.start, segment_end)

        segment_embedding = inference.crop(pyannote_input, adjusted_segment)

        distance = get_embedding_distance(segment_embedding, speaker_embedding)
        if speaker not in speaker_segments:
            speaker_segments[speaker] = {"times": [], "distances": []}
        speaker_segments[speaker]["times"].append((segment.start, segment_end))
        speaker_segments[speaker]["distances"].append(distance)

    return speaker_segments


def filter_voices(d: dict, threshold: float = 0.5):
    """
    This filters out the voices that are below the threshold because
    that indicates that it is our own TTS voice.
    """
    new_d = {}
    for k, v in d.items():
        new_v = {"times": [], "distances": []}
        for i, dist in enumerate(v["distances"]):
            if dist > threshold:
                new_v["times"].append(v["times"][i])
                new_v["distances"].append(dist)
        if new_v["times"]:
            new_d[k] = new_v
    return new_d


def segment_audio(
    audio_data: np.array,
    sample_rate: int,
    speaker_embedding: np.array,
    pipeline,
    inference,
) -> np.array:

    audio_data_tensor = torch.tensor(audio_data).unsqueeze(0)
    start = time.time()
    speaker_segments = identify_speakers(
        audio_data=audio_data_tensor,
        sample_rate=sample_rate,
        speaker_embedding=speaker_embedding,
        pipeline=pipeline,
        inference=inference,
    )

    start = time.time()
    speaker_segments = filter_voices(speaker_segments)
    BUFFER = 0.1  # seconds
    concatenated_audio_data = np.array([], dtype=np.float32)

    start_time = time.time()
    for speaker, data in speaker_segments.items():
        for start, end in data["times"]:
            start = max(0, start - BUFFER)
            end = min(len(audio_data) / sample_rate, end + BUFFER)
            segment = audio_data[int(start * sample_rate) : int(end * sample_rate)]
            concatenated_audio_data = np.concatenate((concatenated_audio_data, segment))

    return concatenated_audio_data


def get_embedding_distance(embedding_1: np.array, embedding_2: np.array) -> float:
    return cdist([embedding_1], [embedding_2], metric="cosine")[0, 0]


def is_embedding_match(embedding_1: np.array, embedding_2: np.array) -> bool:
    return get_embedding_distance(embedding_1, embedding_2) < EMB_MATCH_THRESHOLD
