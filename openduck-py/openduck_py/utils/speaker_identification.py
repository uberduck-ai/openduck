import torch
import numpy as np
from pyannote.audio import Pipeline, Inference, Model
from scipy.spatial.distance import cdist
from typing import Dict
import os

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")


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
    input_filename: str,
    speaker_embedding: np.array,
    pipeline: Pipeline,
    inference: Inference,
) -> Dict:
    """
    Segment the speaker in the input file and return the segments and their distances from the speaker embedding.

    Args:
        input_filename (str): The input file to segment.
        speaker_embedding (np.array): The speaker embedding to compare against.
        pipeline (Pipeline): The diarization pipeline.
        inference (Inference): The embedding inference pipeline.
        filename (str): The filename of the input file.

    Returns:
        Dict: The segments and their distances from the speaker embedding.

    """
    speaker_segments = {}
    output = pipeline(input_filename)
    for segment, _, speaker in output.itertracks(yield_label=True):
        segment_embedding = inference.crop(filename, segment)
        distance = cdist([segment_embedding], [speaker_embedding], metric="cosine")[
            0, 0
        ]
        if speaker not in speaker_segments:
            speaker_segments[speaker] = {"times": [], "distances": []}
        speaker_segments[speaker]["times"].append((segment.start, segment.end))
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


if __name__ == "__main__":
    from pprint import pprint

    filename = "../../../samples/speaker-seg-test.wav"
    pipeline, inference = load_pipelines()
    speaker_embedding = inference("../../../samples/aec-cartoon.wav")
    speaker_segments = identify_speakers(
        filename, speaker_embedding, pipeline, inference
    )

    pprint(speaker_segments, width=1)

    speaker_segments = filter_voices(speaker_segments)
    pprint(speaker_segments, width=1)
