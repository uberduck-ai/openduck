import time
import os
from typing import Dict

import torch
import numpy as np
from scipy.spatial.distance import cdist
from pyannote.core import Segment
from pyannote.audio import Pipeline, Inference, Model


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


# def identify_speakers(
#     audio_data: torch.Tensor,
#     sample_rate: int,
#     speaker_embedding: np.array,
#     pipeline: Pipeline,
#     inference: Inference,
# ) -> Dict:
#     """
#     Segment the speaker in the input file and return the segments and their distances from the speaker embedding.

#     Args:
#         input_filename (str): The input file to segment.
#         speaker_embedding (np.array): The speaker embedding to compare against.
#         pipeline (Pipeline): The diarization pipeline.
#         inference (Inference): The embedding inference pipeline.

#     Returns:
#         Dict: The segments and their distances from the speaker embedding.

#     """
#     # NOTE(wrl): Diarization takes about 25x longer than inference (0.62 seconds vs 0.024 seconds). Can we only run
#     # the inference on 1 second chunks?
#     speaker_segments = {}
#     start = time.time()
#     pyannote_input = {"waveform": audio_data, "sample_rate": sample_rate}
#     output = pipeline(pyannote_input)
#     print(f"Speaker diarization took {time.time() - start:.3f} seconds")
#     for segment, _, speaker in output.itertracks(yield_label=True):
#         start = time.time()
#         print(f"Shape of audio data: {audio_data.shape}")
#         print(f"Segment: {segment.start} - {segment.end}")
#         print(
#             f"length of audio data: {len(audio_data[1])} and sample rate: {sample_rate} so duration is {len(audio_data) / sample_rate}"
#         )
#         segment_embedding = inference.crop(pyannote_input, segment)
#         print(f"Embedding inference took {time.time() - start:.3f} seconds")
#         distance = cdist([segment_embedding], [speaker_embedding], metric="cosine")[
#             0, 0
#         ]
#         if speaker not in speaker_segments:
#             speaker_segments[speaker] = {"times": [], "distances": []}
#         speaker_segments[speaker]["times"].append((segment.start, segment.end))
#         speaker_segments[speaker]["distances"].append(distance)

#     return speaker_segments


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
    print()
    print(f"Speaker diarization took {time.time() - start:.3f} seconds")

    audio_length_seconds = (
        audio_data.shape[1] / sample_rate
    )  # Corrected to use the second dimension of audio_data

    for segment, _, speaker in output.itertracks(yield_label=True):
        # Boundary check for the segment
        segment_end = min(segment.end, audio_length_seconds)
        if segment.start >= segment_end or segment.end - segment.start < 0.3:
            continue  # Skip segments that are out of bounds or have no duration

        start = time.time()
        print(f"Shape of audio data: {audio_data.shape}")
        print(f"Segment: {segment.start} - {segment_end}")
        print(
            f"Length of audio data: {audio_data.shape[1]} samples and sample rate: {sample_rate} so duration is {audio_length_seconds} seconds"
        )
        adjusted_segment = Segment(segment.start, segment_end)

        segment_embedding = inference.crop(pyannote_input, adjusted_segment)
        print(f"Embedding inference took {time.time() - start:.3f} seconds")

        distance = cdist([segment_embedding], [speaker_embedding], metric="cosine")[
            0, 0
        ]
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


import numpy as np
import torch
from typing import List


import numpy as np
import torch


def segment_audio(
    audio_data: np.array,
    sample_rate: int,
    speaker_embedding: np.array,
    pipeline,
    inference,
) -> np.array:

    # converts to [1, n] shape if not already
    # audio_data = audio_data.reshape(1, -1) if len(audio_data.shape) == 1 else audio_data
    audio_data_tensor = torch.tensor(audio_data).unsqueeze(0)
    # Identify speakers
    start = time.time()
    speaker_segments = identify_speakers(
        audio_data=audio_data_tensor,
        sample_rate=sample_rate,
        speaker_embedding=speaker_embedding,
        pipeline=pipeline,
        inference=inference,
    )
    print(f"Speaker identification took {time.time() - start:.3f} seconds")

    # Filter identified speaker segments
    start = time.time()
    speaker_segments = filter_voices(speaker_segments)
    print(f"Speaker filtering took {time.time() - start:.3f} seconds")
    BUFFER = 0.1  # seconds
    concatenated_audio_data = np.array([], dtype=np.float32)

    start_time = time.time()
    for speaker, data in speaker_segments.items():
        for start, end in data["times"]:
            start = max(0, start - BUFFER)
            end = min(len(audio_data) / sample_rate, end + BUFFER)
            segment = audio_data[int(start * sample_rate) : int(end * sample_rate)]
            print(f"Audio data shape: {audio_data.shape}")
            print(f"Segment shape: {segment.shape}")
            print(f"Concatenated audio shape: {concatenated_audio_data.shape}")
            concatenated_audio_data = np.concatenate((concatenated_audio_data, segment))

    print(f"Audio concatenation took {time.time() - start_time:.3f} seconds")
    return concatenated_audio_data


if __name__ == "__main__":
    # print(speaker_embedding)
    # from pprint import pprint
    # import librosa

    # filename = "../../../samples/speaker-seg-test.wav"
    pipeline, inference = load_pipelines()
    # speaker_embedding = inference("../../../samples/aec-cartoon.wav")
    # # speaker_embedding
    # librosa_audio, sr = librosa.load(filename, sr=16000)
    # # # speaker_segments = identify_speakers(
    # # #     audio_data=librosa_audio,
    # # #     sample_rate=sr,
    # # #     speaker_embedding=speaker_embedding,
    # # #     pipeline=pipeline,
    # # #     inference=inference,
    # # # )

    # # # pprint(speaker_segments, width=1)

    # # # speaker_segments = filter_voices(speaker_segments)
    # # # pprint(speaker_segments, width=1)

    # filtered_audio_data = segment_audio(
    #     audio_data=librosa_audio,
    #     sample_rate=sr,
    #     speaker_embedding=speaker_embedding,
    #     pipeline=pipeline,
    #     inference=inference,
    # )
    # # # convert to wavfile and write to filtered.wav

    # from scipy.io.wavfile import write

    # write("filtered2.wav", sr, filtered_audio_data)

    #
    audio_data = np.load('../../1708722359.8986475_audio_data.npy')
    audio_data = torch.tensor(audio_data).view(1, -1)
    input = {"waveform": audio_data, "sample_rate": 16000}
    
    for i in range(150, 120, -1):
        end_time = i / 100
        print(f"Testing end time: {end_time}")
        inference.crop(input, Segment(1,end_time))
    # inference.crop(input, Segment(1,1.35))
