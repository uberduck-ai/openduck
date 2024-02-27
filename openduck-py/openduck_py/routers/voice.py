import asyncio
import os
import re
from tempfile import NamedTemporaryFile
from time import time
from typing import Optional, Dict
import wave
import requests

# NOTE(zach): On Mac OS, the first import fails, but the subsequent one
# succeeds. /shrug.
try:
    import nemo.collections.asr.models as asr_models
except OSError:
    import nemo.collections.asr.models as asr_models


from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

import numpy as np
from scipy.io import wavfile
from sqlalchemy import select
import torch
from torchaudio.functional import resample

from openduck_py.models import DBChatHistory, DBChatRecord
from openduck_py.models.chat_record import EventName
from openduck_py.db import get_db_async, AsyncSession
from openduck_py.prompts import prompt
from openduck_py.voices import styletts2
from openduck_py.settings import IS_DEV, WS_SAMPLE_RATE
from openduck_py.routers.templates import generate
from openduck_py.utils.speaker_identification import (
    segment_audio,
    load_pipelines,
)

if IS_DEV:
    normalize_text = lambda x: x
else:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    normalizer = Normalizer(input_case="cased", lang="en")
    normalize_text = normalizer.normalize


pipeline, inference = load_pipelines()

with open("aec-cartoon-degraded.wav", "wb") as f:
    f.write(
        requests.get(
            "https://s3.us-west-2.amazonaws.com/quack.uberduck.ai/aec-cartoon-degraded.wav"
        ).content
    )

speaker_embedding = inference("aec-cartoon-degraded.wav")

asr_model = asr_models.EncDecCTCModelBPE.from_pretrained(
    model_name="nvidia/stt_en_fastconformer_ctc_large"
)

audio_router = APIRouter(prefix="/audio")


def _transcribe(audio_data):
    with NamedTemporaryFile(suffix=".wav", mode="wb+") as temp_file:
        wavfile.write(temp_file.name, WS_SAMPLE_RATE, audio_data)
        temp_file.flush()
        temp_file.seek(0)
        transcription = asr_model.transcribe([temp_file.name])[0]
    return transcription


class WavAppender:
    def __init__(self, wav_file_path="output.wav"):
        self.wav_file_path = wav_file_path
        self.file = None
        self.params_set = False

    def open_file(self):
        self.file = wave.open(
            self.wav_file_path,
            "wb" if not os.path.exists(self.wav_file_path) else "r+b",
        )

    def append(self, audio_data: np.ndarray):
        if audio_data.dtype == np.float32:
            audio_data = np.round(audio_data * 32767).astype(np.int16)
        assert audio_data.dtype == np.int16
        if not self.file:
            self.open_file()
        if not self.params_set:
            self.file.setnchannels(1)  # Mono
            self.file.setsampwidth(2)  # 16 bits
            self.file.setframerate(WS_SAMPLE_RATE)
            self.params_set = True
        self.file.writeframes(audio_data.tobytes())

    def close_file(self):
        if self.file:
            self.file.close()
            self.file = None
            self.params_set = False


class SileroVad:
    model = None
    # Default window size in examples at https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies#examples
    window_size = 512

    def reset_states(self):
        if hasattr(self, "vad_iterator"):
            self.vad_iterator.reset_states()

    def __call__(self, audio_data):
        if self.model is None:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad"
            )
            self.model = model
            self.utils = utils
            (
                get_speech_timestamps,
                save_audio,
                read_audio,
                VADIterator,
                collect_chunks,
            ) = utils
            self.vad_iterator = VADIterator(self.model)
        speech_dict = self.vad_iterator(audio_data, return_seconds=True)
        return speech_dict


class ResponseAgent:

    def __init__(self, session_id: str):
        self.is_responding = False
        self.session_id = session_id

    def interrupt(self, task: asyncio.Task):
        if self.is_responding:
            print("interrupting!")
            task.cancel()
            self.is_responding = False
        else:
            print("not responding, no need to interrupt.")

    async def start_response(
        self,
        websocket: WebSocket,
        db: AsyncSession,
        audio_data: np.ndarray,
    ):
        print("starting response")
        await log_event(db, self.session_id, "started_response")
        self.is_responding = True

        def _inference(sentence: str):
            audio_chunk = styletts2.styletts2_inference(text=sentence)
            
            audio_chunk_bytes = np.int16(audio_chunk * 32767).tobytes()
            return audio_chunk_bytes

        loop = asyncio.get_running_loop()

        audio_data = segment_audio(
            audio_data=audio_data,
            sample_rate=WS_SAMPLE_RATE,
            speaker_embedding=speaker_embedding,
            pipeline=pipeline,
            inference=inference,
        )
        t0 = time()

        transcription = await loop.run_in_executor(None, _transcribe, audio_data)
        print("transcription", transcription)
        await log_event(db, self.session_id, "transcribed_audio", meta={"transcript": transcription})

        if not transcription:
            return

        t_whisper = time()

        system_prompt = {
            "role": "system",
            "content": prompt("system-prompt"),
        }
        new_message = {"role": "user", "content": transcription}

        chat = (
            await db.execute(
                select(DBChatHistory).where(DBChatHistory.session_id == self.session_id)
            )
        ).scalar_one_or_none()
        if chat is None:
            chat = DBChatHistory(
                session_id=self.session_id, history_json={"messages": [system_prompt]}
            )
            db.add(chat)
        messages = chat.history_json["messages"]
        messages.append(new_message)
        response = await generate({"messages": messages}, [], "gpt-35-turbo-deployment")
        response_message = response.choices[0].message
        completion = response_message.content
        await log_event(db, self.session_id, "generated_completion", meta={"completion": completion})

        t_gpt = time()

        print(f"Used {response.usage.prompt_tokens} prompt tokens and {response.usage.completion_tokens} completion tokens")

        if "$ECHO" in completion:
            print("Echo detected, not sending response.")
            return

        messages.append(
            {"role": response_message.role, "content": completion}
        )
        chat.history_json["messages"] = messages
        await db.commit()

        normalized = normalize_text(response_message.content)
        await log_event(db, self.session_id, "normalized_text", meta={"normalized_text": normalized})
        t_normalize = time()
        sentences = re.split(r"(?<=[.!?]) +", normalized)
        for sentence in sentences:
            audio_chunk_bytes = await loop.run_in_executor(None, _inference, sentence)
            await log_event(db, self.session_id, "generated_tts", audio=np.frombuffer(audio_chunk_bytes, dtype=np.int16))
            await websocket.send_bytes(audio_chunk_bytes)

        t_styletts = time()

        print("Fastconformer", t_whisper - t0)
        print("GPT", t_gpt - t_whisper)
        print("Normalizer", t_normalize - t_gpt)
        print("StyleTTS2 + sending bytes", t_styletts - t_normalize)


def _check_for_exceptions(response_task: Optional[asyncio.Task]) -> bool:
    reset_state = False
    if response_task and response_task.done():
        try:
            response_task.result()
        except asyncio.CancelledError:
            print("response task was cancelled")
        except Exception as e:
            print("response task raised an exception:", e)
        else:
            print(
                "response task completed successfully. Resetting audio_data and response_task"
            )
            reset_state = True
        return reset_state


async def log_event(db: AsyncSession, session_id: str, event: EventName, meta: Optional[Dict[str, str]] = None, audio: Optional[np.ndarray] = None):
    if audio is not None:
        counter = 0
        path = f"/openduck-py/openduck-py/logs/{session_id}/{event}_{counter}.wav"
        session_folder = os.path.dirname(path)
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)
        base_path = path.rsplit("_", 1)[0]
        while os.path.exists(path):
            counter += 1
            path = f"{base_path}_{counter}.wav"

        wavfile.write(path, 16000, audio) 
        print(f"Wrote wavfile to {path}")
        meta = {"audio_url": path}
    record = DBChatRecord(
        session_id=session_id,
        event_name=event,
        meta_json=meta
    )
    db.add(record)
    await db.commit()


@audio_router.websocket("/response")
async def audio_response(
    websocket: WebSocket,
    session_id: str,
    record: bool = False,
    db: AsyncSession = Depends(get_db_async),
):
    await websocket.accept()
    await log_event(db, session_id, "started_session")

    vad = SileroVad()
    responder = ResponseAgent(session_id)
    recorder = WavAppender(wav_file_path=f"{session_id}.wav")

    audio_data = []
    response_task = None
    try:
        while True:
            if _check_for_exceptions(response_task):
                audio_data = []
                response_task = None
            try:
                message = await websocket.receive_bytes()
            except WebSocketDisconnect:
                print("websocket disconnected")
                return

            # NOTE(zach): Client records at 16khz sample rate.
            audio_16k_np = np.frombuffer(message, dtype=np.float32)

            await log_event(db, session_id, "received_audio", audio=audio_16k_np)

            audio_16k: torch.Tensor = torch.tensor(audio_16k_np)
            audio_data.append(audio_16k_np)
            if record:
                recorder.append(audio_16k_np)
            i = 0
            while i < len(audio_16k):
                upper = i + vad.window_size
                if len(audio_16k) - vad.window_size < upper:
                    upper = len(audio_16k)
                audio_16k_chunk = audio_16k[i:upper]
                vad_result = vad(audio_16k_chunk)
                if vad_result:
                    if "end" in vad_result:
                        print("end of speech detected.")
                        
                        await log_event(db, session_id, "detected_end_of_speech")
                        if response_task is None or response_task.done():
                            response_task = asyncio.create_task(
                                responder.start_response(
                                    websocket,
                                    db,
                                    np.concatenate(audio_data),
                                )
                            )
                        else:
                            print("already responding")
                    if "start" in vad_result:
                        print("start of speech detected.")
                        await log_event(db, session_id, "detected_start_of_speech")
                        if response_task and not response_task.done():
                            responder.interrupt(response_task)
                i = upper
    finally:
        recorder.close_file()

    # NOTE(zach): Consider adding a flag to do this rather than leaving it
    # commented, so we can save audio recorded on the server to make sure it
    # sounds right.
    # from scipy.io.wavfile import write
    # output_filename = "user_audio_response.wav"
    # sample_rate = 24000  # Assuming the sample rate is 16000
    # write(output_filename, sample_rate, audio_data)

    # TODO(zach): We never actually close it right now, we wait for the client
    # to close. But we should close it based on some timeout.
    await websocket.close()
    await log_event(db, session_id, "ended_session")
