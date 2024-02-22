import asyncio
import re
from tempfile import NamedTemporaryFile
from time import time

# NOTE(zach): On Mac OS, the first import fails, but the subsequent one
# succeeds. /shrug.
try:
    import nemo.collections.asr.models as asr_models
except OSError:
    import nemo.collections.asr.models as asr_models


from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from nemo_text_processing.text_normalization.normalize import Normalizer
import numpy as np
from scipy.io import wavfile
from sqlalchemy import select
import torch
from torchaudio.functional import resample

from openduck_py.models import DBChatHistory
from openduck_py.db import get_db_async, AsyncSession
from openduck_py.voices import styletts2
from openduck_py.routers.templates import generate

asr_model = asr_models.EncDecCTCModelBPE.from_pretrained(
    model_name="nvidia/stt_en_fastconformer_ctc_large"
)
normalizer = Normalizer(input_case="cased", lang="en")

audio_router = APIRouter(prefix="/audio")


def _transcribe(audio_data):
    with NamedTemporaryFile(suffix=".wav", mode="wb+") as temp_file:
        wavfile.write(temp_file.name, 16000, audio_data)
        temp_file.flush()
        temp_file.seek(0)
        transcription = asr_model.transcribe([temp_file.name])[0]
    return transcription


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
    is_responding = False

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
        session_id: str,
    ):
        print("starting response")
        self.is_responding = True

        def _inference(sentence: str):
            audio_chunk = styletts2.styletts2_inference(text=sentence)
            audio_chunk_bytes = np.int16(audio_chunk * 32767).tobytes()
            return audio_chunk_bytes

        loop = asyncio.get_running_loop()

        t0 = time()

        print("RUNNING TRANSCRIBE IN EXECUTOR")
        transcription = await loop.run_in_executor(None, _transcribe, audio_data)
        print("transcription", transcription)

        if not transcription:
            return

        t_whisper = time()

        system_prompt = {
            "role": "system",
            "content": "You are a children's toy which can answer educational questions. You want to help your user and support them. Give short concise responses no more than 2 sentences.",
        }
        new_message = {"role": "user", "content": transcription}

        chat = (
            await db.execute(
                select(DBChatHistory).where(DBChatHistory.session_id == session_id)
            )
        ).scalar_one_or_none()
        if chat is None:
            chat = DBChatHistory(
                session_id=session_id, history_json={"messages": [system_prompt]}
            )
            db.add(chat)

        messages = chat.history_json["messages"]
        messages.append(new_message)
        response = await generate({"messages": messages}, [], "gpt-35-turbo-deployment")

        t_gpt = time()

        response_message = response.choices[0].message

        messages.append(
            {"role": response_message.role, "content": response_message.content}
        )
        chat.history_json["messages"] = messages
        await db.commit()

        def normalize_text(text):
            normalized_text = normalizer.normalize(text)
            print("Original response:", text)
            print("Normalized response:", normalized_text)
            return normalized_text

        normalized = normalize_text(response_message.content)
        sentences = re.split(r"(?<=[.!?]) +", normalized)
        for sentence in sentences:
            print("RUNNING TTS IN EXECUTOR")
            audio_chunk_bytes = await loop.run_in_executor(None, _inference, sentence)
            print("DONE RUNNING IN EXECUTOR")
            await websocket.send_bytes(audio_chunk_bytes)
            print("DONE SENDING BYTES")

        t_styletts = time()

        print("Fastconformer", t_whisper - t0)
        print("GPT", t_gpt - t_whisper)
        print("StyleTTS2", t_styletts - t_gpt)


def _transcribe(audio_data: np.ndarray):
    t = torch.tensor(audio_data)
    if torch.cuda.is_available():
        t = t.cuda()
    resampled = resample(t, orig_freq=24000, new_freq=16000)
    return whisper_model.transcribe(resampled)["text"]


SILENCE_THRESHOLD = 1.0


@audio_router.websocket("/response")
async def audio_response(
    websocket: WebSocket,
    session_id: str,
    db: AsyncSession = Depends(get_db_async),
):
    await websocket.accept()

    vad = SileroVad()
    responder = ResponseAgent()

    audio_data = []
    response_task = None
    while True:
        # Check for exceptions in response_task
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
                audio_data = []
                response_task = None
        try:
            message = await websocket.receive_bytes()
        except WebSocketDisconnect:
            print("websocket disconnected")
            return
        print("got a message.")

        # NOTE(zach): Client records at 22khz sample rate.
        audio_chunk_24k = np.frombuffer(message, dtype=np.float32)
        audio_16k: torch.Tensor = resample(
            torch.tensor(audio_chunk_24k), orig_freq=24000, new_freq=16000
        )
        audio_data.append(audio_chunk_24k)
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
                    if response_task is None or response_task.done():
                        response_task = asyncio.create_task(
                            responder.start_response(
                                websocket,
                                db,
                                np.concatenate(audio_data),
                                session_id,
                            )
                        )
                    else:
                        print("already responding")
                if "start" in vad_result:
                    print("start of speech detected.")
                    if response_task and not response_task.done():
                        responder.interrupt(response_task)
            i = upper

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
