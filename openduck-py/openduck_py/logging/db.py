import asyncio
import os
from typing import Optional, Dict
from pathlib import Path
from time import time

import numpy as np
from scipy.io import wavfile
import boto3

from openduck_py.db import AsyncSession
from openduck_py.models.chat_record import EventName, DBChatRecord
from openduck_py.settings import (
    WS_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    AUDIO_UPLOAD_BUCKET,
    LOG_TO_S3,
)


async def log_event(
    db: AsyncSession,
    session_id: str,
    event: EventName,
    meta: Optional[Dict[str, str]] = None,
    audio: Optional[np.ndarray] = None,
    latency: Optional[float] = None,
):
    if audio is not None:
        log_path = f"logs/{session_id}/{event}_{time()}.wav"
        abs_path = Path(__file__).resolve().parents[2] / log_path
        session_folder = abs_path.parent
        if not session_folder.exists():
            session_folder.mkdir(parents=True)

        sample_rate = WS_SAMPLE_RATE
        if event == "generated_tts":
            sample_rate = OUTPUT_SAMPLE_RATE

        # Use scipy's write function in an executor to run it in a thread
        await asyncio.get_running_loop().run_in_executor(
            None, wavfile.write, abs_path, sample_rate, audio
        )
        print(f"Wrote wavfile to {abs_path}")

        if LOG_TO_S3:
            # Use boto3's upload_file function in an executor
            s3_client = boto3.client("s3")
            await asyncio.get_running_loop().run_in_executor(
                None,
                s3_client.upload_file,
                str(abs_path),
                AUDIO_UPLOAD_BUCKET,
                log_path,
            )
            print(f"Uploaded wavfile to s3://{AUDIO_UPLOAD_BUCKET}/{log_path}")

        meta = {"audio_url": log_path}
    record = DBChatRecord(
        session_id=session_id,
        event_name=event,
        meta_json=meta,
        latency_seconds=latency,
    )
    db.add(record)
    await db.commit()
