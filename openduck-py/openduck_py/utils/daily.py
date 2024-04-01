import asyncio
import os
import time
from typing import Optional
import aiofiles
import tempfile

from daily import EventHandler, CallClient
import httpx
from pydantic import BaseModel

from openduck_py.utils.s3 import upload_to_s3_bucket
from openduck_py.settings import RECORDING_UPLOAD_BUCKET
from openduck_py.db import DBChatRecording

DAILY_API_KEY = os.environ.get("DAILY_API_KEY")


class RoomCreateResponse(BaseModel):
    url: str
    name: str
    privacy: str


async def start_recording(room_url: str) -> Optional[str]:
    daily_recording_id = None
    NUM_ATTEMPTS = 10
    async with httpx.AsyncClient() as _http_client:
        room_id = room_url.split("/")[-1]
        print(f"Room ID: {room_id}")
        for attempt in range(3):
            _recording_response = await _http_client.post(
                f"https://api.daily.co/v1/rooms/{room_id}/recordings/start",
                headers={"Authorization": f"Bearer {os.environ['DAILY_API_KEY']}"},
            )
            if _recording_response.status_code == 404 and attempt < NUM_ATTEMPTS:
                await asyncio.sleep(0.1)  # Sleep for 100ms before retrying
            else:
                _recording_response.raise_for_status()
                daily_recording_id = _recording_response.json()["recordingId"]
                break
    return daily_recording_id


async def stop_and_download_recording(room_id: str, recording_id: str) -> str:
    async with httpx.AsyncClient() as _http_client:
        _recording_response = await _http_client.post(
            f"https://api.daily.co/v1/rooms/{room_id}/recordings/stop",
            headers={"Authorization": f"Bearer {os.environ['DAILY_API_KEY']}"},
        )
        start_time = time.time()
        recording_status = ""
        while recording_status != "finished":
            if time.time() - start_time > 10:
                print("Recording status not finished after 10 seconds.")
                break
            await asyncio.sleep(0.5)
            recording_status_response = await _http_client.get(
                f"https://api.daily.co/v1/recordings/{recording_id}",
                headers={"Authorization": f"Bearer {os.environ['DAILY_API_KEY']}"},
            )
            recording_status = recording_status_response.json().get("status")

        if recording_status == "finished":
            access_link_response = await _http_client.get(
                f"https://api.daily.co/v1/recordings/{recording_id}/access-link",
                headers={"Authorization": f"Bearer {os.environ['DAILY_API_KEY']}"},
            )
            file_url = access_link_response.json().get("download_link")

            resp = await _http_client.get(file_url)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                async with aiofiles.open(tmp_file.name, "wb") as out_file:
                    await out_file.write(resp.content)
                downloaded_file_path = tmp_file.name

        # Open the downloaded file and upload it to the specified S3 bucket
        async with aiofiles.open(downloaded_file_path, "rb") as file_to_upload:
            s3_filename = f"recordings/{room_id}/{recording_id}.mp4"
            await upload_to_s3_bucket(
                file_to_upload, RECORDING_UPLOAD_BUCKET, s3_filename
            )
            print(f"Uploaded recording to s3://{RECORDING_UPLOAD_BUCKET}/{s3_filename}")

        # DBChatRecording.create(
        #     chat_session_id=room_id,
        #     url=f"s3://{RECORDING_UPLOAD_BUCKET}/{s3_filename}",
        # )

        return downloaded_file_path


async def create_room(exp=None) -> dict:
    headers = {
        "Authorization": f"Bearer {DAILY_API_KEY}",
        "Content-Type": "application/json",
    }
    if exp is None:
        # Default room lifetime is 1 hour.
        exp = int(time.time()) + 3600
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.daily.co/v1/rooms",
            headers=headers,
            json={
                "properties": {
                    "enable_chat": True,
                    "start_video_off": True,
                    "start_audio_off": False,
                    "exp": exp,
                }
            },
        )
    response.raise_for_status()
    return response.json()


class CustomEventHandler(EventHandler):
    def __init__(self):
        self.client = CallClient(event_handler=self)
        self.left = False

    def _leave_callback(self, *args, **kwargs):
        self.left = True
        self.client.release()

    def on_active_speaker_change(self, participant):
        print("Active speaker change", participant)

    def on_participant_counts_updated(self, counts):
        print("Participant counts updated", counts)
        print(self.client.participants())

    def on_participant_left(self, participant, reason):
        print("Participant left", participant, reason)
        print(self.client.participants())
        participants = self.client.participants()
        if (
            len(
                list(
                    filter(
                        lambda x: not x["info"]["userName"].endswith(" (AI)"),
                        participants.values(),
                    )
                )
            )
            == 0
        ):
            print("Last participant left, ending the call")
            self.client.leave(completion=self._leave_callback)
