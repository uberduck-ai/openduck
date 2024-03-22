import os
import time
import sys

from daily import EventHandler, CallClient
import httpx
from pydantic import BaseModel

DAILY_API_KEY = os.environ.get("DAILY_API_KEY")


class RoomCreateResponse(BaseModel):
    url: str
    name: str
    privacy: str


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
        sys.exit()

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
