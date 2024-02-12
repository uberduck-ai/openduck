import os
import subprocess
import time

from fastapi import APIRouter, HTTPException
import requests

from openduck_py.clients.daily_call_client import get_meeting_token

call_router = APIRouter(prefix="/call")


@call_router.post("/start")
def start_call():
    daily_api_key = os.getenv("DAILY_API_KEY")
    api_path = os.getenv("DAILY_API_PATH") or "https://api.daily.co/v1"

    timeout = int(os.getenv("BOT_MAX_DURATION") or 300)
    exp = time.time() + timeout
    res = requests.post(
        f"{api_path}/rooms",
        headers={"Authorization": f"Bearer {daily_api_key}"},
        json={
            "properties": {
                "exp": exp,
                "enable_chat": True,
                "enable_emoji_reactions": True,
                "eject_at_room_exp": True,
                "enable_prejoin_ui": False,
            }
        },
    )
    if res.status_code != 200:
        raise HTTPException(
            status_code=500, detail=f"Unable to create room: {res.text}"
        )
    room_url = res.json()["url"]
    room_name = res.json()["name"]

    meeting_token = get_meeting_token(room_name, daily_api_key, exp)

    _path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../clients/daily_call_client.py"
    )
    proc = subprocess.Popen(
        [f"python {_path} -u {room_url} -t {meeting_token}"],
        shell=True,
        bufsize=1,
    )

    # Don't return until the bot has joined the room, but wait for at most 2 seconds.
    attempts = 0
    while attempts < 20:
        time.sleep(0.1)
        attempts += 1
        res = requests.get(
            f"{api_path}/rooms/{room_name}/get-session-data",
            headers={"Authorization": f"Bearer {daily_api_key}"},
        )
        if res.status_code == 200:
            break
    print(f"Took {attempts} attempts to join room {room_name}")

    return {"room_url": room_url, "token": meeting_token}
