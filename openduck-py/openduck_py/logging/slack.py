import os
import requests
from openduck_py.settings import IS_DEV
from openduck_py.utils.s3 import s3_client

LOGGING_BUCKET = os.environ.get("OUTPUT_BUCKET")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_LOGS_CHANNEL_ID = os.environ.get("SLACK_LOGS_CHANNEL")


_client = s3_client()  # NOTE (Sam): should we move this to utils/s3?


def log_audio_to_slack(audio_path):
    _client.upload_file(audio_path, LOGGING_BUCKET, audio_path)
    url = f"https://{LOGGING_BUCKET}.s3.amazonaws.com/{audio_path}"
    requests.post(
        "https://slack.com/api/chat.postMessage",
        params={
            "channel": SLACK_LOGS_CHANNEL_ID,
            "text": f"Dev mode: {IS_DEV} \n Audio recording: {url}",
        },
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
    )
