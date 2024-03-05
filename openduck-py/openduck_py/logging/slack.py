import os
import requests
from openduck_py.settings import IS_DEV
from openduck_py.utils.s3 import s3_client

LOGGING_BUCKET = os.environ.get("OUTPUT_BUCKET")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_LOGS_CHANNEL_ID = os.environ.get("SLACK_LOGS_CHANNEL_ID")


def log_audio_to_slack(audio_path):

    s3_key = audio_path.replace("//", "_")
    s3_client.upload_file(audio_path, LOGGING_BUCKET, s3_key)
    url = f"https://{LOGGING_BUCKET}.s3.amazonaws.com/{s3_key}"
    requests.post(
        "https://slack.com/api/chat.postMessage",
        params={
            "channel": SLACK_LOGS_CHANNEL_ID,
            "text": f"Dev mode: {IS_DEV} \n Audio recording: {url}",
        },
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
    )
