import os
import requests

from slack_sdk import WebClient

from openduck_py.settings import IS_DEV

LOGGING_BUCKET = os.environ.get("OUTPUT_BUCKET")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_LOGS_CHANNEL_ID = os.environ.get("SLACK_LOGS_CHANNEL")

slack_client = WebClient(token=SLACK_BOT_TOKEN)


def log_audio_to_slack(audio_path, remote_path):
    print("log_audio_to_slack", audio_path, LOGGING_BUCKET, SLACK_LOGS_CHANNEL_ID)
    assert os.path.exists(audio_path), f"{audio_path} does not exist"

    result = slack_client.files_upload(
        channels=[SLACK_LOGS_CHANNEL_ID],
        file=audio_path,
        title=f"{'[dev] ' if IS_DEV else ''}Call recording",
        initial_coment="yo this is my recording",
    )
