import os

HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
EMB_MATCH_THRESHOLD = 0.5
WS_SAMPLE_RATE = 16000
IS_DEV = os.environ.get("IS_DEV", "False").lower() in [
    "true",
    "1",
]