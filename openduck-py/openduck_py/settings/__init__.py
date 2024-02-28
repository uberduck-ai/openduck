import os

HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
WS_SAMPLE_RATE = 16000
EMB_MATCH_THRESHOLD = 0.5
IS_DEV = os.environ.get("IS_DEV", "False").lower() in [
    "true",
    "1",
]
