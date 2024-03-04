import os

HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
EMB_MATCH_THRESHOLD = 0.5
WS_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
IS_DEV = os.environ.get("IS_DEV", "False").lower() in [
    "true",
    "1",
]
CHUNK_SIZE = 1024
