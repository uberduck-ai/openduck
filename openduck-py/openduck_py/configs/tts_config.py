from typing import Literal, Optional

TTSProviders = Literal["styletts2", "gptsovits", "elevenlabs", "openai"]


class TTSConfig:
    def __init__(
        self, provider: TTSProviders = "gptsovits", voice_id: Optional[str] = "/openduck-py/openduck-py/models/styletts2/cartoon-boy-upbeat.wav"
    ):
        self.provider = provider
        self.voice_id = voice_id
