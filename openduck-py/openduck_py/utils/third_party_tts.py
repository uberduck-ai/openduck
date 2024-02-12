import logging
from openduck_py.utils.s3 import async_session, upload_to_s3_bucket

log = logging.getLogger(__name__)


def is_third_party_voice(voice: str) -> bool:
    return (
        voice.startswith("polly_")
        or voice.startswith("azure_")
        or voice.startswith("google_")
    )


# POLLY


async def aio_polly_tts(
    text: str,
    voice_id: str,
    language_code: str,
    engine: str,
    upload_path: str,
    output_format: str = "mp3",
):
    # Credentials for AWS are loaded from the environment or AWS credentials file
    async with async_session().client("polly", region_name="us-west-2") as polly:
        # Calling Polly to synthesize speech
        response = await polly.synthesize_speech(
            Text=text,
            VoiceId=voice_id,
            OutputFormat=output_format,
            LanguageCode=language_code,
            Engine=engine,
        )

        if upload_path:
            # Upload the audio file to S3
            await upload_to_s3_bucket(
                response["AudioStream"], "uberduck-audio-outputs", upload_path
            )
        return response
