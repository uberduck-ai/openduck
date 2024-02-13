import io
import os
import logging
from openduck_py.utils.s3 import async_session, upload_to_s3_bucket
import azure.cognitiveservices.speech as azure_speechsdk
from google.cloud import texttospeech
from asgiref.sync import sync_to_async

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


# GOOGLE


def google_tts(text: str, voice_name: str, language_code: str):
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request, select the language code and the ssml voice gender
    vsp = texttospeech.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=vsp, audio_config=audio_config
    )
    return response


aio_google_tts = sync_to_async(google_tts)

# AZURE


async def aio_azure_tts(text: str, voice_name: str, upload_path: str):
    speech_config = azure_speechsdk.SpeechConfig(
        subscription=os.environ["AZURE_SPEECH_KEY"], region="westus"
    )
    speech_config.speech_synthesis_voice_name = voice_name

    # Create an instance of a speech synthesizer using the default speaker as audio output.
    synthesizer = azure_speechsdk.SpeechSynthesizer(speech_config=speech_config)

    def _speak_text_async():
        return synthesizer.speak_text(text)

    result = await sync_to_async(_speak_text_async)()
    if result.reason == azure_speechsdk.ResultReason.SynthesizingAudioCompleted:
        bio = io.BytesIO(result.audio_data)
        await upload_to_s3_bucket(bio, "uberduck-audio-outputs", upload_path)
        return upload_path
    elif result.reason == azure_speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        if cancellation_details.reason == azure_speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                log.error(f"Error details: {cancellation_details.error_details}")
                raise Exception(
                    "Speech synthesis canceled: " + cancellation_details.error_details
                )
