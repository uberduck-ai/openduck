from io import BytesIO
import os
import azure.cognitiveservices.speech as speechsdk
import boto3
from google.cloud import texttospeech
from openai import OpenAI
from openduck_py.db import Session
from openduck_py.models import DBVoice
from openduck_py.utils.s3 import upload_fileobj, delete_object

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


def simplify_language(language: str):
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant that simplifies a language name like 'Indian English' to it's most basic form, 'English'. Return only a single word.
For example, if the user inputs 'Castilian Spanish', return 'Spanish'. If the user inputs 'Belgian French', return 'French'.""",
            },
            {"role": "user", "content": language},
        ],
    )
    return response.choices[0].message.content


def test_sentence(language: str):
    response = client.chat.completions.create(
        # model="gpt-4-deployment",
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates short test sentences in a given language to test a text-to-speech system.",
            },
            {
                "role": "user",
                "content": f"generate the following sentence in {language}: 'This is some test audio. I hope you like how it sounds.'",
            },
        ],
    )
    return response.choices[0].message.content


languages = set()

polly_client = boto3.client("polly", region_name="us-west-2")


def list_all_polly_voices():
    response = polly_client.describe_voices()
    return response["Voices"]


def add_all_polly_voices(voices, db):
    prefix = "polly_"
    for polly_voice in voices:
        print(polly_voice)
        voice_uuid = prefix + polly_voice["Id"]
        voice = db.execute(DBVoice.get(voice_uuid=voice_uuid)).first()
        if voice:
            continue
        languages.add(polly_voice["LanguageName"])

        simplified_language = simplify_language(polly_voice["LanguageName"]).lower()
        sentence = test_sentence(polly_voice["LanguageName"])
        print(f"Simplified language: {simplified_language}")
        print(f"Test sentence: {sentence}")

        polly_response = polly_client.synthesize_speech(
            OutputFormat="mp3",
            Text=sentence,
            VoiceId=polly_voice["Id"],
            LanguageCode=polly_voice["LanguageCode"],
            Engine=(
                "neural" if "neural" in polly_voice["SupportedEngines"] else "standard"
            ),
        )
        stream = polly_response["AudioStream"]
        delete_object(
            "uberduck-images", f"commercial/polly/{voice_uuid}/samples/{voice_uuid}.wav"
        )
        s3_upload_path = f"commercial/polly/{voice_uuid}/samples/{voice_uuid}.mp3"
        upload_fileobj(s3_upload_path, "uberduck-images", stream)

        dbvoice = DBVoice(
            name=voice_uuid,
            voice_uuid=voice_uuid,
            display_name=polly_voice["Name"],
            gender=polly_voice["Gender"],
            language=simplified_language.lower(),
            meta_json={
                "third_party": "polly",
                "SupportedEngines": polly_voice["SupportedEngines"],
                "Id": polly_voice["Id"],
                "LanguageName": polly_voice["LanguageName"],
                "LanguageCode": polly_voice["LanguageCode"],
            },
            sample_url="https://uberduck-images.s3.amazonaws.com/" + s3_upload_path,
        )
        db.add(dbvoice)
        db.commit()


def add_all_azure_voices(voices, db, speech_key, service_region):
    # Create an instance of a speech config with specified subscription key and service region.
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=service_region
    )

    # Create an instance of a speech synthesizer using the default speaker as audio output.
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    prefix = "azure_"
    for azure_voice in voices:
        print(azure_voice)
        voice_uuid = prefix + azure_voice.short_name
        voice = db.execute(DBVoice.get(voice_uuid=voice_uuid)).first()
        if voice:
            continue
        languages.add(azure_voice.locale)

        simplified_language = simplify_language(azure_voice.locale).lower()
        sentence = test_sentence(azure_voice.locale)
        print(f"Simplified language: {simplified_language}")
        print(f"Test sentence: {sentence}")

        # Generate speech with azure:
        speech_config.speech_synthesis_voice_name = azure_voice.short_name
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        result = synthesizer.speak_text(sentence)
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized to speaker for text [{}]".format(sentence))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(
                "There was an error with this voice: "
                + azure_voice.short_name
                + ". "
                + cancellation_details.reason.name
            )
            continue

        bio = BytesIO(result.audio_data)

        s3_upload_path = f"commercial/azure/{voice_uuid}/samples/{voice_uuid}.wav"
        upload_fileobj(s3_upload_path, "uberduck-images", bio)

        dbvoice = DBVoice(
            name=voice_uuid,
            voice_uuid=voice_uuid,
            display_name=azure_voice.local_name,
            gender=azure_voice.gender.name,
            language=simplified_language.lower(),
            meta_json={
                "third_party": "azure",
                "local_name": azure_voice.local_name,
                "locale": azure_voice.locale,
                "name": azure_voice.name,
                "short_name": azure_voice.short_name,
                "style_list": azure_voice.style_list,
                "voice_path": azure_voice.voice_path,
                "voice_type": azure_voice.voice_type.name,
            },
            sample_url="https://uberduck-images.s3.amazonaws.com/" + s3_upload_path,
        )
        db.add(dbvoice)
        db.commit()


def list_all_azure_tts_voices(speech_key, service_region):
    # Create an instance of a speech config with specified subscription key and service region.
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=service_region
    )
    # Create an instance of a speech synthesizer using the default speaker as audio output.
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    # Get the list of voices
    voices_result = synthesizer.get_voices_async().get()

    # Check if the operation was successful
    if voices_result.reason == speechsdk.ResultReason.VoicesListRetrieved:
        return voices_result.voices
    else:
        raise Exception(
            f"Could not retrieve the list of voices. Error: {voices_result.reason}"
        )


def list_all_google_tts_voices():
    # Initialize the client
    client = texttospeech.TextToSpeechClient()

    # Build the request for listing voices
    request = texttospeech.ListVoicesRequest()

    # Perform the list voices request
    response = client.list_voices(request=request)

    return response.voices


def add_all_google_voices(voices, db):
    # Iterate through the voices and print their details
    for voice in voices:
        # languages = ", ".join(voice.language_codes)
        # print(
        #     f"Name: {voice.name}, Languages: {languages}, Gender: {voice.ssml_gender.name}, Natural Sample Rate Hertz: {voice.natural_sample_rate_hertz}"
        # )
        voice_uuid = "google_" + voice.name
        dbvoice = db.execute(DBVoice.get(voice_uuid=voice_uuid)).first()
        if dbvoice:
            continue
        print(voice)
        simplified_language = simplify_language(voice.language_codes[0]).lower()
        sentence = test_sentence(voice.language_codes[0])

        genders = {
            "MALE": "Male",
            "FEMALE": "Female",
        }

        client = texttospeech.TextToSpeechClient()

        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=sentence)

        vsp = texttospeech.VoiceSelectionParams(
            language_code=voice.language_codes[0], name=voice.name
        )

        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Perform the text-to-speech request on the text input with the selected voice parameters and audio file type
        response = client.synthesize_speech(
            input=synthesis_input, voice=vsp, audio_config=audio_config
        )
        bio = BytesIO(response.audio_content)
        s3_upload_path = f"commercial/azure/{voice_uuid}/samples/{voice_uuid}.mp3"
        upload_fileobj(s3_upload_path, "uberduck-images", bio)

        dbvoice = DBVoice(
            name=voice_uuid,
            voice_uuid=voice_uuid,
            display_name=voice.name,
            gender=genders.get(voice.ssml_gender.name),
            language=simplified_language.lower(),
            meta_json={
                "third_party": "google",
                "language_codes": list(voice.language_codes),
                "name": voice.name,
                "ssml_gender": voice.ssml_gender.name,
                "natural_sample_rate_hertz": voice.natural_sample_rate_hertz,
            },
            sample_url="https://uberduck-images.s3.amazonaws.com/" + s3_upload_path,
        )
        db.add(dbvoice)
        db.commit()


if __name__ == "__main__":
    db = Session()
    voices = list_all_polly_voices()
    add_all_polly_voices(voices, db)

    speech_key = os.getenv("AZURE_SPEECH_KEY")
    service_region = "westus"
    voices = list_all_azure_tts_voices(speech_key, service_region)
    add_all_azure_voices(voices, db, speech_key, service_region)

    voices = list_all_google_tts_voices()
    add_all_google_voices(voices, db)
