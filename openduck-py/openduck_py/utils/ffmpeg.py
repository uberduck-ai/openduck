import subprocess


def chop_video(input_file: str, start_time: float, end_time: float):
    pass 

def convert_seconds_to_time(seconds: float, format: Literal["srt", "ffmpeg"]):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60

    if format == "ffmpeg":
        return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}"
    elif format == "srt":
        milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
        return (
            f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"
        )
    else:
        raise ValueError(
            "Unsupported format. Supported formats are 'srt' and 'ffmpeg'."
        )


def sentences_to_srt(sentences, output_file_name: str):
    with open(output_file_name, "w") as file:
        for i, sentence in enumerate(sentences, start=1):
            start_time = convert_seconds_to_time(sentence["start"], format="srt")
            end_time = convert_seconds_to_time(sentence["end"], format="srt")
            file.write(f"{i}\n")
            file.write(f"{start_time} --> {end_time}\n")
            file.write(f"{sentence['text']}\n\n")
    return output_file_name