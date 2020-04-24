import wave


def is_wav_file(file, min_sr=22050):
    try:
        with wave.open(file, "rb") as wav:
            return wav.getframerate() >= min_sr
    except wave.Error:
        return False
