import logging
import os
import wave
from contextlib import contextmanager
from shutil import copyfileobj
from typing import Tuple

from .random import random_str


@contextmanager
def _wav_open(stream, mode="rb") -> wave.Wave_read:
    """ Opens file for header read (which will seek forward) and resets stream seek """
    try:
        with wave.open(stream, mode) as wav:
            yield wav
    finally:
        stream.seek(0)


def _save_file(stream, directory: str, filename: str, buffer_size=16384) -> str:
    """ Tries to save a file from stream and return filepath """
    try:
        # can throw error if directory exists, this is not expected
        os.makedirs(directory)
    except OSError:
        raise NameError('Directory already exists.')

    filepath = os.path.join(directory, filename)

    with open(filepath, "wb") as output:
        copyfileobj(stream, output, buffer_size)
        return filepath


def is_wav_file(stream):
    """ Checks file headers """
    try:
        with _wav_open(stream):
            return True
    except wave.Error as e:
        logging.getLogger(__name__).log(logging.INFO, e)
        return False


def check_sampling(stream, min_sr=22050):
    """ Checks file headers for sampling frequency """
    try:
        with _wav_open(stream) as wav:
            return wav.getframerate() >= min_sr
    except wave.Error as e:
        logging.getLogger(__name__).log(logging.INFO, e)
        return False


def save_music_file(stream, upload_dir: str, filename: str) -> Tuple[str, str]:
    """ Save music file into directory and returns new directory - filepath tuple """
    directory = random_str()
    return directory, _save_file(stream, os.path.join(upload_dir, directory), filename)
