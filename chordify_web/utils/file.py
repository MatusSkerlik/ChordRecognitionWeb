import logging
import os
import wave
from contextlib import contextmanager

from .random import random_str


@contextmanager
def _wav_open(stream, mode="rb"):
    # Code to acquire resource, e.g.:
    try:
        with wave.open(stream, mode) as wav:
            yield wav
    finally:
        stream.seek(0)


def _save_file(stream, dst: str, buffer_size=16384) -> str:
    from shutil import copyfileobj

    _path = dst.split(os.sep)[:-1]
    for i, _dir in enumerate(_path):
        _sub_path = os.path.join(*_path[:i+1])
        if not os.path.exists(_sub_path):
            os.mkdir(_sub_path)

    _dst = open(dst, "wb")
    try:
        copyfileobj(stream, _dst, buffer_size)
        return dst
    finally:
        _dst.close()


def is_wav_file(stream):
    try:
        with _wav_open(stream):
            return True
    except wave.Error as e:
        logging.getLogger(__name__).log(logging.INFO, e)
        return False


def check_sampling(stream, min_sr=22050):
    try:
        with _wav_open(stream) as wav:
            return wav.getframerate() >= min_sr
    except wave.Error as e:
        logging.getLogger(__name__).log(logging.INFO, e)
        return False


def save_music_file(stream, root_dir: str) -> str:
    _path = os.path.join(root_dir, random_str(), random_str())
    return _save_file(stream, _path)
