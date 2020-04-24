from chordify import Transcript

""" This module provides factories for Transcript and Learner object from chordify package """


def get_default_transcript(config=None):
    return Transcript(config)
