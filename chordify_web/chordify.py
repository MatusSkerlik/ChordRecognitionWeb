""" This module provides factories for chordify package """
from flask import current_app as app

from chordify.app import TranscriptBuilder, Transcript


def get_default_transcript() -> Transcript:
    """ Returns Transcript object in default state. """
    return TranscriptBuilder.default()


def get_configured_transcript() -> Transcript:
    """ Returns Transcript object configured via app config. """
    return TranscriptBuilder(app.config).build()
