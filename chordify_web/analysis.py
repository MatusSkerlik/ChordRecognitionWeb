import os
from typing import Sequence, Tuple

from flask import (
    Blueprint, current_app as app, session, g, render_template, url_for)
from werkzeug.exceptions import NotFound

from .chordify import get_configured_transcript

bp = Blueprint('analysis', __name__, url_prefix='/analysis')


def resolve_token(token: str):
    """ Resolve directory where audio file is saved """
    secret = session.get(token, None)
    return secret


def format_transcription(iterable: Sequence[Tuple[float, object]]):
    """ Format transcription as html for render in template """
    result = '<span><b>start stop chord</b></span>'
    start = 0.0
    for row in iterable:
        result += '<span>%f %f %s</span><br>' % (start, row[0], row[1])
        start = row[0]
    return result


def save_transcription(directory: str, filename: str, iterable: Sequence[Tuple[float, object]]) -> str:
    """ Save transcription iterator to a file and returns a relative path """
    result = ''
    start = 0.0
    for row in iterable:
        result += '%f %f %s\n' % (start, row[0], row[1])
        start = row[0]

    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as file:
        file.write(result)
        file.flush()
        return filepath


@bp.route('/<filename_token>', methods=['GET'])
def index(filename_token):
    token_dir = resolve_token(filename_token)
    if token_dir:
        directory = os.path.join(app.config['UPLOAD_DIR'], token_dir)
        if os.path.exists(directory) and os.path.isdir(directory):
            filepath = os.path.join(directory, app.config['AUDIO_FILE_NAME'])
            transcript = get_configured_transcript()
            iterable = transcript.from_audio(filepath)
            save_transcription(directory, app.config['TRANSCRIPTION_FILE_NAME'], iterable)
            g.transcription_url = url_for('download.index', filename_token=filename_token)
            g.transcription = format_transcription(iterable)
            return render_template("analysis.html")
    raise NotFound
