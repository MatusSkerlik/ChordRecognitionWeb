import os

from flask import (
    Blueprint, current_app as app, session)
from werkzeug.exceptions import InternalServerError

from chordify_web.chordify import get_default_transcript

bp = Blueprint('analysis', __name__, url_prefix='/analysis')


def resolve_token(token: str):
    secret = session.get(token, None)
    return secret


@bp.route('/<filename_token>', methods=['GET'])
def index(filename_token):
    if filename_token:
        filepath = resolve_token(filename_token)
        if filepath and os.path.exists(filepath):
            transcript = get_default_transcript(app.config)
            return str(transcript.from_audio(filepath))
    return "Token supplied is not valid.", 500


@bp.errorhandler(InternalServerError)
def handle_500(e):
    return "Unexpected error", 500
