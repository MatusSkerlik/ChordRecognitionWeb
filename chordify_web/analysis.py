import os

from flask import (
    Blueprint, render_template, redirect, url_for, current_app as app)

from chordify_web.chordify import get_default_transcript

bp = Blueprint('analysis', __name__, url_prefix='/analysis')


@bp.route('/<filename>', methods=['GET'])
def index(filename):
    if filename:
        transcript = get_default_transcript(app.config)
        return str(transcript.from_audio(os.path.join(app.config["UPLOAD_DIR"], filename)))
    return str(filename)
