import os

from flask import (
    Blueprint, send_from_directory, current_app as app, session)
from werkzeug.exceptions import NotFound

bp = Blueprint('download', __name__, url_prefix='/download')


def _resolve_token(token: str):
    """ Resolve directory where audio file is saved """
    secret = session.get(token, None)
    return secret


@bp.route('/<filename_token>', methods=['GET'])
def index(filename_token):
    token_dir = _resolve_token(filename_token)
    if token_dir:
        directory = os.path.join(app.config['UPLOAD_DIR'], token_dir)
        filename = app.config['TRANSCRIPTION_FILE_NAME']
        return send_from_directory(directory, filename, as_attachment=True)
    raise NotFound
