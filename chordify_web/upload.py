import logging

from flask import (
    Blueprint, request, flash, redirect, url_for, render_template, current_app as app, session)
from werkzeug.exceptions import InternalServerError

from chordify_web.utils import is_wav_file, check_sampling, random_str
from chordify_web.utils.decorator import require_mime
from chordify_web.utils.file import save_music_file

logger = logging.getLogger(__name__)
bp = Blueprint('upload', __name__, url_prefix='/upload')


def generate_token(secret):
    """ Generate token and saves it in globals """
    _token = random_str()
    session[_token] = secret
    return _token


@bp.route('/', methods=['GET', 'POST'])
@require_mime("multipart/form-data", 'POST')
def index():
    if request.method == 'POST':
        # check if the post request has the file part

        if 'file' in request.files:
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file and file.filename != '':
                if is_wav_file(file) and check_sampling(file):
                    filepath = save_music_file(file, app.config["UPLOAD_DIR"])
                    return redirect(url_for('analysis.index', filename_token=generate_token(filepath)))
                else:
                    flash("File is not valid wav file.")
            else:
                flash("No selected file.")
        else:
            flash("No file sent.")
        return redirect(request.url)
    return render_template("upload.html")


@bp.errorhandler(InternalServerError)
def handle_500(e):
    return "Unexpected error", 500
