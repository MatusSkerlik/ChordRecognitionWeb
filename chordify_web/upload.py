import logging

from flask import (
    Blueprint, request, flash, redirect, url_for, render_template, current_app as app, session)

from .utils import is_wav_file, check_sampling, random_str, save_music_file, require_mime

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
                if is_wav_file(file):
                    if check_sampling(file):
                        secret, filepath = save_music_file(
                            file,
                            app.config["UPLOAD_DIR"],
                            app.config["AUDIO_FILE_NAME"]
                        )
                        return redirect(url_for('analysis.index', filename_token=generate_token(secret)))
                    else:
                        flash("File has wrong bitrate ( bitrate > 22050).")
                else:
                    flash("File is not valid wav file.")
            else:
                flash("No selected file.")
        else:
            flash("No file sent.")
        return render_template("upload_error.html"), 400
    return render_template("upload.html")
