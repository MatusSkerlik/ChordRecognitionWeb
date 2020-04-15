import os

from flask import (
    Blueprint, request, redirect, url_for, flash, current_app as app)
from werkzeug.utils import secure_filename

from chordify import Transcript

ALLOWED_MUSICAL_EXTENSIONS = {'mp3', 'wav', 'ogg'}

bp = Blueprint('upload', __name__, url_prefix='/upload')

transcript = Transcript()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_MUSICAL_EXTENSIONS


@bp.route('/', methods=['GET', 'POST'])
def upload_music():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            return str(transcript.from_audio(filepath))
    else:
        return "Wrong method"
