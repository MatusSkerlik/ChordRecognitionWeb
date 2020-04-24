import os

from flask import (
    Blueprint, request, flash, redirect, current_app as app, url_for, render_template)
from werkzeug.exceptions import InternalServerError
from werkzeug.utils import secure_filename

from chordify_web.utils.decorator import require_mime
from chordify_web.utils.filetype import is_wav_file

bp = Blueprint('upload', __name__, url_prefix='/upload')


@bp.route('/', methods=['GET', 'POST'])
@require_mime("multipart/form-data", 'POST')
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file sent.')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file.')
            return redirect(request.url)
        if file and is_wav_file(file):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_DIR'], filename))
            return redirect(url_for('analysis.index',
                                    filename=filename))
    return render_template("upload.html")


@bp.errorhandler(InternalServerError)
def handle_500(e):
    return None, 500
