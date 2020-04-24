from flask import (
    Blueprint, url_for, redirect)

bp = Blueprint('base', __name__, url_prefix=None)


@bp.route('/', methods=['GET'])
def index():
    return redirect(url_for("upload.index"))