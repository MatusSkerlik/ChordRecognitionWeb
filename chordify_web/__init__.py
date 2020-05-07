import os

from flask import Flask, redirect, url_for, render_template
from flask_session import Session
from werkzeug.exceptions import NotFound, InternalServerError, Unauthorized

from chordify_web.logging import setup_logging


def create_app(test_config=None):
    # create and configure the app

    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=False)
    else:
        # load the tests config if passed in
        app.config.from_mapping(test_config)

    # setup logging after config was loaded
    setup_logging(config=app.config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import upload, analysis, download

    app.register_blueprint(upload.bp)
    app.register_blueprint(analysis.bp)
    app.register_blueprint(download.bp)

    @app.route('/')
    def run():
        return redirect(url_for('upload.index'))

    @app.errorhandler(Unauthorized)
    def handle_401(e):
        return render_template("error401.html"), 401

    @app.errorhandler(NotFound)
    def handle_404(e):
        return render_template("error404.html"), 404

    @app.errorhandler(InternalServerError)
    def handle_500(e):
        return render_template("error500.html"), 500

    Session(app)
    return app
