import os

from flask import Flask

from chordify_web.logging import setup_logging


def create_app(test_config=None):
    # create and configure the app

    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
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

    from . import base, upload, analysis
    app.register_blueprint(base.bp)
    app.register_blueprint(upload.bp)
    app.register_blueprint(analysis.bp)

    return app
