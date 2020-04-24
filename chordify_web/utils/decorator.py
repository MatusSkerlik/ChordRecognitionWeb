import logging
from functools import wraps

from flask import request, redirect, flash


def require_mime(mimetype, method='POST'):
    def decorator(func):
        """
        Decorator which returns a 415 Unsupported Media Type if the client sends
        something other than a certain mimetype
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            if request.method == method:
                if request.mimetype != mimetype:
                    flash("Request must be multipart/form-data")
                    return redirect(request.url)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def suppress_exception(log_level=logging.ERROR):
    def decorator(func):
        """ Suppress error, log and return None """

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(log_level, e)
                return None

        return wrapper

    return decorator
