import logging

from gui.error_box import ErrorBox


def on_event_error_wrapper(logger: logging.Logger | None = None):
    """
    Wrap function to catch exceptions and show them in the ErrorBox.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                if logger is not None:
                    logger.exception(e)
                ErrorBox.from_exception(e)

        return wrapper

    return decorator
