import logging

from gui.error_box import ErrorBox


def button_error_wrapper(func, logger: logging.Logger | None = None):
    """
    Wrap function to catch exceptions and show them in the ErrorBox.
    """

    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            if logger is not None:
                logger.exception(e)
            ErrorBox.from_exception(e)

    return wrapper
