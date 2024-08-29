"""
Error box to show error messages.
"""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import ttkbootstrap.dialogs.dialogs as dialogs

"""
ErrorBox class for showing error messages.
It is a messagebox with a title and message.
"""
class ErrorBox():
    def __init__(self, title, message):
        self.title = title
        self.message = message
        self.error_box = dialogs.MessageDialog(
            title=self.title,
            message=self.message,
            style="danger",
            buttons=["OK"],
            default="OK",
            alert=True,
        )
        self.error_box.show()

    @classmethod
    def from_exception(cls, exception: Exception):
        title = exception.__class__.__name__
        message = str(exception)
        return cls(title, message)