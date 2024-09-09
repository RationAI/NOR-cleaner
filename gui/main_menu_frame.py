"""
Main menu window.
"""

import logging

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from gui import TrainFrame
from gui.error_box import ErrorBox
from gui.error_wrapper import button_error_wrapper

logger = logging.getLogger(__name__)

BUTTON_WIDTH = 20


class MainMenuFrame(ttk.Frame):
    """
    Main menu window class.
    Consists of the following widgets:
    - Title label
    - Train model button
    - Predict button
    - Exit button
    """

    def __init__(self, master, *args, **kwargs):
        super().__init__(master, padding=15, *args, **kwargs)
        self.pack(fill=BOTH, expand=YES)

        self.title_label = ttk.Label(
            self, text="Main Menu", style="primary.TLabel"
        )
        self.title_label.pack(pady=10)

        self.create_train_button()
        self.create_predict_button()
        self.create_exit_button()
    
    def create_train_button(self):
        self.train_button = ttk.Button(
            self,
            text="Train model",
            style="primary.TButton",
            command=button_error_wrapper(self._on_train, logger=logger),
            width=BUTTON_WIDTH,
        )
        self.train_button.pack(pady=10)

    def _on_train(self):
        """
        When the user clicks the "Train model" button.
        The window is switched to the TrainWindow.
        """
        self.master.switch_frame(TrainFrame)

    def create_predict_button(self):
        self.predict_button = ttk.Button(
            self, text="Predict", style="primary.TButton", width=BUTTON_WIDTH
        )
        self.predict_button.pack(pady=10)

    def create_exit_button(self):
        self.exit_button = ttk.Button(
            self,
            text="Exit",
            style="danger.TButton",
            command=self._quit_app,
            width=BUTTON_WIDTH,
        )
        self.exit_button.pack(pady=10)

    def _quit_app(self):
        logger.info("Quitting the application")
        self.master.destroy()


if __name__ == "__main__":
    root = ttk.Window()

    main_menu = MainMenuFrame(root)
    main_menu.pack(fill=BOTH, expand=YES)

    logging.basicConfig(level=logging.DEBUG)

    root.mainloop()
