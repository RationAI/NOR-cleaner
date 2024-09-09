import logging
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from gui.main_menu_frame import MainMenuFrame


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MenuApplication(ttk.Window):
    def __init__(self):
        super().__init__()

        # Set size of the window
        self.geometry("800x400")

        # Set title of the window
        self.title("NOR Cleaner")

        self.current_frame = None
        self.switch_frame(MainMenuFrame)


    def switch_frame(self, frame_class: ttk.Frame) -> None:
        """Destroy current frame and replace it with a new one"""
        new_frame = frame_class(self)
        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = new_frame
        self.current_frame.pack(fill=BOTH, expand=YES)
    
        # Create "Back to main menu" button if the frame is not MainMenuFrame
        if frame_class is not MainMenuFrame:
            self._back_to_menu_button = ttk.Button(
                self.current_frame,
                text="Back to main menu",
                style="primary.TButton",
                command=self._on_back_to_menu,
            )
            self._back_to_menu_button.pack(pady=10)

    def _on_back_to_menu(self):
        """Switch to the MainMenuFrame"""
        self.switch_frame(MainMenuFrame)
    

if __name__ == "__main__":
    app = MenuApplication()

    logger.info("Starting the application")

    app.mainloop()
