"""
GUI window for training the model and saving it
"""

import pathlib
import tkinter.filedialog

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

import lib.dataset_names as dataset_names
from model.train import train_save_model
from model.classifier import SELECTED_MODEL

import logging

logger = logging.getLogger(__name__)

"""
TrainWindow class for training the model and saving it.
Consists of the following widgets:
- Title label
- Choose training data button
- Train and save model button
"""


class TrainWindow(ttk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, padding=15, *args, **kwargs)
        self.pack(fill=BOTH, expand=YES)

        self.title_label = ttk.Label(
            self, text="Train the model", style="primary.TLabel"
        )
        self.title_label.pack(pady=10)

        self.default_path = pathlib.Path(
            pathlib.Path().absolute(), dataset_names.DATA_DIR
        )
        self.path_var = ttk.StringVar(value=self.default_path)

        # header and labelframe option container
        option_text = "Select training data"
        self.option_lf = ttk.Labelframe(self, text=option_text, padding=15)
        self.option_lf.pack(fill=X, expand=YES, anchor=N)

        self.create_path_row()
        self.create_train_button()

    def create_path_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_lf)
        path_row.pack(fill=X, expand=YES)

        path_lbl = ttk.Label(path_row, text="Path", width=8)
        path_lbl.pack(side=LEFT, padx=(15, 0))

        path_ent = ttk.Entry(path_row, textvariable=self.path_var, width=50)
        path_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)

        browse_btn = ttk.Button(
            master=path_row, text="Browse", command=self.on_browse, width=8
        )
        browse_btn.pack(side=LEFT, padx=5)

    def create_train_button(self):
        """Add train button to labelframe"""
        train_btn = ttk.Button(
            self.option_lf, text="Train and save model", command=self.on_train
        )
        train_btn.pack(pady=10)

    def on_browse(self):
        """Open file dialog to select training data"""
        path = tkinter.filedialog.askopenfilename(
            title="Select training data", filetypes=[("CSV files", "*.csv")]
        )
        if path:
            self.path_var.set(path)

    def on_train(self):
        """Train the model and save it"""
        save_path = self.path_var.get() + "/model.json"
        logger.info(f"Training model with data at {save_path}")
        logger.info(f"Saving model to {save_path}")
        train_save_model(SELECTED_MODEL, save_path)


if __name__ == "__main__":
    root = ttk.Window()
    train_window = TrainWindow(root)
    train_window.pack()

    logging.basicConfig(level=logging.INFO)

    root.mainloop()
