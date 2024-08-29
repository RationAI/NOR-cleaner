"""
GUI window for training the model and saving it
"""

import logging
import pathlib
import tkinter.filedialog

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

import lib.dataset_names as dataset_names
from gui.error_box import ErrorBox
from model.classifier import SELECTED_MODEL
from model.train import train, ModelType

logger = logging.getLogger(__name__)


class TrainWindow(ttk.Frame):
    """
    TrainWindow class for training the model and saving it.
    Consists of the following widgets:
    - Title label
    - Choose training data button
    - Train and save model button
    """

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
        self.data_path_var = ttk.StringVar(value=self.default_path)
        self.save_model_path_var = ttk.StringVar(
            value=pathlib.Path(self.default_path, "models", "model.json")
        )

        # header and labelframe option container
        option_text = "Select training data"
        self.option_lf = ttk.Labelframe(self, text=option_text, padding=15)
        self.option_lf.pack(fill=X, expand=YES, anchor=N)

        self.create_path_row()
        self.create_save_model_row()
        self.create_train_button()

    def create_path_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_lf)
        path_row.pack(fill=X, expand=YES)

        path_lbl = ttk.Label(path_row, text="Take data from", width=15)
        path_lbl.pack(side=LEFT, padx=(15, 0))

        path_ent = ttk.Entry(
            path_row, textvariable=self.data_path_var, width=50
        )
        path_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)

        browse_btn = ttk.Button(
            master=path_row,
            text="Browse",
            command=self.on_browse_data,
            width=8,
        )
        browse_btn.pack(side=LEFT, padx=5)

    def create_save_model_row(self):
        """Add save model path row to labelframe"""
        save_model_row = ttk.Frame(self.option_lf)
        save_model_row.pack(fill=X, expand=YES, pady=10)

        save_model_lbl = ttk.Label(
            save_model_row, text="Save model to", width=15
        )
        save_model_lbl.pack(side=LEFT, padx=(15, 0))

        save_model_ent = ttk.Entry(
            save_model_row, textvariable=self.save_model_path_var, width=50
        )
        save_model_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)

        browse_btn = ttk.Button(
            master=save_model_row,
            text="Browse",
            command=self.on_browse_save_model,
            width=8,
        )
        browse_btn.pack(side=LEFT, padx=5)

    def create_train_button(self):
        """Add train button to labelframe"""
        train_btn = ttk.Button(
            self.option_lf, text="Train and save model", command=self.on_train
        )
        train_btn.pack(pady=10)

    def on_browse_data(self):
        """Open file dialog to select training data"""
        path = tkinter.filedialog.askopenfilename(
            title="Select training data", filetypes=[("CSV files", "*.csv")]
        )
        if path:
            self.data_path_var.set(path)

    def on_browse_save_model(self):
        """Open file dialog to select save model path"""
        path = tkinter.filedialog.asksaveasfilename(
            title="Save model to", filetypes=[("JSON files", "*.json")]
        )
        if path:
            self.save_model_path_var.set(path)
        
    def _valid_path_parent(self, path: str) -> None:
        """
        Check if every parent directory in the path exists
        """
        for directory in pathlib.Path(path).parents:
            if not directory.exists():
                raise FileNotFoundError(f"Directory does not exist: {directory}")
        
    def _check_save_path(self, save_path: str) -> None:
        self._valid_path_parent(save_path)
        
        if not save_path.endswith(".json"):
            raise ValueError("Model must be saved as a JSON file")

    def on_train(self):
        """Train the model and save it"""
        try:
            self._train()
        except Exception as e:
            logger.error(e)
            ErrorBox.from_exception(e)

    def _train(self):
        save_path = self.save_model_path_var.get()
        self._check_save_path(save_path)

        data_path = self.data_path_var.get()
        logger.info(f"Training model with data at {data_path}")
        model = train(SELECTED_MODEL, data_path)
        logger.info("Model trained successfully")
        self.save_model(model, save_path)

    def save_model(self, model: ModelType, save_path: str) -> None:
        """Save the model to the given path"""
        logger.info(f"Saving model to {save_path}")
        model.save_model(save_path)
        logger.info("Model saved successfully")


if __name__ == "__main__":
    root = ttk.Window()
    train_window = TrainWindow(root)
    train_window.pack()

    logging.basicConfig(level=logging.INFO)

    root.mainloop()
