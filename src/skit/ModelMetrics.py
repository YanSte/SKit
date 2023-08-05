import pandas as pd
import numpy as np
import time

from IPython.display import display
from skit.show import show_text, show_history

class ModelMetrics:
    def __init__(self, versions):
        """
        Initialize the ModelMetrics.

        Parameters
        ----------
        versions : list
            A list of model version names to track performance for.
        """
        self.output = {}
        for version in versions:
            self.output[version] = {
                "history":         None,
                "duration":        None,
                "best_model_path": None,
                "board_path":      None
            }

    def reset(self, version=None):
        default_dict = {
            "history":         None,
            "duration":        None,
            "best_model_path": None,
            "board_path":      None
            }

        if version is not None:
            self.output[version] = default_dict
        else:
            # Reset all versions
            for version in self.output.keys():
                self.output[version] = default_dict.copy()

    def get_best_accuracy(self, version):
        """
        Get the best training and validation accuracy for the specified model version.

        Parameters
        ----------
        version : str
            The name of the model version for which to get the accuracy score.

        Returns
        -------
        dict
            A dictionary containing the best training and validation accuracy.
        """
        history = self.output[version]['history']

        # Find the index of the best validation accuracy
        best_val_index = np.argmax(history.history['val_accuracy'])

        # Get the training accuracy at the epoch of the best validation accuracy
        best_train_accuracy = history.history['accuracy'][best_val_index]

        # Get the best validation accuracy
        best_val_accuracy = history.history['val_accuracy'][best_val_index]

        return {
            'best_train_accuracy': best_train_accuracy,
            'best_val_accuracy': best_val_accuracy,
        }

    def get_best_report(self, version):
        """
        Get the best report for the specified model version.

        Parameters
        ----------
        version : str
            The name of the model version for which to get the accuracy score.

        Returns
        -------
        pandas.DataFrame
            A dictionary containing the best report.
        """
        metrics = self.get_best_accuracy(version)

        return {
            'version': version,
            'best_train_accuracy': metrics['best_train_accuracy'],
            'best_val_accuracy': metrics['best_val_accuracy'],
            'duration': self.output[version]['duration'],
            'best_model_path': self.output[version]['best_model_path'],
            'board_path': self.output[version]['board_path'],
        }

    def show_report(self):
        """
        Display a tabular report of the best model performance.
        """
        # Initialize the report DataFrame
        df = pd.DataFrame(columns=['version', 'best_train_accuracy', 'best_val_accuracy', 'duration', 'best_model_path', 'board_path'])

        for version in self.output.keys():
            # Get the best training and validation accuracy for this version
            report = self.get_best_report(version)

            # Add the data to the DataFrame
            df = df.append(report, ignore_index=True)

        # Set 'version' as the index of the DataFrame
        df.set_index('version', inplace=True)

        # Apply formatting to the duration and accuracy columns
        df['duration'] = df['duration'].apply(lambda x: "{:.2f}".format(x))
        df[['best_train_accuracy', 'best_val_accuracy']] = df[['best_train_accuracy', 'best_val_accuracy']].applymap(lambda x: "{:.2f}%".format(x*100))

        # Highlight the maximum in the 'best_val_accuracy' column
        styled_df = df.style.highlight_max(subset=['best_val_accuracy'], color='lightgreen')

        # Display the report
        display(styled_df)

    def show_best_result(self, version):
        """
        Display the result (best train accuracy, best validation accuracy and duration) for a specific model version.

        Parameters
        ----------
        version : str
            The model version for which the result will be displayed.
        """
        result = self.get_best_report(version)

        if result is not None:
            best_train_accuracy = result.get('best_train_accuracy', None)
            best_val_accuracy = result.get('best_val_accuracy', None)
            duration = result.get('duration', None)

            if best_train_accuracy is not None and best_val_accuracy is not None and duration is not None:
                show_text("b", f"Train Accuracy = {best_train_accuracy * 100:.2f}% - Validation Accuracy = {best_val_accuracy * 100:.2f}% - Duration = {duration:.2f}")
            else:
                show_text("b", f"Result not available for version {version}")
        else:
            show_text("b", f"Version {version} not found in the output")


    def start_timer(self, version):
        """
        Start the timer for tracking model training or evaluation duration.

        Parameters
        ----------
        version : str
            The name of the model version for which to start the timer.
        """
        self.output[version]['duration'] = time.time()

    def stop_timer(self, version):
        """
        Stop the timer for tracking model training or evaluation duration.

        Parameters
        ----------
        version : str
            The name of the model version for which to stop the timer.
        """
        if self.output[version]['duration'] is not None:
            duration = time.time() - self.output[version]['duration']
            self.output[version]['duration'] = duration

    def add_best_model_path(self, version, path):
        """
        Add the link of the best model for the specified model version.

        Parameters
        ----------
        version : str
            The name of the model version for which to add the best model link.
        link : str
            The link or path to the best model.
        """
        self.output[version]['best_model_path'] = path

    def add_board_path(self, version, path):
        """
        Add the link of the tensor board for the specified model version.

        Parameters
        ----------
        version : str
            The name of the model version for which to add the tensor board link.
        link : str
            The link or path to the tensor board.
        """
        self.output[version]['board_path'] = path

    def add_history(self, version, history):
        """
        Add the history of the specified model version.

        Parameters
        ----------
        version : str
            The name of the model version for which to add the accuracy score.
        history : dict
            The accuracy score to be added.
        """
        self.output[version]['history'] = history.history

    def show_history(
        self,
        version,
        figsize=(8,6),
        plot = {"Accuracy":['accuracy','val_accuracy'], 'Loss':['loss', 'val_loss']}
    ):
        history = self.output[version]['history']
        display(show_history(history, figsize = figsize, plot = plot))

    def get_best_model_path(self):
        """
        Get the path of the best model based on accuracy.

        Returns
        -------
        str or None
            The path of the best model based on the highest accuracy score.
            Returns None if no model has been added or no best model path is available.
        """
        best_accuracy = -1
        best_model_path = None

        for version, metrics in self.output.items():
            accuracy = self.get_best_accuracy(version).get('best_val_accuracy', None)
            model_path = metrics.get('best_model_path', None)

            if accuracy is not None and model_path is not None and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = model_path

        return best_model_path
