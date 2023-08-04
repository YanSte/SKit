import pandas as pd
import time

from IPython.display import display
from skit.show import show_text


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
                "accuracy":        None,
                "duration":        None,
                "best_model_path": None,
                "board_path":      None
            }

    def reset(self):
        self.output = {}
        for version in versions:
            self.output[version] = {
                "accuracy":        None,
                "duration":        None,
                "best_model_path": None,
                "board_path":      None
            }

    def show_report(self):
        """
        Display a tabular report of the model performance.
        """
        # Convert the output dictionary to a DataFrame
        df = pd.DataFrame.from_dict(self.output, orient='index')

        # Convert the 'accuracy' and 'duration' columns to numeric (float) data types
        df['accuracy'] = df['accuracy'].astype(float)
        df['duration'] = df['duration'].astype(float)

        # Sort the DataFrame by accuracy in descending order to find the best model
        df_sorted = df.sort_values(by='accuracy', ascending=False)

        # Highlight the best accuracy cell
        df_styled = df_sorted.style.apply(lambda x: ['background: yellow' if x.name == df_sorted.index[0] and col == 'accuracy' else '' for col in x], axis=1)

        # Format the accuracy and duration columns to display with two decimal places
        df['accuracy'] = df['accuracy'].map("{:.2f}".format)
        df['duration'] = df['duration'].map("{:.2f}".format)

        # Display the report
        display(df_styled)

    def show_result(self, version):
        """
        Display the result (accuracy and duration) for a specific model version.

        Parameters
        ----------
        version : str
            The model version for which the result will be displayed.
        """
        result = self.output.get(version, None)
        if result is not None:
            accuracy = result.get('accuracy', None)
            duration = result.get('duration', None)

            if accuracy is not None and duration is not None:
                show_text("b", f"Accuracy = {accuracy:.2f} & Duration = {duration:.2f}")
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

    def add_best_model_link(self, version, link):
        """
        Add the link of the best model for the specified model version.

        Parameters
        ----------
        version : str
            The name of the model version for which to add the best model link.
        link : str
            The link or path to the best model.
        """
        self.output[version]['best_model_path'] = link

    def add_board_path(self, version, link):
        """
        Add the link of the tensor board for the specified model version.

        Parameters
        ----------
        version : str
            The name of the model version for which to add the tensor board link.
        link : str
            The link or path to the tensor board.
        """
        self.output[version]['board_path'] = link

    def add_accuracy(self, version, accuracy):
        """
        Add the accuracy score for the specified model version.

        Parameters
        ----------
        version : str
            The name of the model version for which to add the accuracy score.
        accuracy : float
            The accuracy score to be added.
        """
        self.output[version]['accuracy'] = accuracy

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
            accuracy = metrics.get('accuracy', None)
            model_path = metrics.get('best_model_path', None)

            if accuracy is not None and model_path is not None and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = model_path

        return best_model_path
