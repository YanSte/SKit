# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 YanSte

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import pandas as pd
import numpy as np
import time

from enum import Enum
from IPython.display import display
from skit.show import show_text, show_history

class Metric(Enum):
    ACCURACY = "accuracy"
    AUC = "auc"
    VAL_AUC = "val_auc"
    VAL_ACCURACY = "val_accuracy"

    @property
    def train_metric_key(self):
        return self.value

    @property
    def val_metric_key(self):
        if self == Metric.ACCURACY:
            return "val_accuracy"
        elif self == Metric.AUC:
            return "val_auc"
        elif self == Metric.VAL_AUC:
            return "val_auc"
        elif self == Metric.VAL_ACCURACY:
            return "val_accuracy"

    @property
    def plot_labels(self):
        """
        Get the curve labels corresponding to the given Metric enum.

        Parameters:
        metric_enum (Metric): The Metric enum value.

        Returns:
        dict: A dictionary mapping curve labels to metric names.
        """
        if self == Metric.ACCURACY or self == Metric.VAL_ACCURACY:
            return {
                'Accuracy': {
                    'Training Accuracy': 'accuracy',
                    'Validation Accuracy': 'val_accuracy'
                },
                'Loss': {
                    'Training Loss': 'loss',
                    'Validation Loss': 'val_loss'
                }
            }
        elif self == Metric.AUC or self == Metric.VAL_AUC:
            return {
                'Accuracy': {
                    'Training AUC': 'auc',
                    'Validation AUC': 'val_auc'
                },
                'Loss': {
                    'Training Loss': 'loss',
                    'Validation Loss': 'val_loss'
                }
            }

class ModelMetrics:
    def __init__(self, versions, metric_to_monitor=Metric.ACCURACY):
        self.output = {}
        self.metric_to_monitor = metric_to_monitor
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

    def get_best_metric(self, version):
        history = self.output[version]['history'].history

        train_metric_key = self.metric_to_monitor.train_metric_key
        val_metric_key = self.metric_to_monitor.val_metric_key

        best_val_index = np.argmax(history[train_metric_key])
        best_train_metric = history[train_metric_key][best_val_index]
        best_val_metric = history[val_metric_key][best_val_index]

        return {
            f'best_train_{self.metric_to_monitor.name.lower()}': best_train_metric,
            f'best_val_{self.metric_to_monitor.name.lower()}': best_val_metric,
        }

    def get_best_report(self, version):
        """
        Get the best model report for a specific model version.

        Parameters
        ----------
        version : str
            The model version for which to get the best model report.

        Returns
        -------
        dict or None
            The best model report containing training and validation metrics, duration, and paths.
            Returns None if the specified version is not found in the output.
        """
        if version not in self.output:
            return None

        metrics = self.get_best_metric(version)

        return {
            'version': version,
            f'best_train_{self.metric_to_monitor.name.lower()}': metrics[f'best_train_{self.metric_to_monitor.name.lower()}'],
            f'best_val_{self.metric_to_monitor.name.lower()}': metrics[f'best_val_{self.metric_to_monitor.name.lower()}'],
            'duration': self.output[version]['duration'],
            'best_model_path': self.output[version]['best_model_path'],
            'board_path': self.output[version]['board_path'],
        }

    def show_report(self):
        """
        Display a tabular report of the best model performance.
        """
        # Initialize the report DataFrame
        columns = ['version', f'best_train_{self.metric_to_monitor.name.lower()}', f'best_val_{self.metric_to_monitor.name.lower()}', 'duration', 'best_model_path', 'board_path']

        df = pd.DataFrame(columns=columns)

        for version in self.output.keys():
            # Get the best training and validation metric for this version
            report = self.get_best_report(version)

            # Add the data to the DataFrame
            df = pd.concat([df, pd.DataFrame([report])], ignore_index=True)

        # Set 'version' as the index of the DataFrame
        df.set_index('version', inplace=True)

        # Apply formatting to the duration and metric columns
        df['duration'] = df['duration'].apply(lambda x: "{:.2f}".format(x))

        metric_columns = [f'best_train_{self.metric_to_monitor.name.lower()}', f'best_val_{self.metric_to_monitor.name.lower()}']
        df[metric_columns] = df[metric_columns].applymap(lambda x: "{:.2f}".format(x*100) if self.metric_to_monitor != Metric.VAL_ACCURACY else "{:.2f}%".format(x))

        # Highlight the maximum in the metric column
        styled_df = df.style.highlight_max(subset=[f'best_val_{self.metric_to_monitor.name.lower()}'], color='lightgreen')

        # Display the report
        display(styled_df)


    def show_best_result(self, version):
        """
        Display the result (best train metric, best validation metric, and duration) for a specific model version.

        Parameters
        ----------
        version : str
            The model version for which the result will be displayed.
        """
        if version not in self.output:
            show_text("b", f"No result available for {version}")

        result = self.get_best_report(version)

        if result is not None:
            best_train_metric = result.get(f'best_train_{self.metric_to_monitor.name.lower()}', None)
            best_val_metric = result.get(f'best_val_{self.metric_to_monitor.name.lower()}', None)
            duration = result.get('duration', None)

            metric_name = self.metric_to_monitor.name.lower()
            metric_suffix = '%' if self.metric_to_monitor != Metric.VAL_ACCURACY else ''

            if best_train_metric is not None and best_val_metric is not None and duration is not None:
                show_text("b", f"Train {metric_name.capitalize()} = {best_train_metric * 100:.2f}{metric_suffix} - Validation {metric_name.capitalize()} = {best_val_metric * 100:.2f}{metric_suffix} - Duration = {duration:.2f}")
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
        self.output[version]['history'] = history

    def show_history(
        self,
        version,
        figsize=(8,6)
    ):
        history = self.output[version]['history']
        plot = self.metric_to_monitor.plot_labels
        display(show_history(history, figsize=figsize, plot=plot))

    def get_best_model_path(self, version):
        """
        Get the path of the best model based on accuracy.

        Parameters
        ----------
        version : str
            The name of the model version for which to get the best model path.

        Returns
        -------
        str or None
            The path of the best model based on the highest accuracy score.
            Returns None if no model has been added or no best model path is available.
        """
        report = self.get_best_report(version)
        best_model_path = report.get('best_model_path')

        if best_model_path is not None:
            return best_model_path
        else:
            return None
