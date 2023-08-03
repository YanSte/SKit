
from google.colab import drive
import subprocess  # Add this line to import the subprocess module
import os


def setup_kaggle_dataset(kaggle_dataset_url, kaggle_config_dir, run_dir):
    try:
        import google.colab
        # Your code that uses google.colab goes here
    except ImportError:
        print("Optional Colab feature requires, please `pip install google-colab")

    try:
        gdrive_path = '/content/gdrive'
        my_drive_path = f'{gdrive_path}/My Drive'

        # Mount Google Drive
        drive.mount(gdrive_path)

        # Set Kaggle config directory
        os.environ['KAGGLE_CONFIG_DIR'] = f"{my_drive_path}/{kaggle_config_dir}"

        # Create the run directory if it doesn't exist
        full_run_dir = f"{my_drive_path}/{run_dir}"
        if not os.path.exists(full_run_dir):
            os.makedirs(full_run_dir)

        # Change to the Kaggle directory
        os.chdir(full_run_dir)

        # Download the dataset using Kaggle CLI
        subprocess.run(['kaggle', 'datasets', 'download', '-d', kaggle_dataset_url])

        # Unzip the downloaded files
        subprocess.run(['unzip', '*.zip'], shell=True)
        os.remove('*.zip')

        print("Dataset downloaded and unzipped successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
