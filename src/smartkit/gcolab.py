from config import IN_COLAB

if IN_COLAB:
    try:
        import subprocess
        import shutil
        import os
        import google.colab
        from google.colab import drive

    except ImportError:
        print(f"Missing some imports: {ImportError}")

    def install_kaggle():
        subprocess.run(['pip', 'install', 'kaggle'])

    def set_environ_kaggle_config(kaggle_config_dir):
        gdrive_path = '/content/gdrive'
        my_drive_path = f'{gdrive_path}/My Drive'
        drive.mount(gdrive_path)
        os.environ['KAGGLE_CONFIG_DIR'] = f"{my_drive_path}/{kaggle_config_dir}"

    def setup_kaggle_dataset(kaggle_dataset_url, kaggle_config_dir, run_dir):
        try:
            install_kaggle()
            set_environ_kaggle_config(kaggle_config_dir)

            # Créer le répertoire Kaggle s'il n'existe pas
            os.makedirs(dataset_destination_path, exist_ok=True)

            # Télécharger le jeu de données en utilisant le CLI Kaggle
            !kaggle datasets download -d {kaggle_dataset_url} -p {dataset_destination_path}

            # Dézipper les fichiers téléchargés
            !unzip \*.zip && rm *.zip
            print("Dataset downloaded and unzipped successfully!")

        except Exception as e:
            print(f"An error occurred: {e}")
