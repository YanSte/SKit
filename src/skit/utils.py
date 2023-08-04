import os
import shutil
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------------
# Folder
# -------------------------------------------------------------

def mkdir(path):
    '''
    Create a subdirectory
    Mode is 0750, do nothing if exist
    args:
        path : directory to create
    return:
        none
    '''
    os.makedirs(path, mode=0o750, exist_ok=True)

def rmdir(path):
    """
    Deletes a folder at the specified path.

    :param path: The path of the folder to delete.
    :type path: str
    """

    # Vérifie si le dossier existe
    if os.path.exists(path):
        # Utilise shutil.rmtree pour supprimer le dossier
        shutil.rmtree(path)
    else:
        print(f"No folder found at {path}.")


def ls(directory_path, filetype='all'):
    if filetype == 'dir':
        return [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    elif filetype == 'file':
        return [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))]
    else:
        return ls(directory_path, 'dir') + ls(directory_path, 'file')

def get_directory_size(path):
    """
    Return the directory size, but only 1 level
    args:
        path : directory path
    return:
        size in Mo
    """
    size=0
    for f in os.listdir(path):
        if os.path.isfile(path+'/'+f):
            size+=os.path.getsize(path+'/'+f)
    return size/(1024*1024)


def count_files(data_dir, file_type):
    """
    Count the number of specific type files in a given directory and its subdirectories.

    Args:
        data_dir (str): The directory to start the search from.
        file_type (str): The type of file to count. Should include the dot, like '.txt' or '.jpg'.

    Returns:
        dict: A dictionary where the keys are the paths of the directories and
              the values are the counts of the specific type files in those directories.
    """
    # Function to count files in parallel
    def count_files_in_directory(path, file_type):
        file_count = 0
        for file_path in path.glob(f'*{file_type}'):
            if file_path.is_file():
                file_count += 1
        return file_count

    data_dir = Path(data_dir)
    file_counts = {}
    total_dirs = sum(1 for _ in data_dir.rglob('*'))

    with tqdm(total=total_dirs, desc="Counting files", unit="dir") as pbar:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(count_files_in_directory, path, file_type): path for path in data_dir.rglob('*')}

            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    count = future.result()
                    file_counts[str(path.parent)] = count
                except Exception as exc:
                    print(f"Error counting files in {path}: {exc}")
                pbar.update(1)

    return file_counts
