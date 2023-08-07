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
    """
    Create a directory at the specified path if it doesn't exist.

    Parameters:
    -----------
    path : str
        Path of the directory to create.

    Notes:
    ------
    The directory mode will be set to 0750.
    If the directory already exists, the function will do nothing.
    """
    os.makedirs(path, mode=0o750, exist_ok=True)

def rmdir(path):
    """
    Deletes a directory at the specified path.

    Parameters:
    -----------
    path : str
        Path of the directory to delete.

    Notes:
    ------
    If the directory doesn't exist, a message will be printed.
    """
    # Vérifie si le dossier existe
    if os.path.exists(path):
        # Utilise shutil.rmtree pour supprimer le dossier
        shutil.rmtree(path)
    else:
        print(f"No folder found at {path}.")


def ls(directory_path, filetype='all'):
    """
    List files or directories inside the given directory based on the specified file type.

    Parameters:
    -----------
    directory_path : str
        Path of the directory to list its contents.

    filetype : {'all', 'dir', 'file'}, optional
        Type of file or directory to list.
        - 'all' : Lists both files and directories.
        - 'dir' : Lists only directories.
        - 'file': Lists only files.
        Default is 'all'.

    Returns:
    --------
    list
        List containing the names of files or directories.
    """
    if filetype == 'dir':
        return [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    elif filetype == 'file':
        return [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))]
    else:
        return ls(directory_path, 'dir') + ls(directory_path, 'file')

def get_directory_size(path):
    """
    Calculates the total size of files in a directory, limited to only 1 level deep.

    Parameters:
    -----------
    path : str
        Path of the directory whose size is to be determined.

    Returns:
    --------
    float
        Total size of the directory in megabytes (MB).
    """
    size=0
    for f in os.listdir(path):
        if os.path.isfile(path+'/'+f):
            size+=os.path.getsize(path+'/'+f)
    return size/(1024*1024)


def count_files(data_dir, file_type):
    """
    Count the number of specific file types in a directory and its subdirectories.

    Parameters:
    -----------
    data_dir : str
        Starting directory to begin the search.

    file_type : str
        Type of file to count. Should include the dot, e.g., '.txt' or '.jpg'.

    Returns:
    --------
    dict
        A dictionary with keys as paths of directories and values as the counts of the specific file types in those directories.

    Notes:
    ------
    This function uses parallel processing to count files for faster results.
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
