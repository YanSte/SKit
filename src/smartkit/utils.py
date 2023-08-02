from genericpath import isdir
import os, sys
import string
import random
import numpy as np
from pathlib import Path
from glob import glob
import yaml



import datetime
from IPython.display import display,Image,Markdown,HTML


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
