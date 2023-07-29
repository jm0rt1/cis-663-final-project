#!/usr/bin/env python3
# script that renames entire folder of true images to the naming convention true1.jpg, true2.jpg, etc.

import os
from pathlib import Path


def rename_files_in_directory(directory):
    i = 1
    for file in os.listdir(directory):
        if file.endswith('.jpg'):  # change this if your images are in a different format
            os.rename(os.path.join(directory, file),
                      os.path.join(directory, f"true{i}.jpg"))
            i += 1


def confirm(directory):
    """ask user to confirm if they want to rename files in the found directory"""
    response = input(
        f"Are you sure you want to rename files in {directory}? (y/n) ")
    if response.lower() == 'y':
        return True
    elif response.lower() == 'n':
        return False
    else:
        print("Invalid response. Please enter 'y' or 'n'.")
        return confirm(directory)


def cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", help="directory where your true images are")
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    if confirm(directory):
        rename_files_in_directory(args.directory)


if __name__ == "__main__":
    # change this to the directory where your true images are
    cli()
