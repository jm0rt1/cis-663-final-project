import logging
import logging.handlers
from src.shared.settings import GlobalSettings
from src.face_recognition.face_recognition import run_experiment

import os


def initialize_logging():

    file_handler = logging.handlers.RotatingFileHandler(
        GlobalSettings.GLOBAL_LOGS_DIR/GlobalSettings.LoggingParams.GLOBAL_FILE_NAME,
        backupCount=GlobalSettings.LoggingParams.BACKUP_COUNT)

    logging.getLogger().addHandler(file_handler)
    file_handler.doRollover()
    logging.info("Global Logging Started")


def main():
    """run a console menu that has two options, runs in a while loop so multiple options can be selected"""

    initialize_logging()

    percentage = 20

    directory = "tests/test_files/inputs/tom_cruise"
    count = 0
    # count number of files in directory containing true images
    for file in os.listdir(directory):
        if "true" in file:
            count += 1

    run_experiment(calculate_n_components_to_add_from_lfw(count, 5), directory)
    run_experiment(calculate_n_components_to_add_from_lfw(
        count, 10), directory)
    run_experiment(calculate_n_components_to_add_from_lfw(
        count, 15), directory)
    run_experiment(calculate_n_components_to_add_from_lfw(
        count, 20), directory)
    run_experiment(calculate_n_components_to_add_from_lfw(
        count, 25), directory)
    run_experiment(calculate_n_components_to_add_from_lfw(
        count, 30), directory)
    run_experiment(calculate_n_components_to_add_from_lfw(
        count, 70), directory)


def calculate_n_components_to_add_from_lfw(directory_true_count: int, desired_percentage: int):
    """calculate the number of components to add to the dataset to achieve a desired percentage of true images"""

    # get the number of components to add to the dataset
    n_components_to_add = directory_true_count / \
        (desired_percentage / 100)

    # round the number of components to add to the dataset
    n_components_to_add = round(n_components_to_add)

    return n_components_to_add
