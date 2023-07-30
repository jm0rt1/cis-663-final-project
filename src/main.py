import logging
import logging.handlers
import os

from src.shared.settings import GlobalSettings
from src.face_recognition.face_recognition import run_experiment


def initialize_logging() -> None:
    """
    Initialize logging configurations using settings from the GlobalSettings module.
    """
    file_handler = logging.handlers.RotatingFileHandler(
        GlobalSettings.GLOBAL_LOGS_DIR / GlobalSettings.LoggingParams.GLOBAL_FILE_NAME,
        backupCount=GlobalSettings.LoggingParams.BACKUP_COUNT)

    logging.getLogger().addHandler(file_handler)
    file_handler.doRollover()
    logging.info("Global Logging Started")


def calculate_n_components_to_add_from_lfw(directory_true_count: int, desired_percentage: int) -> int:
    """
    Calculate the number of components to add to the dataset to achieve a desired percentage of true images.

    Args:
        directory_true_count (int): Number of true images in the directory.
        desired_percentage (int): Desired percentage of true images.

    Returns:
        int: Rounded number of components to add.
    """
    n_components_to_add = directory_true_count / (desired_percentage / 100)
    return round(n_components_to_add)


def main() -> None:
    """
    Run a console menu that has multiple options for running face recognition experiments.
    """
    initialize_logging()

    directory = "tests/test_files/inputs/tom_cruise"
    # count number of files in directory containing true images
    count = sum("true" in file for file in os.listdir(directory))

    for percentage in [5, 10, 15, 20, 25, 30, 70]:
        run_experiment(calculate_n_components_to_add_from_lfw(
            count, percentage), directory)
