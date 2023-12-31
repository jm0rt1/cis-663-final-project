import logging
import logging.handlers
import os

from src.shared.settings import GlobalSettings
from src.experiment.experiment import run_experiment


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


def main() -> None:
    """
    Run a console menu that has multiple options for running face recognition experiments.
    """
    initialize_logging()

    directory = "tests/test_files/inputs/me"
    # count number of files in directory containing true images

    for percentage in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 95]:
        # for percentage in [50]:

        print(f"Running experiment with {percentage}% true images...")
        if percentage == 5:
            run_experiment(percentage, directory, True)
        else:
            run_experiment(percentage, directory, False)
