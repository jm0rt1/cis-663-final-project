import logging
import logging.handlers
from src.shared.settings import GlobalSettings
from src.face_recognition.face_recognition import run_experiment


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
    run_experiment(100, "tests/test_files/inputs/tom_cruise")
    run_experiment(5, "tests/test_files/inputs/tom_cruise")
    run_experiment(759, "tests/test_files/inputs/tom_cruise")
