import logging

# Format: LEVEL:__name__:TIME:MESSAGE
LOG_FORMAT = "%(levelname)s:%(name)s:%(asctime)s:%(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_CONFIG_KWARGS = {
    "format": LOG_FORMAT,
    "datefmt": LOG_DATE_FORMAT,
    # Log both to a file and to the console
    "handlers": [
        logging.FileHandler("logs/log.log", mode="a"),
        logging.StreamHandler(),
    ],
}
