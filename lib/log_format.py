# Format: LEVEL:__name__:TIME:MESSAGE
LOG_FORMAT = "%(levelname)s:%(name)s:%(asctime)s:%(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_CONFIG_KWARGS = {
    "format": LOG_FORMAT,
    "datefmt": LOG_DATE_FORMAT,
    "filename": "logs/log.log",
    "filemode": "a",
}
