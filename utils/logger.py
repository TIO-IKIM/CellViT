# -*- coding: utf-8 -*-
# Logging Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import datetime
from typing import Literal, Union
from pathlib import Path
import logging
import logging.handlers
import os
import sys


class Logger:
    """Initialize a Logger for sys-logging and RotatingFileHandler-logging by using python logging module.
    The logger can be used out of the box without any changes, but is also adaptable for specific use cases.
    In basic configuration, just the log level must be provided. If log_dir is provided, another handler object is created
    logging into a file into the log_dir directory. The filename can be changes by using comment, which basically is the filename.
    To create different log files with specific timestamp set 'use_timestamp' = True. This adds an additional timestamp to the filename.

    Args:
        level (Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]): Logger.level
        log_dir (Union[Path, str], optional): Path to save logfile in. Defaults to None.
        comment (str, optional): additional comment for save file. Defaults to 'logs'.
        formatter (str, optional): Custom formatter. Defaults to None.
        use_timestamp (bool, optional): Using timestamp for time-logging. Defaults to False.
        file_level (Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], optional): Set Logger.level. for output file.
            Can be useful if a different logging level should be used for terminal output and logging file.
            If no level is selected, file level logging is the same as for console. Defaults to None.
    """

    def __init__(
        self,
        level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        log_dir: Union[Path, str] = None,
        comment: str = "logs",
        formatter: str = None,
        use_timestamp: bool = False,
        file_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = None,
    ) -> None:
        self.level = level
        self.comment = comment
        self.log_parent_dir = log_dir
        self.use_timestamp = use_timestamp
        if formatter is None:
            self.formatter = "%(asctime)s [%(levelname)s] - %(message)s"
        else:
            self.formatter = formatter
        if file_level is None:
            self.file_level = level
        else:
            self.file_level = file_level

    def create_handler(self, logger: logging.Logger) -> None:
        """Create logging handler for sys output and rotating files in parent_dir.

        Args:
            logger (logging.Logger): The Logger
        """
        log_handlers = {"StreamHandler": logging.StreamHandler(stream=sys.stdout)}
        fh_formatter = logging.Formatter(f"{self.formatter}")
        log_handlers["StreamHandler"].setLevel(self.level)

        if self.log_parent_dir is not None:
            log_parent_dir = Path(self.log_parent_dir)
            if self.use_timestamp:
                log_name = f'{datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")}_{self.comment}.log'
            else:
                log_name = f"{self.comment}.log"
            log_parent_dir.mkdir(parents=True, exist_ok=True)

            should_roll_over = os.path.isfile(log_parent_dir / log_name)

            log_handlers["FileHandler"] = logging.handlers.RotatingFileHandler(
                log_parent_dir / log_name, backupCount=5
            )

            if should_roll_over:  # log already exists, roll over!
                log_handlers["FileHandler"].doRollover()
            log_handlers["FileHandler"].setLevel(self.file_level)

        for handler in log_handlers.values():
            handler.setFormatter(fh_formatter)
            logger.addHandler(handler)

    def create_logger(self) -> logging.Logger:
        """Create the logger

        Returns:
            Logger: The logger to be used.
        """
        logger = logging.getLogger("__main__")
        logger.addHandler(logging.NullHandler())

        logger.setLevel(
            "DEBUG"
        )  # set to debug because each handler level must be equal or lower
        self.create_handler(logger)

        return logger
