import logging
import os.path
import time
from typing import Optional

from .misc import singleton, master_only
from termcolor import colored
import sys


class Formatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    time_str = "%(asctime)s"
    level_str = "[%(levelname)7s]"
    msg_str = "%(message)s"
    file_str = "(%(filename)s:%(lineno)d)"

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class SteamFormatter(Formatter):

    FORMATS = {
        logging.DEBUG: colored(Formatter.msg_str, "cyan"),
        logging.INFO: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "white",
            attrs=["dark"],
        )
        + colored(Formatter.msg_str, "white"),
        logging.WARNING: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "yellow",
            attrs=["dark"],
        )
        + colored(Formatter.msg_str, "yellow"),
        logging.ERROR: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "red",
            attrs=["dark"],
        )
        + colored(Formatter.msg_str, "red")
        + colored(" " + Formatter.file_str, "red", attrs=["dark"]),
        logging.CRITICAL: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "red",
            attrs=["dark", "bold"],
        )
        + colored(
            Formatter.msg_str,
            "red",
            attrs=["bold"],
        )
        + colored(" " + Formatter.file_str, "red", attrs=["dark", "bold"]),
    }


class FileFormatter(Formatter):

    FORMATS = {
        logging.INFO: " ".join(
            [Formatter.time_str, Formatter.level_str, Formatter.msg_str]
        ),
        logging.WARNING: " ".join(
            [Formatter.time_str, Formatter.level_str, Formatter.msg_str]
        ),
        logging.ERROR: " ".join(
            [
                Formatter.time_str,
                Formatter.level_str,
                Formatter.msg_str,
                Formatter.file_str,
            ]
        ),
        logging.CRITICAL: " ".join(
            [
                Formatter.time_str,
                Formatter.level_str,
                Formatter.msg_str,
                Formatter.file_str,
            ]
        ),
    }


@singleton
class ExpLogger(logging.Logger):

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(time.time()))
        super().__init__(name)
        self.setLevel(logging.DEBUG)
        self.stream_level = logging.INFO

        self.set_log_stream()
        self.filehandler = None

    @master_only
    def set_log_stream(self):
        self.stearmhandler = logging.StreamHandler()
        self.stearmhandler.setFormatter(SteamFormatter())
        self.stearmhandler.setLevel(self.stream_level)

        self.addHandler(self.stearmhandler)

    def remove_log_stream(self):
        self.removeHandler(self.stearmhandler)

    @master_only
    def set_log_file(self, path: str, name: Optional[str] = None):
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(
            path, f"{self.name}.log" if name is None else f"{name}.log"
        )
        self.filehandler = logging.FileHandler(file_path)
        self.filehandler.setFormatter(FileFormatter())
        # File keeps DEBUG so per-iteration diagnostics are still recoverable
        # even though the console stays at INFO.
        self.filehandler.setLevel(logging.DEBUG)
        self.addHandler(self.filehandler)

    @master_only
    def info(self, msg, **kwargs) -> None:
        return super().info(msg, **kwargs)

    @master_only
    def warning(self, msg, **kwargs) -> None:
        return super().warning(msg, **kwargs)

    @master_only
    def error(self, msg, **kwargs) -> None:
        return super().error(msg, **kwargs)

    @master_only
    def debug(self, msg, **kwargs) -> None:
        # Stream handler defaults to DEBUG; per-iteration diagnostics are noisy
        # (large float reprs) and only useful when actively debugging, so keep
        # them out of the console unless --log-level debug is passed.
        if self.stream_level > logging.DEBUG:
            return None
        return super().debug(msg, **kwargs)

    def set_stream_level(self, level: int) -> None:
        """Set the console verbosity, independent of the file handler."""
        self.stream_level = level
        self.stearmhandler.setLevel(level)

    @master_only
    def critical(self, msg, **kwargs) -> None:
        return super().critical(msg, **kwargs)


logger = ExpLogger()

class StreamToLogger():
    def __init__(self, logger, log_level):
        super().__init__()
        self.logger = logger
        self.log_level = log_level

    def write(self, message):
        if message.strip():
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        pass