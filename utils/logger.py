import os
import logging
from time import strftime, gmtime

LOGGING_DIR = "dataset/.logs"

if not os.path.exists(LOGGING_DIR):
    os.makedirs(LOGGING_DIR, exist_ok=True)

logging.getLogger("urllib3").setLevel(logging.WARNING)


class Logger:
    """
    Creates and initializes a logger using the `logging` Python module
    """

    # TODO: Modify filename to *.log

    def __init__(self, name, filename="log.txt"):
        """
        Initializes the logger
        :param filename:
        """
        self.timestamp = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        self.filename = f"{name}_{self.timestamp}_{filename}"
        self.name = name

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%m-%d-%Y %H:%M",
            handlers=[
                # logging.StreamHandler(),
                logging.FileHandler(os.path.join(LOGGING_DIR, self.filename)),
            ],
        )

        self.logger_object = logging.getLogger(name)

    def get_logger(self):
        """
        :return: the logger object
        """
        return self.logger_object
