import logging
import os
from pathlib import Path

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))


def get_logger(logger_name, logging_config={}):
    config_dict = {
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "level": logging.INFO,
        "filename": os.path.join(parent_dir, "logs", "default.log"),
        "filemode": "a",
    }

    for k, v in logging_config.items():
        config_dict[k] = v
        if k == "filename":
            file_parent = Path(v).parent
            if not file_parent.exists:
                file_parent.mkdir(exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(config_dict["level"])

    formatter = logging.Formatter(config_dict["format"])
    # file handler
    file_handler = logging.FileHandler(
        config_dict["filename"], config_dict["filemode"]
    )
    file_handler.setLevel(config_dict["level"])
    file_handler.setFormatter(formatter)
    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    if len(logger.handlers) == 0:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
