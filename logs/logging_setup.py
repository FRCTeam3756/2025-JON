import os
import logging
from config import LoggingConfig

def setup_logger(file_name):
    log_dir = "logs/files"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, f"{file_name}.log")
    
    if not os.path.exists(log_file_path):
        open(log_file_path, 'a').close()

    logger = logging.getLogger(file_name)
    logger.setLevel(LoggingConfig.LOG_LEVEL)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(LoggingConfig.LOG_LEVEL)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger
