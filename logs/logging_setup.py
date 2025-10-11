import os
import io
import sys
import logging
from logging import Logger
from datetime import datetime
from config import LoggingConfig

def setup_logger(file_name: str) -> Logger:
    log_dir = f"logs/files/{datetime.now():%Y-%m-%d_%H-%M-%S}"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{file_name}.log")

    logger = logging.getLogger(file_name)
    logger.setLevel(LoggingConfig.LOG_LEVEL)

    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(LoggingConfig.LOG_LEVEL)
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger
