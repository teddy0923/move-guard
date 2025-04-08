# core/logging_setup.py
import logging
import os
from datetime import datetime


def setup_logging(config):
    """
    Configure logging based on the provided configuration

    Args:
        config: Dictionary with logging configuration parameters
    """
    log_level = getattr(logging, config.get('level', 'INFO'))
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # File handler if needed
    if config.get('save_to_file', False):
        log_file = config.get('log_file', 'logs/movement_analysis.log')

        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Add timestamp to log filename if it doesn't already have an extension
        if '.' not in os.path.basename(log_file):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{log_file}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    logging.info("Logging system initialized")