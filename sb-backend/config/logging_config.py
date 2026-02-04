import logging.config
import os
import yaml
from pathlib import Path
from datetime import datetime


def setup_logging(config_path: str | Path | None = None) -> None:
    # Use environment variable if provided, otherwise default to "logging.yaml"
    if config_path is None:
        config_path = os.getenv('LOGGING_CONFIG_PATH', 'logging.yaml')
    
    path = Path(config_path)

    with path.open("r") as f:
        config = yaml.safe_load(f)
    
    # Set log directory from environment variable
    log_dir = os.getenv('LOG_DIR', 'logs')
    
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Get current date and time in the format used in the config: YYYY-MM-DD HH:MM:SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Update file handlers with the log directory and timestamp
    for handler_name, handler_config in config.get('handlers', {}).items():
        if handler_config.get('class') in [
            'logging.handlers.TimedRotatingFileHandler',
            'logging.FileHandler'
        ]:
            filename = handler_config.get('filename', '')
            if filename:
                # Extract base filename and extension
                base_filename = os.path.basename(filename)
                name_parts = os.path.splitext(base_filename)
                
                # Append timestamp to filename: filename_YYYY-MM-DD_HH-MM-SS.log
                timestamped_filename = f"{name_parts[0]}_{timestamp}{name_parts[1]}"
                handler_config['filename'] = os.path.join(log_dir, timestamped_filename)

    logging.config.dictConfig(config)


def get_logger(name: str):
    return logging.getLogger(name)


def get_audit_logger():
    return logging.getLogger("audit")
