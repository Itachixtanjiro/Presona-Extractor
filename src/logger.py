import logging
from src.config import LOG_FILE, LOG_LEVEL

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create a file handler
        handler = logging.FileHandler(LOG_FILE)
        
        # Create a formatter and set it for the handler
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(handler)
        
        # Set the logger level
        logger.setLevel(LOG_LEVEL)
        
    return logger

# Example of a base logger for the application
log = get_logger("persona_extractor")