import logging
import sys

def setup_logger(name: str = "feedback_classifier") -> logging.Logger:
    """Sets up a standardized logger with console output.
    
    Args:
        name: Name of the logger.
        
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
