import logging
import os 
from datetime import datetime 

# create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO
)

# Function to get logger
def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance with the specified name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Add a file handler to the logger if it doesn't have one
    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger