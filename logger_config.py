import logging
import os
from datetime import datetime

def setup_logging():
    """
    Configures logging to save each run to a timestamped file in the 'logs' directory.
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{timestamp}.log")

    # Configure logging with UTF-8 encoding for both file and console
    # We use basicConfig to set the root logger handlers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Force UTF-8 encoding for console handler (StreamHandler)
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            # Set encoding for console output
            if hasattr(handler.stream, 'reconfigure'):
                # Python 3.7+ - reconfigure the stream
                handler.stream.reconfigure(encoding='utf-8')
            else:
                # Fallback for older Python or streams without reconfigure
                import sys
                if handler.stream in (sys.stdout, sys.stderr):
                    # Wrap the stream with UTF-8 encoding
                    import codecs
                    handler.stream = codecs.getwriter('utf-8')(handler.stream.buffer)
    
    # Create a logger for this module to confirm setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Saving logs to {log_filename}")
