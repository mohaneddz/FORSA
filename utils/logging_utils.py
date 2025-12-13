import time
import logging
import functools
import inspect

logger = logging.getLogger("performance")

def log_execution_time(func):
    """
    Decorator to log the execution time of a function (sync or async).
    """
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__name__
            logger.info(f"Starting execution of: {func_name}")
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"Finished execution of: {func_name} - Duration: {duration:.4f} seconds")
        return wrapper
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__name__
            logger.info(f"Starting execution of: {func_name}")
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"Finished execution of: {func_name} - Duration: {duration:.4f} seconds")
        return wrapper
