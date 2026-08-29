import logging
import sys

def configure_logging(level:str = "INFO") -> None:
    """
    Called once at process startup - no left as a basicConfig() call
    buried inside a throwaway script.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout
    )