"""Structured logging for FemSolverClaude.

Usage
-----
    from femsolver.utils.logger import get_logger
    log = get_logger()
    log.info("Solved in 0.003 s")
    log.debug("Assembling element 42")

The logger name is ``"femsolver"`` by default.  The default level is ``INFO``.
Set the environment variable ``FEMSOLVER_LOG_LEVEL=DEBUG`` to enable debug
output, or call ``get_logger(level="DEBUG")`` explicitly.

Log format
----------
    [femsolver] INFO: Solved 'my_problem' in 0.003 s (42 DOFs)
    [femsolver] DEBUG: Assembling element 1
"""
from __future__ import annotations

import logging
import os
import sys

_DEFAULT_LEVEL = os.environ.get("FEMSOLVER_LOG_LEVEL", "INFO").upper()
_FMT = "[%(name)s] %(levelname)s: %(message)s"


def get_logger(name: str = "femsolver", level: str = _DEFAULT_LEVEL) -> logging.Logger:
    """Return (and configure on first call) the named logger.

    Calling this function multiple times with the same *name* is safe —
    handlers are only attached once.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger
