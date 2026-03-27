import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class JSONLineHandler(logging.Handler):

    def __init__(self, filepath: str):
        super().__init__()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(filepath, "a", buffering=1)

    def emit(self, record: logging.LogRecord):
        entry = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in logging.LogRecord.__dict__ and not k.startswith("_"):
                entry[k] = v
        try:
            self._file.write(json.dumps(entry) + "\n")
        except Exception:
            pass


def get_logger(name: str, log_file: str = "./logs/pharma_pipeline.jsonl",
               level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)

    logger.addHandler(JSONLineHandler(log_file))
    logger.propagate = False
    return logger


class Timer:
    def __init__(self, logger: logging.Logger, label: str, **extra):
        self._logger = logger
        self._label  = label
        self._extra  = extra
        self._start  = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 1)
        self._logger.info(
            f"{self._label} completed in {elapsed_ms}ms",
            extra={"elapsed_ms": elapsed_ms, "label": self._label, **self._extra}
        )