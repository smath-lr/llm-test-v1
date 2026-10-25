import logging
from pathlib import Path


def init_logger(log_name: str, log_file: Path = None, level=logging.INFO):
    logger = logging.getLogger(log_name)
    if logger.handlers:
        return logger, []  # 已经初始化过，没有新增handler

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    new_handlers = []

    # 控制台输出handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    new_handlers.append(sh)

    # 文件输出handler
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        new_handlers.append(fh)

    return logger, new_handlers
