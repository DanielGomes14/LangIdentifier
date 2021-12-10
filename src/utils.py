import sys
import logging
import os


def open_file(fname, option='r', encoding='utf-8'):
    try:
        return open(fname, option, encoding=encoding)
    except FileNotFoundError:
        logging.error(f"Could not open file {fname}")
        sys.exit(1)


def open_dir(dname):
    try:
        return os.listdir(dname)
    except:
        logging.info(f"Directory {dname} not found")
        sys.exit(1)