import logging
import sys
import torch
import os
import time


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


class LOG:
    def __init__(self):
        handler = logging.StreamHandler()
        self.formatter = logging.Formatter(
            '[%(asctime)s] %(filename)-8s :: %(message)s',
            datefmt='%m-%d %H:%M:%S')
        handler.setFormatter(self.formatter)
        self.LOGGER = logging.getLogger('global')
        self.LOGGER.addHandler(handler)
        self.LOGGER.setLevel(logging.INFO)

    def add_file_headler(self, save_path):
        save_path = os.path.join(save_path, f'log_{time_string()}.log')
        fhandler = logging.FileHandler(save_path, mode='w')
        fhandler.setFormatter(self.formatter)
        self.LOGGER.addHandler(fhandler)

    def print_baisic_info(self):
        self.LOGGER.info("python version : {}".format(sys.version.replace('\n', ' ')))
        self.LOGGER.info("torch  version : {}".format(torch.__version__))


LOGGER = LOG()

if __name__ == '__main__':
    log = LOG()
    log.add_file_headler('/home/yuming/reid_multi_net/src/reid_multi_net/debug')
    log.LOGGER.info('info message')
    log.LOGGER.warning('warning message')
    log.LOGGER.error('error message')
    log.print_baisic_info()
