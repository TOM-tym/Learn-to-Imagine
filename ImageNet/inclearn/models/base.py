import abc
import logging
import os

import torch

LOGGER = logging.Logger("IncLearn", level="INFO")

from inclearn.utils import LOGGER as logger


class IncrementalLearner(abc.ABC):
    """Base incremental learner.

    Methods are called in this order (& repeated for each new task):

    1. set_task_info
    2. before_task
    3. train_task
    4. after_task
    5. eval_task
    """

    def __init__(self, *args, **kwargs):
        self._network = None
        self.frozen_model = None

    def set_task_info(self, task_info):
        self._task = task_info["task"]
        self._total_n_classes = task_info["total_n_classes"]
        self._task_size = task_info["increment"]
        self._n_train_data = task_info["n_train_data"]
        self._n_test_data = task_info["n_test_data"]
        self._n_tasks = task_info["max_task"]

    def before_task(self, train_loader, val_loader):
        logger.LOGGER.info("Before task")
        self.eval()
        self._before_task(train_loader, val_loader,taskid=self._task)

    def train_task(self, train_loader, pseudo_memory_loader, val_loader, train_generator_config=None,
                   train_generator_data=None, freeze_layers=False):
        logger.LOGGER.info("train task")
        if not train_generator_config:
            self.train()
        else:
            self.eval()
        self._train_task(train_loader, pseudo_memory_loader, val_loader, train_generator_config,
                         train_generator_data=train_generator_data, freeze_layers=freeze_layers)

    def after_task_intensive(self, inc_dataset, **kwargs):
        logger.LOGGER.info("after task")
        self.eval()
        self._after_task_intensive(inc_dataset, **kwargs)

    def after_task(self, inc_dataset):
        logger.LOGGER.info("after task")
        self.eval()
        self._after_task(inc_dataset)

    def eval_task(self, data_loader):
        logger.LOGGER.info("eval task")
        self.eval()
        return self._eval_task(data_loader)

    def get_memory(self):
        return None

    def get_val_memory(self):
        return None

    def _before_task(self, data_loader, val_loader, taskid):
        pass

    def _train_task(self, train_loader, pseudo_memory_loader, val_loader,
                    train_generator_config=None, train_generator_data=None, freeze_layers=False):
        raise NotImplementedError

    def _after_task_intensive(self, data_loader, *kwargs):
        pass

    def _after_task(self, data_loader):
        pass

    def _eval_task(self, data_loader):
        raise NotImplementedError

    def save_metadata(self, directory, run_id):
        pass

    def load_metadata(self, directory, run_id):
        pass

    @property
    def _new_task_index(self):
        return self._task * self._task_size

    @property
    def inc_dataset(self):
        return self.__inc_dataset

    @inc_dataset.setter
    def inc_dataset(self, inc_dataset):
        self.__inc_dataset = inc_dataset

    @property
    def network(self):
        return self._network

    def save_parameters(self, directory, run_id):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth")
        logger.LOGGER.info(f"Saving model at {path}.")
        torch.save(self.network.state_dict(), path)

    def load_parameters(self, directory, run_id, device):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth")
        if not os.path.exists(path):
            logger.LOGGER.warning(f"Model path '{path}'not exists! Ignoring...")
            return False

        logger.LOGGER.info(f"Loading model at {path}.")
        try:
            self.network.load_state_dict(torch.load(path, map_location=device))
        except Exception as e:
            logger.LOGGER.warning(f"Old method to save weights, it's deprecated! {e}")
            self._network = torch.load(path, map_location=device)
        return True

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
