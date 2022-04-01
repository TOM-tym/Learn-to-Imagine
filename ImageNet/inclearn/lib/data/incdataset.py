import logging
import random
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from .datasets import (
    APY, CUB200, LAD, AwA2, ImageNet100, ImageNet100UCIR, ImageNet1000, TinyImageNet200, iCIFAR10,
    iCIFAR100, get_imagenet_32_datasets
)

from inclearn.lib.data.samplers import AuxSampler, MyRandomSampler
from inclearn.utils import LOGGER as logger
from inclearn.lib.data.datasets import ImageNet900
from torch.utils.data import DataLoader


class IncrementalDataset:
    """Incremental generator of datasets.

    :param dataset_name: Among a list of available dataset, that can easily
                         be defined (see at file's end).
    :param random_order: Shuffle the class ordering, else use a cherry-picked
                         ordering.
    :param shuffle: Shuffle batch order between epochs.
    :param workers: Number of workers loading the data.
    :param batch_size: The batch size.
    :param seed: Seed to force determinist class ordering.
    :param increment: Number of class to add at each task.
    :param validation_split: Percent of training data to allocate for validation.
    :param onehot: Returns targets encoded as onehot vectors instead of scalars.
                   Memory is expected to be already given in an onehot format.
    :param initial_increment: Initial increment may be defined if you want to train
                              on more classes than usual for the first task, like
                              UCIR does.
    """

    def __init__(
            self,
            dataset_name,
            random_order=False,
            shuffle=True,
            workers=10,
            batch_size=128,
            seed=1,
            increment=10,
            validation_split=0.,
            onehot=False,
            initial_increment=None,
            sampler=None,
            sampler_config=None,
            data_path="data",
            class_order=None,
            dataset_transforms=None,
            all_test_classes=False,
            metadata_path=None,
            extra_dataset_name='imagenet_32'
    ):
        datasets = _get_datasets(dataset_name)
        if metadata_path:
            print("Adding metadata path {}".format(metadata_path))
            datasets[0].metadata_path = metadata_path

        self._setup_data(
            datasets,
            random_order=random_order,
            class_order=class_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split,
            initial_increment=initial_increment,
            data_path=data_path,
            extra_dataset=extra_dataset_name,
        )

        dataset = datasets[0]()
        dataset.set_custom_transforms(dataset_transforms)
        self.train_transforms = dataset.train_transforms  # FIXME handle multiple datasets
        self.test_transforms = dataset.test_transforms
        self.common_transforms = dataset.common_transforms

        self.open_image = datasets[0].open_image

        self._current_task = 0

        self._seed = seed
        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self._onehot = onehot
        self._sampler = sampler
        self._sampler_config = sampler_config
        self._all_test_classes = all_test_classes
        self.extra_dataset_name = extra_dataset_name

    @property
    def n_tasks(self):
        return len(self.increments)

    @property
    def n_classes(self):
        return sum(self.increments)

    def new_task(self, memory=None, pseudo_memory=None, memory_val=None, pseudo_memory_usage='mixed',
                 pseudo_memory_n_samples=2, pseudo_same_class=False):
        assert pseudo_memory_usage in ('pure', 'mixed', 'enhanced')
        logger.LOGGER.info(f'Pseudo memory n_samples is {pseudo_memory_n_samples}.')
        pseudo_memory_loader = None

        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")

        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])

        x_train, y_train = self._select(
            self.data_train, self.targets_train, low_range=min_class, high_range=max_class
        )
        pure_new_data = (x_train, y_train)
        nb_new_classes = len(np.unique(y_train))
        x_val, y_val = self._select(
            self.data_val, self.targets_val, low_range=min_class, high_range=max_class
        )
        if self._all_test_classes is True:
            logger.LOGGER.info("Testing on all classes!")
            x_test, y_test = self._select(
                self.data_test, self.targets_test, high_range=sum(self.increments)
            )
        elif self._all_test_classes is not None or self._all_test_classes is not False:
            max_class = sum(self.increments[:self._current_task + 1 + self._all_test_classes])
            logger.LOGGER.info(
                f"Testing on {self._all_test_classes} unseen tasks (max class = {max_class})."
            )
            x_test, y_test = self._select(self.data_test, self.targets_test, high_range=max_class)
        else:
            x_test, y_test = self._select(self.data_test, self.targets_test, high_range=max_class)

        if self._onehot:
            def to_onehot(x):
                n = np.max(x) + 1
                return np.eye(n)[x]

            y_train = to_onehot(y_train)

        if memory is not None:
            logger.LOGGER.info("Set memory of size: {}.".format(memory[0].shape[0]))
            x_train, y_train, train_memory_flags = self._add_memory(x_train, y_train, *memory)
        else:
            train_memory_flags = np.zeros((x_train.shape[0],))

        if memory_val is not None:
            logger.LOGGER.info("Set validation memory of size: {}.".format(memory_val[0].shape[0]))
            x_val, y_val, val_memory_flags = self._add_memory(x_val, y_val, *memory_val)
        else:
            val_memory_flags = np.zeros((x_val.shape[0],))

        train_loader = self.get_loader(x_train, y_train, train_memory_flags, mode="train")
        val_loader = self.get_loader(x_val, y_val, val_memory_flags,
                                     mode="train") if len(x_val) > 0 else None
        test_loader = self.get_loader(x_test, y_test, np.zeros((x_test.shape[0],)), mode="test")
        if self.extra_dataset_name == 'imagenet_32':
            aux_loader = DataLoader(self.aux_data, batch_size=self._batch_size, num_workers=self._workers,
                                    shuffle=False,
                                    drop_last=False)
        elif self.extra_dataset_name == 'imagenet900':
            aux_loader = self.get_loader(np.asarray(self.aux_data.data), np.asarray(self.aux_data.targets),
                                         np.zeros((np.array(self.aux_data.targets).shape[0])), mode="test",
                                         open_image=True, batch_size_expect=128)
        else:
            raise NameError

        if pseudo_memory is not None:
            if pseudo_memory_usage == 'pure':
                logger.LOGGER.info(
                    "Added pseudo memory of size: {} to the training set.".format(pseudo_memory[0].shape[0]))
                x_train, y_train, train_memory_flags = self._add_memory(x_train, y_train, *pseudo_memory,
                                                                        unlabeled=False)
            elif pseudo_memory_usage == 'mixed':
                # get sampler
                pseudo_memory_loader = self.get_pseudo_memory_loader(pseudo_memory, pseudo_memory_n_samples,
                                                                     batch_size=self._batch_size,
                                                                     farther_dataset_batch_size=self._batch_size,
                                                                     farther_dataset_idx=train_loader.sampler.iter_list,
                                                                     farther_dataset_targets=y_train,
                                                                     farther_dataset_mem_flag=train_memory_flags,
                                                                     same_class=pseudo_same_class,
                                                                     farther_is_mem=True)
        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "total_n_classes": sum(self.increments),
            "increment": nb_new_classes,  # self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": x_train.shape[0],
            "n_test_data": x_test.shape[0]
        }

        self._current_task += 1

        return task_info, train_loader, val_loader, test_loader, aux_loader, pseudo_memory_loader, pure_new_data

    def get_pseudo_memory_loader(self, pseudo_memory, pseudo_memory_n_samples, batch_size, farther_dataset_batch_size,
                                 farther_dataset_idx, farther_dataset_targets, farther_dataset_mem_flag, same_class,
                                 farther_is_mem, father_classes_per_batch=None):
        aux_sampler = AuxSampler(pseudo_memory[1], batch_size=batch_size, n_sample=pseudo_memory_n_samples,
                                 farther_dataset_batch_size=farther_dataset_batch_size,
                                 farther_dataset_idx=farther_dataset_idx,
                                 farther_dataset_targets=farther_dataset_targets,
                                 farther_dataset_mem_flag=farther_dataset_mem_flag, same_class=same_class,
                                 farther_is_mem=farther_is_mem,
                                 father_classes_per_batch = father_classes_per_batch
                                 )
        pseudo_memory_loader = self.get_loader(*pseudo_memory, memory_flags=np.zeros(pseudo_memory[0].shape),
                                               sampler=aux_sampler, sampler_init=False)
        return pseudo_memory_loader

    def _add_memory(self, x, y, data_memory, targets_memory, previous_memory_flags=None, unlabeled=False):
        if self._onehot:  # Need to add dummy zeros to match the number of targets:
            targets_memory = np.concatenate(
                (
                    targets_memory,
                    np.zeros((targets_memory.shape[0], self.increments[self._current_task]))
                ),
                axis=1
            )
        flag = 1 if not unlabeled else 2
        if previous_memory_flags is None:
            memory_flags = np.concatenate((np.zeros((x.shape[0],)), flag * np.ones((data_memory.shape[0],))))
        else:
            memory_flags = np.concatenate((previous_memory_flags, flag * np.ones((data_memory.shape[0],))))

        x = np.concatenate((x, data_memory))
        y = np.concatenate((y, targets_memory))

        return x, y, memory_flags

    def get_custom_loader(
            self, class_indexes, memory=None, mode="test", data_source="train", sampler=None
    ):
        """Returns a custom loader.

        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if not isinstance(class_indexes, list):  # TODO: deprecated, should always give a list
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(
                x, y, low_range=class_index, high_range=class_index + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        if len(data) == 0:
            assert memory is not None
        else:
            data = np.concatenate(data)
            targets = np.concatenate(targets)

        if (not isinstance(memory, tuple) and
            memory is not None) or (isinstance(memory, tuple) and memory[0] is not None):
            if len(data) > 0:
                data, targets, memory_flags = self._add_memory(data, targets, *memory)
            else:
                data, targets = memory
                memory_flags = np.ones((data.shape[0],))
        else:
            memory_flags = np.zeros((data.shape[0],))

        return data, self.get_loader(
            data, targets, memory_flags, shuffle=False, mode=mode, sampler=sampler
        )

    def get_memory_loader(self, data, targets):
        return self.get_loader(
            data, targets, np.ones((data.shape[0],)), shuffle=True, mode="train"
        )

    def _select(self, x, y, low_range=0, high_range=0):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def get_loader(self, x, y, memory_flags, shuffle=True, mode="train", sampler=None, sampler_init=True,
                   open_image=False, batch_size_expect=None, multi_processing=True):
        if mode == "train":
            trsf = transforms.Compose([*self.train_transforms, *self.common_transforms])
        elif mode == "test":
            trsf = transforms.Compose([*self.test_transforms, *self.common_transforms])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=1.), *self.test_transforms,
                    *self.common_transforms
                ]
            )
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))

        sampler = sampler or self._sampler
        num_workers = self._workers
        if sampler is not None and mode == "train" and sampler_init:
            logger.LOGGER.info("Using sampler {}".format(sampler))
            sampler = sampler(y, memory_flags, batch_size=self._batch_size, **self._sampler_config)
            batch_size = 1
            return DataLoader(
                DummyDataset(x, y, memory_flags, trsf, open_image=self.open_image),
                num_workers=num_workers,
                batch_sampler=sampler,
            )
        elif sampler is not None and mode == "train":
            sampler = sampler
            batch_size = 1
            return DataLoader(
                DummyDataset(x, y, memory_flags, trsf, open_image=self.open_image),
                num_workers=num_workers,
                batch_sampler=sampler
            )
        elif sampler is not None and mode == "test" and not sampler_init:
            sampler = sampler
            batch_size = 1
            return DataLoader(
                DummyDataset(x, y, memory_flags, trsf, open_image=self.open_image),
                num_workers=num_workers,
                batch_sampler=sampler
            )
        elif mode == 'test' or mode == 'flip':
            if batch_size_expect is not None:
                batch_size = batch_size_expect
                num_workers = 8
            else:
                batch_size = self._batch_size
                num_workers = 4
            return DataLoader(
                DummyDataset(x, y, memory_flags, trsf, open_image=self.open_image),
                batch_size=batch_size,
                num_workers=num_workers,
            )
        else:  # 'mode' == train, shuffle = True, sample = None
            # we replace custom sampler here
            sampler = MyRandomSampler(data_source=y)
            batch_size = self._batch_size
            return DataLoader(
                DummyDataset(x, y, memory_flags, trsf, open_image=self.open_image),
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=sampler,
                drop_last=True
            )

    def _setup_data(
            self,
            datasets,
            random_order=False,
            class_order=None,
            seed=1,
            increment=10,
            validation_split=0.,
            initial_increment=None,
            data_path="data",
            extra_dataset='imagenet_32'
    ):
        # FIXME: handles online loading of images
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.increments = []
        self.class_order = []

        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:
            train_dataset = dataset().base_dataset(data_path, train=True, download=True)
            test_dataset = dataset().base_dataset(data_path, train=False, download=True)

            x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
            x_val, y_val, x_train, y_train = self._split_per_class(
                x_train, y_train, validation_split
            )
            x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

            order = list(range(len(np.unique(y_train))))
            if random_order:
                random.seed(seed)  # Ensure that following order is determined by seed:
                random.shuffle(order)
            elif class_order:
                order = class_order
            elif dataset.class_order is not None:
                order = dataset.class_order
            elif train_dataset.class_order is not None:
                order = train_dataset.class_order

            logger.LOGGER.info("Dataset {}: class ordering: {}.".format(dataset.__name__, order))

            self.class_order.append(order)

            y_train = self._map_new_class_index(y_train, order)
            y_val = self._map_new_class_index(y_val, order)
            y_test = self._map_new_class_index(y_test, order)

            y_train += current_class_idx
            y_val += current_class_idx
            y_test += current_class_idx

            current_class_idx += len(order)
            if len(datasets) > 1:
                self.increments.append(len(order))
            elif initial_increment is None:
                nb_steps = len(order) / increment
                remainder = len(order) - int(nb_steps) * increment

                if not nb_steps.is_integer():
                    logger.LOGGER.warning(
                        f"THe last step will have sligthly less sample ({remainder} vs {increment})."
                    )
                    self.increments = [increment for _ in range(int(nb_steps))]
                    self.increments.append(remainder)
                else:
                    self.increments = [increment for _ in range(int(nb_steps))]
            else:
                self.increments = [initial_increment]

                nb_steps = (len(order) - initial_increment) / increment
                remainder = (len(order) - initial_increment) - int(nb_steps) * increment
                if not nb_steps.is_integer():
                    logger.LOGGER.warning(
                        f"THe last step will have sligthly less sample ({remainder} vs {increment})."
                    )
                    self.increments.extend([increment for _ in range(int(nb_steps))])
                    self.increments.append(remainder)
                else:
                    self.increments.extend([increment for _ in range(int(nb_steps))])

            self.data_train.append(x_train)
            self.targets_train.append(y_train)
            self.data_val.append(x_val)
            self.targets_val.append(y_val)
            self.data_test.append(x_test)
            self.targets_test.append(y_test)

        self.data_train = np.concatenate(self.data_train)
        self.targets_train = np.concatenate(self.targets_train)
        self.data_val = np.concatenate(self.data_val)
        self.targets_val = np.concatenate(self.targets_val)
        self.data_test = np.concatenate(self.data_test)
        self.targets_test = np.concatenate(self.targets_test)

        if extra_dataset == 'imagenet_32':
            self.aux_data = get_imagenet_32_datasets('imagenet_32', os.path.join(data_path, 'imagenet_32'))
        elif extra_dataset == 'imagenet900':
            ds = _get_datasets('imagenet900')[0]
            self.aux_data = ds().base_dataset(data_path, train=True, download=True)

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))

    @staticmethod
    def _split_per_class(x, y, validation_split=0.):
        """Splits train data for a subset of validation data.

        Split is done so that each class has a much data.
        """
        shuffled_indexes = np.random.permutation(x.shape[0])
        x = x[shuffled_indexes]
        y = y[shuffled_indexes]

        x_val, y_val = [], []
        x_train, y_train = [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val.append(x[val_indexes])
            y_val.append(y[val_indexes])
            x_train.append(x[train_indexes])
            y_train.append(y[train_indexes])

        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        return x_val, y_val, x_train, y_train


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, memory_flags, trsf, open_image=False):
        self.x, self.y = x, y
        self.memory_flags = memory_flags
        self.trsf = trsf
        self.open_image = open_image

        assert x.shape[0] == y.shape[0] == memory_flags.shape[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        memory_flag = self.memory_flags[idx]

        if self.open_image:
            img = Image.open(x.item()).convert("RGB")
        else:
            img = Image.fromarray(x.astype("uint8"))

        img = self.trsf(img)
        return {"inputs": img, "targets": y, "memory_flags": memory_flag}


def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif dataset_name == "imagenet100":
        return ImageNet100
    elif dataset_name == "imagenet900":
        return ImageNet900
    elif dataset_name == "imagenet100ucir":
        return ImageNet100UCIR
    elif dataset_name == "imagenet1000":
        return ImageNet1000
    elif dataset_name == "tinyimagenet":
        return TinyImageNet200
    elif dataset_name == "awa2":
        return AwA2
    elif dataset_name == "cub200":
        return CUB200
    elif dataset_name == "apy":
        return APY
    elif dataset_name == "lad":
        return LAD
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))
