import random

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler


class MemoryOverSampler(BatchSampler):

    def __init__(self, y, memory_flags, batch_size=128, **kwargs):
        self.indexes = self._oversample(y, memory_flags)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __iter__(self):
        np.random.shuffle(self.indexes)

        for batch_index in range(len(self)):
            low_index = batch_index * self.batch_size
            high_index = (batch_index + 1) * self.batch_size

            yield self.indexes[low_index:high_index].tolist()

    def _oversample(self, y, memory_flags):
        old_indexes = np.where(memory_flags == 1.)[0]
        new_indexes = np.where(memory_flags == 0.)[0]

        old, new = y[old_indexes], y[new_indexes]

        old_qt = self._mean_quantity(old)
        new_qt = self._mean_quantity(new)

        assert new_qt > old_qt, (new_qt, old_qt)
        factor = new_qt / old_qt

        indexes = [np.where(memory_flags == 0)[0]]
        for class_id in np.unique(y):
            indexes.append(np.repeat(np.where(old == class_id)[0], factor))

        indexes = np.concatenate(indexes)
        return indexes

    @staticmethod
    def _mean_quantity(y):
        return np.mean(np.bincount(y))


class MultiSampler(BatchSampler):
    """Sample same batch several times. Every time it's a little bit different
    due to data augmentation. To be used with ensembling models."""

    def __init__(self, nb_samples, batch_size, factor=1, **kwargs):
        self.nb_samples = nb_samples
        self.factor = factor
        self.batch_size = batch_size

    def __len__(self):
        return len(self.y) / self.batch_size

    def __iter__(self):
        pass


class TripletCKSampler(BatchSampler):
    """Samples positives pair that will be then be mixed in triplets.

    C = number of classes
    K = number of instances per class

    References:
        * Facenet: A unified embedding for face recognition and clustering
          Schroff et al.
          CVPR 2015.
    """

    def __init__(self, y, nb_per_class=4, nb_classes=20):
        assert len(np.unique(y)) >= nb_classes

        self.y = y
        self.nb_per_class = nb_per_class
        self.nb_classes = nb_classes

        self._classes = np.unique(y)
        self._class_to_indexes = {
            class_idx: np.where(y == class_idx)[0] for class_idx in self._classes
        }

    def __len__(self):
        return len(self.y) // (self.nb_per_class * self.nb_classes)

    def __iter__(self):
        for _ in range(len(self)):
            indexes = []

            classes = np.random.choice(self._classes, size=self.nb_classes, replace=False)
            for class_id in classes:
                class_indexes = np.random.choice(
                    self._class_to_indexes[class_id],
                    size=self.nb_per_class,
                    replace=bool(len(self._class_to_indexes[class_id]) < self.nb_per_class)
                )

                indexes.extend(class_indexes.tolist())

            yield indexes


class TripletSampler(BatchSampler):
    """Samples elements so that each batch is constitued by a third of anchor, a third
    of positive, and a third of negative.

    Reference:
        * Openface: A general-purpose face recognition library with mobile applications.
          Amos et al.
          2016
     """

    def __init__(self, y, batch_size=128):
        self.y = y
        self.batch_size = (batch_size // 3)
        print("Triplet Sampler has a batch size of {}.".format(3 * self.batch_size))

        self._classes = set(np.unique(y).tolist())
        self._class_to_indexes = {
            class_idx: np.where(y == class_idx)[0] for class_idx in self._classes
        }
        self._indexes = np.arange(len(y))

    def __len__(self):
        return len(self.y) // self.batch_size

    def __iter__(self):
        self._random_permut()

        for batch_index in range(len(self)):
            indexes = []

            for i in range(self.batch_size):
                anchor_index = self._indexes[batch_index * i]
                anchor_class = self.y[batch_index * i]

                pos_index = anchor_index
                while pos_index == anchor_index:
                    pos_index = np.random.choice(self._class_to_indexes[anchor_class])

                neg_class = np.random.choice(list(self._classes - set([anchor_class])))
                neg_index = np.random.choice(self._class_to_indexes[neg_class])

                indexes.append(anchor_index)
                indexes.append(pos_index)
                indexes.append(neg_index)

            yield indexes

    def _random_permut(self):
        shuffled_indexes = np.random.permutation(len(self.y))
        self.y = self.y[shuffled_indexes]
        self._indexes = self._indexes[shuffled_indexes]


class NPairSampler(BatchSampler):

    def __init__(self, y, n_classes=10, n_samples=2, **kwargs):
        self.y = y
        self.n_classes = n_classes
        self.n_samples = n_samples

        self._classes = np.sort(np.unique(y))
        self._distribution = np.bincount(y) / np.bincount(y).sum()
        self._distribution = self._distribution[self._distribution > 0]
        self._batch_size = self.n_samples * self.n_classes

        self._class_to_indexes = {
            class_index: np.where(y == class_index)[0] for class_index in self._classes
        }

        self._class_counter = {class_index: 0 for class_index in self._classes}

        self.classes_per_batch = []
        self.first = True
        for indexes in self._class_to_indexes.values():
            np.random.shuffle(indexes)

        count = 0
        while count + self._batch_size < len(self.y):
            classes = np.random.choice(
                self._classes, self.n_classes, replace=False, p=self._distribution
            )
            self.classes_per_batch.append(classes)
            count += self.n_classes * self.n_samples

    def re_generate(self):
        self.classes_per_batch = []
        for indexes in self._class_to_indexes.values():
            np.random.shuffle(indexes)
        count = 0

        while count + self._batch_size < len(self.y):
            classes = np.random.choice(
                self._classes, self.n_classes, replace=False, p=self._distribution
            )
            self.classes_per_batch.append(classes)
            count += self.n_classes * self.n_samples

    def __iter__(self):

        if self.first:
            self.first = False
        else:
            self.re_generate()
        count = 0
        for classes in self.classes_per_batch:
            if not count + self._batch_size < len(self.y):
                break
            batch_indexes = []

            for class_index in classes:
                class_counter = self._class_counter[class_index]
                class_indexes = self._class_to_indexes[class_index]

                class_batch_indexes = class_indexes[class_counter:class_counter + self.n_samples]
                batch_indexes.extend(class_batch_indexes)

                self._class_counter[class_index] += self.n_samples

                if self._class_counter[class_index] + self.n_samples > len(
                        self._class_to_indexes[class_index]
                ):
                    np.random.shuffle(self._class_to_indexes[class_index])
                    self._class_counter[class_index] = 0

            yield batch_indexes

            count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.y) // self._batch_size


class AuxSampler(BatchSampler):
    def __init__(self, y, batch_size, n_sample, farther_dataset_batch_size, farther_dataset_idx,
                 farther_dataset_targets,
                 farther_dataset_mem_flag, same_class=False, farther_is_mem=False,
                 father_classes_per_batch=None):
        self._batch_size = batch_size
        self._farther_dataset_batch_size = farther_dataset_batch_size
        self.y = np.asarray(y)
        self.unique_cls = np.unique(self.y)
        self.n_samples = n_sample
        self.same_class = same_class
        self.current_classes = []
        self.farther_dataset_idx = farther_dataset_idx
        self.farther_dataset_batch_size = farther_dataset_batch_size
        self.farther_dataset_targets = farther_dataset_targets
        self.farther_dataset_mem_flag = farther_dataset_mem_flag
        self.father_classes_per_batch = father_classes_per_batch
        self.farther_is_mem = farther_is_mem
        unique_class_per_batch = []

        all_p_classes = self.current_unique_classes
        if father_classes_per_batch is None:
            for i in range(len(farther_dataset_idx) // farther_dataset_batch_size):
                current_idx = farther_dataset_idx[i * farther_dataset_batch_size:(i + 1) * farther_dataset_batch_size]
                current_targets = farther_dataset_targets[current_idx]
                current_mem_flags = farther_dataset_mem_flag[current_idx]
                target_mode = 1 if farther_is_mem else 0
                current_classes = np.unique(current_targets[current_mem_flags == target_mode])
                if not self.same_class:
                    valid_classes = torch.tensor(list(set(all_p_classes.tolist()) - set(current_classes.tolist())))
                    iidx = torch.randperm(valid_classes.shape[0])
                    valid_classes = valid_classes[iidx[:current_classes.shape[0]]]
                else:
                    valid_classes = current_classes
                unique_class_per_batch.append(valid_classes)

            self.unique_class_per_batch = unique_class_per_batch
        else:
            if self.same_class:
                self.unique_class_per_batch = father_classes_per_batch
            else:
                self.unique_class_per_batch = []
                for current_classes in father_classes_per_batch:
                    valid_classes = torch.tensor(list(set(all_p_classes.tolist()) - set(current_classes.tolist())))
                    iidx = torch.randperm(valid_classes.shape[0])
                    valid_classes = valid_classes[iidx[:current_classes.shape[0]]]
                    self.unique_class_per_batch.append(valid_classes)

    def re_generate(self, farther_dataset_idx, father_classes_per_batch=None):
        all_p_classes = self.current_unique_classes
        unique_class_per_batch = []
        if father_classes_per_batch is None:
            for i in range(len(farther_dataset_idx) // self.farther_dataset_batch_size):
                current_idx = farther_dataset_idx[i * self.farther_dataset_batch_size:(i + 1) * self.farther_dataset_batch_size]
                current_targets = self.farther_dataset_targets[current_idx]
                current_mem_flags = self.farther_dataset_mem_flag[current_idx]
                target_mode = 1 if self.farther_is_mem else 0
                current_classes = np.unique(current_targets[current_mem_flags == target_mode])
                if not self.same_class:
                    valid_classes = torch.tensor(list(set(all_p_classes.tolist()) - set(current_classes.tolist())))
                    iidx = torch.randperm(valid_classes.shape[0])
                    valid_classes = valid_classes[iidx[:current_classes.shape[0]]]
                else:
                    valid_classes = current_classes
                unique_class_per_batch.append(valid_classes)

            self.unique_class_per_batch = unique_class_per_batch
        else:
            if self.same_class:
                self.unique_class_per_batch = father_classes_per_batch
            else:
                self.unique_class_per_batch = []
                for current_classes in father_classes_per_batch:
                    valid_classes = torch.tensor(list(set(all_p_classes.tolist()) - set(current_classes.tolist())))
                    iidx = torch.randperm(valid_classes.shape[0])
                    valid_classes = valid_classes[iidx[:current_classes.shape[0]]]
                    self.unique_class_per_batch.append(valid_classes)

    @property
    def current_unique_classes(self):
        return torch.from_numpy(np.unique(self.y))

    @property
    def return_n_classes(self):
        return int(self._batch_size / self.n_samples)

    def __len__(self):
        return len(self.y)

    def __iter__(self):
        for current_classes in self.unique_class_per_batch:
            ret = []
            n_classes = self.return_n_classes
            candidate_cls = np.random.choice(self.unique_cls, n_classes)
            if len(current_classes):
                candidate_cls = np.random.choice(current_classes, min(n_classes, len(current_classes)), replace=False)
            for cls in candidate_cls:
                cls_mask = self.y == cls
                res_idx = np.where(cls_mask == True)[0]
                if not len(res_idx):
                    continue
                ret.extend(np.random.choice(res_idx, self.n_samples, replace=True).tolist())
            if not len(ret):
                # when there is no corresponding images in unlabeled dataset,
                # randomly select images to continue sampling.
                ret = np.random.choice(np.arange(len(self.y)), self.n_samples * len(candidate_cls),
                                       replace=True).tolist()
            yield ret


class MyRandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.first = True

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))
        n = len(self.data_source)
        if self.replacement:
            self._iter_list = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist()
        else:
            self._iter_list = torch.randperm(n).tolist()

    def re_generate(self):
        n = len(self.data_source)
        if self.replacement:
            self._iter_list = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist()
        else:
            self._iter_list = torch.randperm(n).tolist()

    @property
    def iter_list(self):
        return self._iter_list

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        if self.first:
            self.first = False
            return iter(self._iter_list)
        else:
            self.re_generate()
            return iter(self._iter_list)

    def __len__(self):
        return self.num_samples
