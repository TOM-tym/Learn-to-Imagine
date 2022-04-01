import collections
import copy
import logging
import os
import pickle

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from inclearn.lib import factory, herding, losses, network, schedulers, utils
from inclearn.models.base import IncrementalLearner

EPSILON = 1e-8

from inclearn.utils import LOGGER as logger


class ICarl(IncrementalLearner):
    """Implementation of iCarl. Modify from https://github.com/arthurdouillard/incremental_learning.pytorch

    # References:
    - iCaRL: Incremental Classifier and Representation Learning
      Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
      https://arxiv.org/abs/1611.07725

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args):
        super().__init__()

        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._warmup_config = args.get("warmup", {})
        if self._warmup_config and self._warmup_config["total_epoch"] > 0:
            self._lr /= self._warmup_config["multiplier"]

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._n_classes = 0
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._last_results = None
        self._validation_percent = args["validation"]

        self._rotations_config = args.get("rotations_config", {})
        self._random_noise_config = args.get("random_noise_config", {})

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {
                "type": "fc",
                "use_bias": True
            }),
            device=self._device,
            extract_no_act=True,
            classifier_no_act=False,
            rotations_predictor=bool(self._rotations_config)
        )

        self._examplars = {}
        self._means = None
        self._herding_indexes = []
        self._data_memory, self._targets_memory = None, None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._epoch_metrics = collections.defaultdict(list)

        self._class_means = None
        self._generator_config = args.get("generator_config", {})
        self._train_generator_config = self._generator_config.get("train_config", {})
        self._train_generator_label_batchsize = self._train_generator_config.get("train_generator_label_batchsize", 128)
        self._generator_scheduling = self._train_generator_config.get('scheduling')
        self._generator_lr_decay = self._train_generator_config.get('lr_decay')
        self._generator_optimizer = None
        self._n_epochs_generator = self._train_generator_config.get("epoch", 40)
        self._generator_cos_margin = self._generator_config.get("cos_margin", 0.8)
        self._use_same_class = args.get("use_same_classes", True)
        self._generator_use_same_class = self._generator_config.get("use_same_classes", True)
        if self._generator_config:
            self.class_encoders = {}

        self._diff_cls_mapping = {}


    def save_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")

        logger.LOGGER.info("Saving metadata at {}.".format(path))
        with open(path, "wb+") as f:
            pickle.dump(
                [self._data_memory, self._targets_memory, self._herding_indexes, self._class_means],
                f
            )

    def load_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")
        if not os.path.exists(path):
            return

        logger.LOGGER.info("Loading metadata at {}.".format(path))
        with open(path, "rb") as f:
            self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = pickle.load(
                f
            )

    @property
    def epoch_metrics(self):
        return dict(self._epoch_metrics)

    # ----------
    # Public API
    # ----------

    def _before_task(self, train_loader, val_loader, taskid):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)
        logger.LOGGER.info("Now {} examplars per class.".format(self._memory_per_class))

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )
        base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )
        self._generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._generator_optimizer, self._scheduling, gamma=self._lr_decay
        )
        if self._warmup_config:
            if self._warmup_config.get("only_first_step", True) and self._task != 0:
                pass
            else:
                logger.LOGGER.info("Using WarmUp")
                self._scheduler = schedulers.GradualWarmupScheduler(
                    optimizer=self._optimizer,
                    after_scheduler=base_scheduler,
                    **self._warmup_config
                )
        else:
            self._scheduler = base_scheduler

    def _train_task(self, train_loader, pseudo_memory_loader, val_loader, train_generator_config=None,
                    train_generator_data=None, freeze_layers=False):
        logger.LOGGER.debug("nb {}.".format(len(train_loader.dataset)))
        n_epochs = self._n_epochs_generator if train_generator_config else self._n_epochs
        self._training_step(train_loader, pseudo_memory_loader, val_loader, 0, n_epochs, train_generator_config,
                            train_generator_data=None)

    def _training_step(self, train_loader, pseudo_memory_loader, val_loader, initial_epoch, nb_epochs, fine_tune=False,
                       record_bn=True, clipper=None, train_generator_config=None, train_generator_data=None):
        best_epoch, best_acc = -1, -1.
        wait = 0

        grad, act = None, None
        if len(self._multiple_devices) > 1:
            logger.LOGGER.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
            training_network = nn.DataParallel(self._network, self._multiple_devices)
        else:
            training_network = self._network

        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            if epoch == nb_epochs - 1 and record_bn and len(self._multiple_devices) == 1 and \
                    hasattr(training_network.convnet, "record_mode"):
                logger.LOGGER.info("Recording BN means & vars for MCBN...")
                training_network.convnet.clear_records()
                training_network.convnet.record_mode()
            if train_generator_config is not None:
                train_loader = train_generator_data['labeled_loader']
            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]
                unlabeled_aux_data_dict = None
                memory_dict = None
                if pseudo_memory_loader is not None:
                    memory_mask = memory_flags == 1
                    new_mask = memory_flags == 0
                    if train_generator_config is not None:
                        used_targets = targets[new_mask]
                        unique_current_cls = torch.unique(used_targets)
                    else:
                        unique_current_cls = torch.unique(targets[memory_mask])
                    use_same_classes = self._generator_use_same_class
                    all_p_classes = pseudo_memory_loader.batch_sampler.current_unique_classes
                    valid_classes = torch.tensor(list(set(all_p_classes.tolist()) - set(unique_current_cls.tolist())))
                    if use_same_classes or not len(valid_classes):
                        pseudo_memory_loader.batch_sampler.point_current_classes(unique_current_cls)
                    else:
                        pseudo_memory_loader.batch_sampler.point_current_classes(valid_classes)

                    unlabeled_aux_iter = iter(pseudo_memory_loader)
                    unlabeled_aux_data_dict = next(unlabeled_aux_iter)
                    if train_generator_config is not None:
                        memory_loader = train_generator_data['memory_loader']  # type:torch.utils.DataLoader
                        memory_loader.batch_sampler.point_current_classes(unique_current_cls)
                        memory_iter = iter(memory_loader)
                        memory_dict = next(memory_iter)
                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer.zero_grad()
                if train_generator_config:
                    self._generator_optimizer.zero_grad()
                loss = self._forward_loss(training_network, inputs, targets, memory_flags,
                                          unlabeled_aux_data_dict, fine_tune=fine_tune, gradcam_grad=grad,
                                          gradcam_act=act, train_generator_config=train_generator_config,
                                          use_generator=bool(self._generator_config), memory_data=memory_dict)
                if loss > 1e-10:
                    if train_generator_config is not None:
                        loss.backward()
                        self._generator_optimizer.step()
                        # self._optimizer.step()
                    else:
                        loss.backward()
                        self._optimizer.step()
                if clipper:
                    training_network.apply(clipper)

                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            if self._scheduler and train_generator_config is None:
                self._scheduler.step(epoch)
            if train_generator_config is not None:
                self._generator_scheduler.step(epoch)

            if self._eval_every_x_epochs and epoch != 0 and epoch % self._eval_every_x_epochs == 0:
                self._network.eval()
                self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                ytrue, ypred = self._eval_task(val_loader)
                acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
                logger.LOGGER.info("Val accuracy: {}".format(acc))
                self._network.train()

                if acc > best_acc:
                    best_epoch = epoch
                    best_acc = acc
                    wait = 0
                else:
                    wait += 1

                if self._early_stopping and self._early_stopping["patience"] > wait:
                    logger.LOGGER.warning("Early stopping!")
                    break

        if self._eval_every_x_epochs:
            logger.LOGGER.info("Best accuracy reached at epoch {} with {}%.".format(best_epoch, best_acc))

        if len(self._multiple_devices) == 1 and hasattr(training_network.convnet, "record_mode"):
            training_network.convnet.normal_mode()

    def _print_metrics(self, prog_bar, epoch, nb_epochs, nb_batches):
        pretty_metrics = ", ".join(
            "{}: {}".format(metric_name, round(metric_value / nb_batches, 3))
            for metric_name, metric_value in self._metrics.items()
        )

        prog_bar.set_description(
            "T{}/{}, E{}/{} => {}".format(
                self._task + 1, self._n_tasks, epoch + 1, nb_epochs, pretty_metrics
            )
        )

    def mix_features_generator(self, inputs, memory_flags, outputs, mem_out, targets, mem_targets, training_network,
                               unlabeled_data, train_generator=False):
        unlabeled_img, unlabeled_pseudo_label = unlabeled_data['inputs'], unlabeled_data['targets']
        unlabeled_img = unlabeled_img.to(self._device)
        unlabeled_output = training_network(unlabeled_img)
        labeled_data_mask = memory_flags == 0
        origin_raw_features = outputs['raw_features']
        origin_features = outputs['features']
        origin_logits = outputs['logits']
        origin_outputs_raw_logits = outputs['raw_logits']
        origin_stage2_feature_map = outputs['stage2_feature_map']
        if not train_generator:
            memory_targets = targets[~labeled_data_mask]
            if (~labeled_data_mask).sum():
                memory_feat_out_stage1 = torch.stack([i for i in outputs['attention'][2][~labeled_data_mask]])
            else:
                memory_feat_out_stage1 = []
        else:  # when training generator, we have memory data and labeled data inputted separately.
            memory_targets = mem_targets
            memory_feat_out_stage1 = torch.stack([i for i in mem_out['attention'][2]])

        unlabeled_feat_out_stage1 = torch.stack([i for i in unlabeled_output['attention'][2]], dim=0)
        mixed_feature_map = torch.tensor([]).to(self._device)
        mixed_targets = torch.tensor([], dtype=torch.long).to(self._device)
        mixed_mem_flag = torch.tensor([], dtype=torch.long)
        cycle_features = torch.tensor([]).to(self._device)
        cycle_targets = torch.tensor([], dtype=torch.long).to(self._device)
        labeled_features_to_be_mixed = memory_feat_out_stage1
        labeled_targets_to_be_mixed = memory_targets
        for cls in torch.unique(labeled_targets_to_be_mixed):
            cls = cls.item()
            cls_mask = labeled_targets_to_be_mixed == cls
            if not self._generator_use_same_class or not self._use_same_class:
                u_cls = np.random.choice(torch.unique(unlabeled_pseudo_label).cpu().numpy(), 1).item()
                self._diff_cls_mapping[cls] = u_cls
            else:
                u_cls = cls
            cls_u_mask = unlabeled_pseudo_label == u_cls
            cls_label_features = labeled_features_to_be_mixed[cls_mask]
            cls_unlabeled_features = unlabeled_feat_out_stage1[cls_u_mask]

            # mix features
            if not len(cls_label_features) or not len(cls_unlabeled_features):
                continue
            cls_encoders = self.class_encoders[cls]
            mixed_features = cls_encoders(cls_label_features, cls_unlabeled_features)
            cls_cycle_features = cls_encoders(mixed_features, cls_label_features)
            cycle_features = torch.cat((cycle_features, cls_cycle_features), dim=0)
            # mix other things
            mixed_feature_map = torch.cat((mixed_feature_map, mixed_features), dim=0)
            mixed_targets = torch.cat((mixed_targets, torch.tensor([cls]).repeat(
                cls_label_features.shape[0] * cls_unlabeled_features.shape[0]).to(self._device)), dim=0)
            mixed_mem_flag = torch.cat((mixed_mem_flag,
                                        2 * torch.ones(len(cls_unlabeled_features) * len(cls_label_features),
                                                       dtype=torch.long)), dim=0)
            cycle_targets = torch.cat(
                (cycle_targets, torch.tensor([cls]).repeat(cls_cycle_features.shape[0]).to(self._device)), dim=0)
        # pass through the extended feature map
        if not len(mixed_feature_map):
            return inputs, memory_flags, outputs, targets
        else:
            mixed_mem_outputs = training_network(mixed_feature_map, pre_pass=True)
        #
        new_output = {
            'cycle_feature_map': cycle_features,
            'cycle_targets': cycle_targets,
            'origin_raw_features': outputs['raw_features'],
            'origin_mem_flags': memory_flags,
            'origin_targets': targets,
            'unlabeled_imgs': unlabeled_img,
            'unlabeled_pseudo_label': unlabeled_pseudo_label,
            'origin_unlabeled_output': unlabeled_output,
            'raw_features': torch.cat((origin_raw_features, mixed_mem_outputs['raw_features']), dim=0),
            'features': torch.cat((origin_features, mixed_mem_outputs['features']), dim=0),
            'attention': outputs['attention'],
            'generator_outputs': mixed_feature_map,
            'origin_logits': origin_logits,
            'logits': torch.cat((origin_logits, mixed_mem_outputs['logits']), dim=0),
            'raw_logits': torch.cat((origin_outputs_raw_logits, mixed_mem_outputs['raw_logits']), dim=0),
            'stage2_feature_map': origin_stage2_feature_map,
        }
        #
        # new_inputs = torch.cat((labeled_output_attention[2], extend_mem_feature_map), dim=0)
        new_inputs = inputs
        new_targets = torch.cat((targets, mixed_targets), dim=0)
        mixed_mem_flag = torch.cat((memory_flags.long(), mixed_mem_flag), dim=0)
        # new_inputs = inputs
        return new_inputs, mixed_mem_flag, new_output, new_targets

    def _forward_loss(self, training_network, inputs, targets, memory_flags, unlabeled_data, memory_data=None,
                      fine_tune=False, gradcam_grad=None, gradcam_act=None, train_generator_config=None,
                      use_generator=True, **kwargs):

        inputs, targets = inputs.to(self._device), targets.to(self._device)
        if train_generator_config and memory_data is not None:
            mem_inputs, mem_targets = memory_data['inputs'].to(self._device), memory_data['targets'].to(self._device)
            mem_out = training_network(mem_inputs)
        else:
            mem_out, mem_targets = None, None
        outputs = training_network(inputs)
        if ((unlabeled_data is not None and (0 < memory_flags.sum())
             and not fine_tune) or train_generator_config is not None) \
                and use_generator:
            new_inputs, new_memory_flag, new_output, new_targets = self.mix_features_generator(inputs, memory_flags,
                                                                                               outputs, mem_out,
                                                                                               targets, mem_targets,
                                                                                               training_network,
                                                                                               unlabeled_data,
                                                                                               train_generator_config)
            pre_pass = True
        else:
            new_output = outputs
            new_targets = targets
            new_memory_flag = memory_flags
            new_inputs = inputs
            pre_pass = False

        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)

        if gradcam_act is not None:
            outputs["gradcam_gradients"] = gradcam_grad
            outputs["gradcam_activations"] = gradcam_act

        loss = self._compute_loss(new_inputs, new_output, new_targets, onehot_targets, new_memory_flag,
                                  pre_pass=pre_pass, fine_tune=fine_tune, train_generator_config=train_generator_config)

        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        return loss

    def _after_task_intensive(self, inc_dataset):
        if self._herding_selection["type"] == "confusion":
            self._compute_confusion_matrix()

        self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
            inc_dataset, self._herding_indexes
        )

    def _after_task(self, inc_dataset):
        self._old_model = self._network.copy().freeze().to(self._device)
        self._network.on_task_end()
        # self.plot_tsne()

    def _compute_confusion_matrix(self):
        use_validation = self._validation_percent > 0.
        _, loader = self.inc_dataset.get_custom_loader(
            list(range(self._n_classes - self._task_size, self._n_classes)),
            memory=self.get_val_memory() if use_validation else self.get_memory(),
            mode="test",
            data_source="val" if use_validation else "train"
        )
        ypreds, ytrue = self._eval_task(loader)
        self._last_results = (ypreds, ytrue)

    def plot_tsne(self):
        if self.folder_result:
            loader = self.inc_dataset.get_custom_loader([], memory=self.get_memory())[1]
            embeddings, targets = utils.extract_features(self._network, loader)
            utils.plot_tsne(
                os.path.join(self.folder_result, "tsne_{}".format(self._task)), embeddings, targets
            )

    def _eval_task(self, data_loader):
        ypreds, ytrue = self.compute_accuracy(self._network, data_loader, self._class_means)

        return ypreds, ytrue

    # -----------
    # Private API
    # -----------

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags, pre_pass=False, fine_tune=False,
                      mix_dist_config=None, train_generator_config=None):
        logits = outputs["logits"]

        if self._old_model is None:
            loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)
        else:
            with torch.no_grad():
                old_targets = torch.sigmoid(self._old_model(inputs)["logits"])

            new_targets = onehot_targets.clone()
            new_targets[..., :-self._task_size] = old_targets

            loss = F.binary_cross_entropy_with_logits(logits, new_targets)

        if self._rotations_config:
            rotations_loss = losses.unsupervised_rotations(
                inputs, memory_flags, self._network, self._rotations_config
            )
            loss += rotations_loss
            self._metrics["rot"] += rotations_loss.item()

        return loss

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    # -----------------
    # Memory management
    # -----------------

    def build_examplars(
            self, inc_dataset, herding_indexes, memory_per_class=None, data_source="train"
    ):
        logger.LOGGER.info("Building & updating memory.")
        memory_per_class = memory_per_class or self._memory_per_class
        herding_indexes = copy.deepcopy(herding_indexes)

        data_memory, targets_memory = [], []
        class_means = np.zeros((self._n_classes, self._network.features_dim))

        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(
                class_idx, mode="test", data_source=data_source
            )
            features, targets = utils.extract_features(self._network, loader)
            features_flipped, _ = utils.extract_features(
                self._network,
                inc_dataset.get_custom_loader(class_idx, mode="flip", data_source=data_source)[1]
            )

            if class_idx >= self._n_classes - self._task_size:
                # New class, selecting the examplars:
                if self._herding_selection["type"] == "icarl":
                    selected_indexes = herding.icarl_selection(features, memory_per_class)
                elif self._herding_selection["type"] == "closest":
                    selected_indexes = herding.closest_to_mean(features, memory_per_class)
                elif self._herding_selection["type"] == "random":
                    selected_indexes = herding.random(features, memory_per_class)
                elif self._herding_selection["type"] == "first":
                    selected_indexes = np.arange(memory_per_class)
                elif self._herding_selection["type"] == "kmeans":
                    selected_indexes = herding.kmeans(
                        features, memory_per_class, k=self._herding_selection["k"]
                    )
                elif self._herding_selection["type"] == "confusion":
                    selected_indexes = herding.confusion(
                        *self._last_results,
                        memory_per_class,
                        class_id=class_idx,
                        minimize_confusion=self._herding_selection["minimize_confusion"]
                    )
                elif self._herding_selection["type"] == "var_ratio":
                    selected_indexes = herding.var_ratio(
                        memory_per_class, self._network, loader, **self._herding_selection
                    )
                elif self._herding_selection["type"] == "mcbn":
                    selected_indexes = herding.mcbn(
                        memory_per_class, self._network, loader, **self._herding_selection
                    )
                else:
                    raise ValueError(
                        "Unknown herding selection {}.".format(self._herding_selection)
                    )

                herding_indexes.append(selected_indexes)

            # Reducing examplars:
            try:
                selected_indexes = herding_indexes[class_idx][:memory_per_class]
                herding_indexes[class_idx] = selected_indexes
            except:
                import pdb
                pdb.set_trace()

            # Re-computing the examplar mean (which may have changed due to the training):
            examplar_mean = self.compute_examplar_mean(
                features, features_flipped, selected_indexes, memory_per_class
            )

            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])

            class_means[class_idx, :] = examplar_mean

        data_memory = np.concatenate(data_memory)
        targets_memory = np.concatenate(targets_memory)

        return data_memory, targets_memory, herding_indexes, class_means

    def get_memory(self):
        return self._data_memory, self._targets_memory

    def get_pseudo_memory(self, aux_loader, threshold=1, classes_balance=True, n_classes_samples=100,
                          existing_cls=None):
        # data-mining from "Seed the Views: Hierarchical Semantic Alignment for Contrastive Representation Learning"
        # see: https://arxiv.org/abs/2012.02733

        # 1) "Using the fine-tuned model to infer on all unlabeled samples, and obtaining a confidence distribution for
        # each image."
        pbar = tqdm(aux_loader)
        all_pred = None
        global_scale = self._args.get("softmax_ce_scale", 1)
        # all_input = torch.tensor([])
        self._network.eval()
        last = 0
        with torch.no_grad():
            for idx, inputs in enumerate(pbar):
                image = inputs['images'].to(self._device)
                # labels = inputs['labels']
                output = self._network(image)
                logits = output['logits'] * global_scale
                # To get `distribution`, we apply softmax calculation here.
                distribution = F.softmax(logits, dim=1)
                if all_pred is None:
                    target_size = [len(aux_loader.dataset), distribution.shape[1]]
                    all_pred = torch.empty(*target_size).to(self._device)
                # all_pred = torch.cat((all_pred, distribution), dim=0)
                all_pred[last:last + distribution.shape[0], :] = distribution
                last += distribution.shape[0]
                # all_input = torch.cat((all_input, image.cpu()), dim=0)
        self._network.train()
        # 2) "The information entropy is calculated to measure the confidence degree of each image, and we filter those
        # samples with entropy higher than a threshold."
        all_pred = all_pred.cpu()
        log_pred = torch.log2(all_pred).unsqueeze(-1)
        information_entropy = -all_pred.unsqueeze(1).bmm(log_pred).squeeze()
        selected_index = information_entropy < threshold
        if not selected_index.sum():
            del all_pred
            return np.array([]), torch.tensor([], dtype=torch.long)
        target_pseudo_memory = all_pred[selected_index].max(1)[1]
        data_pseudo_memory = aux_loader.dataset.x_trn[selected_index]
        if classes_balance:
            valid_target = torch.tensor([], dtype=torch.long)
            valid_data = np.array([])
            candidates_cls = torch.unique(target_pseudo_memory)
            class_num = {}
            if existing_cls is not None:
                candidates_cls = list(set(candidates_cls.tolist()) - set(torch.unique(existing_cls).tolist()))
            for cls in candidates_cls:
                cls_mask = target_pseudo_memory == cls
                cls_target = target_pseudo_memory[cls_mask]
                cls_data = data_pseudo_memory[cls_mask]
                cls_n_classes_samples = min(cls_target.shape[0], n_classes_samples)
                cls_InfoEntropy = information_entropy[selected_index][cls_mask]
                _, sorted_idx = torch.sort(cls_InfoEntropy)
                valid_target = torch.cat((valid_target, cls_target[sorted_idx[:cls_n_classes_samples]]), dim=0)
                cls_valid_data = cls_data[sorted_idx[:cls_n_classes_samples]]

                if cls_n_classes_samples == 1:  # fixme: dummy dimensions
                    cls_valid_data = np.expand_dims(cls_valid_data, axis=0)

                if not len(valid_data):
                    valid_data = cls_valid_data
                else:
                    valid_data = np.concatenate((valid_data, cls_valid_data))
                class_num[cls] = len(cls_valid_data)
            logger.LOGGER.info(f'mined unlabeled data: {class_num}')
            del all_pred
            return valid_data, valid_target
        else:
            logger.LOGGER.info(f'mined unlabeled data: {len(data_pseudo_memory)}')
            del all_pred
            return data_pseudo_memory, target_pseudo_memory

    @staticmethod
    def compute_examplar_mean(feat_norm, feat_flip, indexes, nb_max):
        D = feat_norm.T
        D = D / (np.linalg.norm(D, axis=0) + EPSILON)

        D2 = feat_flip.T
        D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

        selected_d = D[..., indexes]
        selected_d2 = D2[..., indexes]

        mean = (np.mean(selected_d, axis=1) + np.mean(selected_d2, axis=1)) / 2
        mean /= (np.linalg.norm(mean) + EPSILON)

        return mean

    @staticmethod
    def compute_accuracy(model, loader, class_means):
        features, targets_ = utils.extract_features(model, loader)

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

        # Compute score for iCaRL
        sqd = cdist(class_means, features, 'sqeuclidean')
        score_icarl = (-sqd).T

        return score_icarl, targets_

    def create_generator_optimizer(self, params, lr):
        if self._generator_optimizer is None:
            # self._generator_optimizer.add_param_group({'params': params, 'lr': lr})
            self._generator_optimizer = factory.get_optimizer(params, self._opt_name, lr, self._weight_decay)
            self._generator_scheduler = factory.get_lr_scheduler(
                self._generator_scheduling,
                self._generator_optimizer,
                nb_epochs=self._n_epochs_generator,
                lr_decay=self._generator_lr_decay,
                task=self._task
            )
        else:
            self.add_params_to_generator_optimizer(params, lr)

    def add_params_to_generator_optimizer(self, params, lr):
        self._generator_optimizer.add_param_group({'params': params, 'lr': lr})

    @property
    def return_network_params_stage4(self):
        return self._network.convnet.stage_4.parameters()


def _clean_list(l):
    for i in range(len(l)):
        l[i] = None


def tensor_inv(a: torch.Tensor, maximum: int):
    all_possible_idx = torch.arange(0, maximum)
    res = torch.tensor(list(set(all_possible_idx.tolist()) - set(a.tolist())))
    return res
