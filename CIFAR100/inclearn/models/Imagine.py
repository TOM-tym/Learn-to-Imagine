import copy
import logging
import math

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import factory
from inclearn.lib import losses, network, utils, data
from inclearn.lib.data import samplers
from inclearn.models.icarl import ICarl

from inclearn.utils import LOGGER as logger
from inclearn.lib.losses.distillation import compute_collapsed_features_single


class Imagine(ICarl):
    """Learning to Imagine: Diversify Memory for Incremental Learning using Unlabeled Data, CVPR2022.

    # Reference:
        * https://github.com/arthurdouillard/incremental_learning.pytorch
          Douillard et al. 2020
    """

    def __init__(self, args):
        self.frozen_model = None
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        # Optimization:
        self._batch_size = args["batch_size"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        # Rehearsal Learning:
        self._memory_size = args["memory_size"]
        self._fixed_memory = args.get("fixed_memory", True)
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args.get("validation")

        self._feature_distill_config = args.get("distill_feat", {})

        self._softmax_ce = args.get("softmax_ce", False)

        self._groupwise_factors = args.get("groupwise_factors", {})
        self._groupwise_factors_bis = args.get("groupwise_factors_bis", {})

        self._class_weights_config = args.get("class_weights_config", {})

        self._evaluation_type = args.get("eval_type", "icarl")
        self._evaluation_config = args.get("evaluation_config", {})

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._consistency_config = args.get("consistency_config", {})
        classifier_kwargs = args.get("classifier_config", {})
        self._class_means = None
        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=classifier_kwargs,
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=args.get("classifier_no_act", True),
            attention_hook=True,
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")

        self._herding_indexes = []

        self._weight_generation = args.get("weight_generation")

        self._post_processing_type = None
        self._data_memory, self._targets_memory = None, None

        self._args = args
        self._args["_logs"] = {}

        self.pretrain_config = args.get("pretrain", {})
        self._pretrain_scheduling = self.pretrain_config["scheduling"]
        self._pretrain_batch_size = self.pretrain_config["batch_size"]
        self._pretrain_opt_name = self.pretrain_config["optimizer"]
        self._pretrain_lr = self.pretrain_config["lr"]
        self._pretrain_weight_decay = self.pretrain_config["weight_decay"]
        self._pretrain_n_epochs = self.pretrain_config["epochs"]
        self._pretrain_lr_decay = self.pretrain_config["lr_decay"]

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
        self._lr_layers = args.get("lr_layers", {})
        if self._generator_config:
            self.class_encoders = {}
            self.training_encoders = {}

        self.soft_cross_entropy = losses.SoftCrossEntropy(reduction='sum')
        self._diff_cls_mapping = {}

    def set_freeze_by_names(self, *args, **kwargs):
        self._network.convnet.set_freeze_by_names(*args, **kwargs)

    @property
    def device(self):
        return self._device

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    def _train_task(self, train_loader, pseudo_memory_loader, val_loader, train_generator_config=None,
                    train_generator_data=None, freeze_layers=False):
        for p in self._network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        logger.LOGGER.debug("nb {}.".format(len(train_loader.dataset)))

        if train_generator_config is not None:
            training_epochs = self._n_epochs_generator
        elif self._task == 0:
            training_epochs = self._pretrain_n_epochs
        else:
            training_epochs = self._n_epochs

        if train_generator_config is not None:
            self._network.freeze()
            for e in self.class_encoders:
                self.class_encoders[e].freeze(trainable=True)
                self.class_encoders[e].train()
        else:
            self._network.freeze(trainable=True)
            if self._generator_config:
                for e in self.class_encoders:
                    self.class_encoders[e].freeze()
                    self.class_encoders[e].eval()

        self._training_step(train_loader, pseudo_memory_loader, val_loader, 0, training_epochs, record_bn=True,
                            clipper=None, train_generator_config=train_generator_config,
                            train_generator_data=train_generator_data)

        self._post_processing_type = None

    def fine_tune(self, pseudo_memory_loader, pseudo_memory_valid_map_idx, val_loader):
        self._network.train()
        # self._network.freeze(trainable=True)
        if self._generator_config:
            for e in self.class_encoders:
                self.class_encoders[e].freeze()
                self.class_encoders[e].eval()
        logger.LOGGER.info("Fine-tuning")
        if self._finetuning_config["scaling"]:
            logger.LOGGER.info(
                "Custom fine-tuning scaling of {}.".format(self._finetuning_config["scaling"])
            )
            self._post_processing_type = self._finetuning_config["scaling"]
        if self._finetuning_config["sampling"] == "undersampling":
            self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                self.inc_dataset, self._herding_indexes
            )
            loader = self.inc_dataset.get_memory_loader(*self.get_memory())
        elif self._finetuning_config["sampling"] == "oversampling":
            _, loader = self.inc_dataset.get_custom_loader(
                list(range(self._n_classes - self._task_size, self._n_classes)),
                memory=self.get_memory(),
                mode="train",
                sampler=samplers.MemoryOverSampler
            )
        if self._finetuning_config["tuning"] == "all":
            parameters = self._network.parameters()
        elif self._finetuning_config["tuning"] == "convnet":
            parameters = self._network.convnet.parameters()
        elif self._finetuning_config["tuning"] == "classifier":
            parameters = self._network.classifier.parameters()
        elif self._finetuning_config["tuning"] == "classifier_scale":
            parameters = [
                {
                    "params": self._network.classifier.parameters(),
                    "lr": self._finetuning_config["lr"]
                }, {
                    "params": self._network.post_processor.parameters(),
                    "lr": self._finetuning_config["lr"]
                }
            ]
        else:
            raise NotImplementedError(
                "Unknwown finetuning parameters {}.".format(self._finetuning_config["tuning"])
            )
        self._optimizer = factory.get_optimizer(
            parameters, self._opt_name, self._finetuning_config["lr"], self.weight_decay
        )
        self._scheduler = None
        self._training_step(loader, pseudo_memory_loader, val_loader, self._n_epochs,
                            self._n_epochs + self._finetuning_config["epochs"], fine_tune=True, record_bn=False)

    @property
    def weight_decay(self):
        if isinstance(self._weight_decay, float):
            return self._weight_decay
        elif isinstance(self._weight_decay, dict):
            start, end = self._weight_decay["start"], self._weight_decay["end"]
            step = (max(start, end) - min(start, end)) / (self._n_tasks - 1)
            factor = -1 if start > end else 1

            return start + factor * self._task * step
        raise TypeError(
            "Invalid type {} for weight decay: {}.".format(
                type(self._weight_decay), self._weight_decay
            )
        )

    def _after_task(self, inc_dataset):
            super()._after_task(inc_dataset)

    def _eval_task(self, test_loader):
        if self._evaluation_type in ("icarl", "nme"):
            return super()._eval_task(test_loader)
        elif self._evaluation_type in ("softmax", "cnn"):
            ypred = []
            ytrue = []

            for input_dict in test_loader:
                ytrue.append(input_dict["targets"].numpy())

                inputs = input_dict["inputs"].to(self._device)
                logits = self._network(inputs)["logits"].detach()

                preds = F.softmax(logits, dim=-1)
                ypred.append(preds.cpu().numpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            self._last_results = (ypred, ytrue)

            return ypred, ytrue
        else:
            raise ValueError(self._evaluation_type)

    def _gen_weights(self):
        if self._weight_generation:
            utils.add_new_weights(
                self._network, self._weight_generation if self._task != 0 else "basic",
                self._n_classes, self._task_size, self.inc_dataset
            )

    def _before_task(self, train_loader, val_loader, taskid):
        self._gen_weights()
        self._n_classes += self._task_size
        logger.LOGGER.info("Now {} examplars per class.".format(self._memory_per_class))
        # classifier
        params = []
        if self._groupwise_factors and isinstance(self._groupwise_factors, dict):
            if self._groupwise_factors_bis and self._task > 0:
                logger.LOGGER.info("Using second set of groupwise lr.")
                groupwise_factor = self._groupwise_factors_bis
            else:
                groupwise_factor = self._groupwise_factors

            for group_name, group_params in self._network.get_group_parameters().items():
                if group_params is None or group_name == "last_block" or 'convnet' in group_name:
                    continue
                factor = groupwise_factor.get(group_name, 1.0)
                if factor == 0.:
                    continue
                # valid_param = filter(lambda p: p.requires_grad, group_params)
                params.append({"params": group_params, "lr": self._lr * factor})
                print(f"Group: {group_name}, lr: {self._lr * factor}.")
        # backbone lr
        if self._lr_layers and taskid != 0:
            lr_layers = self._lr_layers.get("layers_lr", {'convnet.conv1': 0.01,
                                                          'convnet.bn1': 0.01,
                                                          'convnet.stage1': 0.01})
            desired_params_set = set()
            for layer_name, lr in lr_layers.items():
                layer = getattr(self._network.convnet, layer_name.split('convnet.')[-1])
                current_params = layer.parameters()
                params.append({'params': current_params, 'lr': lr})
                desired_params_set.update(set(layer.parameters()))
            all_params_set = set(self._network.convnet.parameters())
            other_params_set = all_params_set.difference(desired_params_set)
            other_params = list(other_params_set)
            params.append({'params': other_params, 'lr': self._lr})
        else:
            params.append({'params': self._network.convnet.parameters()})
        if taskid == 0:  # pretrain
            self._optimizer = factory.get_optimizer(params, self._pretrain_opt_name, self._pretrain_lr,
                                                    self._pretrain_weight_decay)
            self._scheduler = factory.get_lr_scheduler(
                self._pretrain_scheduling,
                self._optimizer,
                nb_epochs=self._pretrain_n_epochs,
                lr_decay=self._pretrain_lr_decay,
                task=self._task
            )
        else:
            self._optimizer = factory.get_optimizer(params, self._opt_name, self._lr, self.weight_decay)
            self._scheduler = factory.get_lr_scheduler(
                self._scheduling,
                self._optimizer,
                nb_epochs=self._n_epochs,
                lr_decay=self._lr_decay,
                task=self._task
            )
        if self._class_weights_config:
            self._class_weights = torch.tensor(
                data.get_class_weights(train_loader.dataset, **self._class_weights_config)
            ).to(self._device)
        else:
            self._class_weights = None

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags, pre_pass=False, fine_tune=False,
                      mix_dist_config=None, train_generator_config=None):
        loss = 0
        if train_generator_config is not None:
            mixed_mask = memory_flags == 2
            new_mask = memory_flags == 0
            mixed_targets = targets[mixed_mask]
            mixed_logits = outputs['logits'][mixed_mask]
            loss_ce = F.cross_entropy(mixed_logits, mixed_targets)

            labeled_feature_map = torch.stack([i for i in outputs['attention'][2]])
            labeled_targets = targets[new_mask]
            mixed_feature_map = outputs['generator_outputs']
            unlabeled_feature_map = outputs['origin_unlabeled_output']['attention'][2]
            unlabeled_targets = outputs['unlabeled_pseudo_label']
            loss_contra = self._compute_train_generator_loss(mixed_feature_map=mixed_feature_map,
                                                             mixed_target=mixed_targets,
                                                             unlabeled_feature_map=unlabeled_feature_map,
                                                             unlabeled_target=unlabeled_targets,
                                                             labeled_feature_map=labeled_feature_map,
                                                             labeled_target=labeled_targets,
                                                             cycle_feature_map=outputs['cycle_feature_map'],
                                                             cycle_target=outputs['cycle_targets'],
                                                             **train_generator_config)
            self._metrics['cce'] += loss_ce.item()
            self._metrics['contrastive'] += loss_contra.item()
            use_ce = train_generator_config.get('use_ce', False)
            if use_ce:
                loss = loss_ce + loss_contra
            else:
                loss = loss_contra

            return loss
        features, logits, atts = outputs["raw_features"], outputs["logits"], outputs["attention"]
        if 'origin_raw_features' in outputs.keys():
            ori_features = outputs['origin_raw_features']
        else:
            ori_features = features
        if 'origin_mem_flags' in outputs.keys():
            ori_mem_flags = outputs['origin_mem_flags']
        else:
            ori_mem_flags = memory_flags

        if self._post_processing_type is None:
            scaled_logits = self._network.post_process(logits)
        else:
            scaled_logits = logits * self._post_processing_type
        ori_unlabeled_old_feat = None
        dist_unlabeled = self._feature_distill_config.get('dist_unlabeled', False)

        if self._old_model is not None:
            with torch.no_grad():
                old_outputs = self._old_model(inputs, pre_pass=False)
                old_features = old_outputs["raw_features"]
                if 'origin_raw_features' in old_outputs.keys():
                    old_ori_features = old_outputs['origin_raw_features']
                else:
                    old_ori_features = old_features
                if dist_unlabeled and 'unlabeled_imgs' in outputs:
                    ori_unlabeled_old_output = self._old_model(outputs['unlabeled_imgs'])
                    ori_unlabeled_old_feat = ori_unlabeled_old_output['raw_features']

        if self._softmax_ce:
            only_new = self._args.get("softmax_ce_only_new", False)
            not_unlabeled = self._args.get("softmax_ce_not_unlabeled", False)
            split = self._args.get("softmax_ce_split", False)
            global_scale = self._args.get("softmax_ce_scale", 1)
            ce_weight = self._args.get("softmax_weight", 1)
            use_unlabeled_ce = self._args.get("use_unlabeled_ce", False)

            if only_new and not fine_tune:
                new_mask = memory_flags == 0
                scaled_logits = scaled_logits[new_mask]
                targets = targets[new_mask]
            elif not fine_tune and not_unlabeled:
                u_mask = memory_flags == 2
                scaled_logits = scaled_logits[~u_mask]
                targets = targets[~u_mask]
            if split:
                new_mask = memory_flags == 0
                new_logits = scaled_logits[new_mask]
                new_targets = targets[new_mask]
                old_logits = scaled_logits[~new_mask]
                old_targets = targets[~new_mask]
                new_ce, old_ce = 0, 0
                if len(new_logits):
                    new_ce += F.cross_entropy(new_logits * global_scale, new_targets, reduction='sum')
                    self._metrics["cce_new"] += new_ce.item()
                if len(old_logits):
                    old_ce += F.cross_entropy(old_logits * global_scale, old_targets, reduction='sum')
                    self._metrics["cce_old"] += old_ce.item()
                loss_ce = (new_ce + old_ce) / len(targets) * ce_weight

            else:
                loss_ce = F.cross_entropy(scaled_logits * global_scale, targets) * ce_weight

            self._metrics["cce"] += loss_ce.item()
            if use_unlabeled_ce and 'unlabeled_imgs' in outputs:
                unlabeled_new_outputs = self._network(outputs['unlabeled_imgs'])
                u_targets = outputs['unlabeled_pseudo_label'].to(self._device)
                u_logits = unlabeled_new_outputs['logits']
                if self._post_processing_type is None:
                    scaled_u_logits = self._network.post_process(u_logits)
                else:
                    scaled_u_logits = logits * self._post_processing_type
                u_loss = F.cross_entropy(scaled_u_logits, u_targets)
                self._metrics["u_loss"] += u_loss.item()
                loss += (loss_ce * len(targets) + u_loss * len(u_targets)) / (len(targets) + len(u_targets))
            else:
                loss += loss_ce

        # --------------------
        # Distillation losses:
        # --------------------

        if self._old_model is not None:
            if self._feature_distill_config:
                if self._feature_distill_config.get("scheduled_factor", False):
                    factor = self._feature_distill_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._feature_distill_config.get("factor", 1.)
                only_old = self._feature_distill_config.get("only_old", False)
                if only_old:
                    mem_mask = ori_mem_flags == 1
                    old_ori_features = old_ori_features[mem_mask]
                    ori_features = ori_features[mem_mask]
                    if mem_mask.sum() == 0:
                        feat_distill_loss = torch.tensor(0).to(self._device)
                    else:
                        feat_distill_loss = factor * losses.embeddings_similarity(old_ori_features, ori_features)
                else:
                    feat_distill_loss = factor * losses.embeddings_similarity(old_ori_features, ori_features)

                if dist_unlabeled:
                    if ori_unlabeled_old_feat is None:
                        pass
                    else:
                        ori_unlabeled_output = outputs['origin_unlabeled_output']['raw_features']
                        feat_distill_loss += factor * losses.embeddings_similarity(ori_unlabeled_old_feat,
                                                                               ori_unlabeled_output)

                loss += feat_distill_loss
                self._metrics["feat_distill"] += feat_distill_loss.item()

        return loss

    def _compute_train_generator_loss(self, mixed_feature_map, mixed_target,
                                      unlabeled_feature_map, unlabeled_target,
                                      labeled_feature_map, labeled_target,
                                      cycle_feature_map, cycle_target,
                                      use_contrastive=True, collapsed_type='spatial', use_channels=True, normalize=True,
                                      use_style=True, style_mean=False, style_weight=1, use_unlabeled_feature_map=True,
                                      only_softmax=False, diff_cls=None, use_cycle=True, cycle_use_channel=True,
                                      cycle_use_style=True, cycle_weight_global=0.5,
                                      **kwargs):
        mixed_cls = torch.unique(mixed_target)
        loss = torch.zeros(1).to(self._device)
        smooth_type = 'normalize' if not only_softmax else 'softmax'
        for cls in mixed_cls:
            labeled_mask = labeled_target == cls
            mixed_mask = mixed_target == cls
            cycle_mask = cycle_target == cls
            cls_mixed_feat_map = mixed_feature_map[mixed_mask]
            cls_labeled_feat_map = labeled_feature_map[labeled_mask]
            cls_cycle_feat_map = cycle_feature_map[cycle_mask]
            num = int(len(cls_cycle_feat_map) / len(cls_labeled_feat_map))

            collapsed_mixed_feat = compute_collapsed_features_single(cls_mixed_feat_map, 'spatial', normalize)
            collapsed_cycle_feat = compute_collapsed_features_single(cls_cycle_feat_map, 'spatial', normalize)
            mixed_feat_gram = compute_gram_matrix(cls_mixed_feat_map, smooth_type)
            collapsed_mixed_feat_c = compute_collapsed_features_single(cls_mixed_feat_map, 'channels', normalize)
            collapsed_cycle_feat_c = compute_collapsed_features_single(cls_cycle_feat_map, 'channels', normalize)

            collapsed_labeled_feat = compute_collapsed_features_single(cls_labeled_feat_map, 'spatial', normalize)
            collapsed_labeled_feat_c = compute_collapsed_features_single(cls_labeled_feat_map, 'channels', normalize)
            if use_unlabeled_feature_map:
                if not self._generator_use_same_class:
                    u_cls = self._diff_cls_mapping[cls.item()]
                else:
                    u_cls = cls.item()
                style_ref = unlabeled_feature_map[unlabeled_target == u_cls]
            else:
                style_ref = cls_labeled_feat_map
            mixed_labeled_gram = compute_gram_matrix(style_ref, smooth_type)

            selected_labeled_feat = collapsed_labeled_feat[
                torch.from_numpy(
                    np.random.choice(
                        np.arange(0, len(collapsed_labeled_feat)),
                        size=len(collapsed_mixed_feat), replace=True)
                )
            ]

            # target_mean = selected_labeled_feat.mean(dim=0)
            selected_labeled_feat_c = collapsed_labeled_feat_c[
                torch.from_numpy(
                    np.random.choice(
                        np.arange(0, len(collapsed_labeled_feat_c)),
                        size=len(collapsed_mixed_feat), replace=True)
                )
            ]

            gram_mean = mixed_labeled_gram.mean(dim=0)
            if not use_unlabeled_feature_map:
                selected_labeled_gram = mixed_labeled_gram[
                    torch.from_numpy(
                        np.random.choice(
                            np.arange(0, len(mixed_labeled_gram)),
                            size=len(mixed_feat_gram), replace=True)
                    )
                ]
            else:
                num = int(len(mixed_feat_gram) / len(mixed_labeled_gram))
                selected_labeled_gram = mixed_labeled_gram.expand(num, -1, -1, -1).reshape(*mixed_feat_gram.shape)
            style_target = gram_mean if style_mean else selected_labeled_gram
            positive = torch.frobenius_norm(collapsed_mixed_feat - selected_labeled_feat, dim=-1).mean() / len(
                mixed_cls)
            positive_c = torch.frobenius_norm(collapsed_mixed_feat_c - selected_labeled_feat_c, dim=-1).mean() / len(
                mixed_cls)

            gram_loss = torch.frobenius_norm(mixed_feat_gram - style_target, dim=-1).mean() / len(mixed_cls)

            cls_labeled_expanded = cls_labeled_feat_map.expand(num, -1, -1, -1, -1).reshape(
                *cls_cycle_feat_map.shape)
            cls_mixed_expanded = cls_mixed_feat_map.expand(num, -1, -1, -1, -1).transpose(0, 1).reshape(
                *cls_cycle_feat_map.shape)

            collapsed_labeled_feat_cycle = compute_collapsed_features_single(cls_mixed_expanded, 'spatial', normalize)
            collapsed_labeled_feat_c_cycle = compute_collapsed_features_single(cls_mixed_expanded, 'channels',
                                                                               normalize)
            style_cycle_gram = compute_gram_matrix(cls_cycle_feat_map, smooth_type)
            style_cycle_target = compute_gram_matrix(cls_labeled_expanded, smooth_type)
            selected_labeled_feat_cycle = collapsed_labeled_feat_cycle[
                torch.from_numpy(
                    np.random.choice(
                        np.arange(0, len(collapsed_labeled_feat_cycle)),
                        size=len(collapsed_cycle_feat), replace=True)
                )
            ]
            selected_labeled_feat_cycle_c = collapsed_labeled_feat_c_cycle[
                torch.from_numpy(
                    np.random.choice(
                        np.arange(0, len(collapsed_labeled_feat_c_cycle)),
                        size=len(collapsed_cycle_feat), replace=True)
                )
            ]

            cycle_sp_loss = torch.frobenius_norm(collapsed_cycle_feat - selected_labeled_feat_cycle,
                                                 dim=-1).mean() / len(mixed_cls)
            cycle_channel_loss = torch.frobenius_norm(collapsed_cycle_feat_c - selected_labeled_feat_cycle_c,
                                                      dim=-1).mean() / len(mixed_cls)
            if use_contrastive:
                loss += positive
                self._metrics['positive'] += positive.item()
                if use_channels:
                    positive_c = positive_c
                    loss += positive_c
                    self._metrics['positive_c'] += positive_c.item()
            if use_style:
                loss += gram_loss * style_weight
                self._metrics['gram_loss'] += gram_loss.item()

            if use_cycle:
                loss += cycle_sp_loss * cycle_weight_global
                self._metrics['cycle_sp_loss'] += cycle_sp_loss.item()
                if cycle_use_channel:
                    loss += cycle_channel_loss * cycle_weight_global
                    self._metrics['cycle_channel_loss'] += cycle_channel_loss.item()
                if cycle_use_style:
                    gram_loss_style = torch.frobenius_norm(style_cycle_gram - style_cycle_target, dim=-1).mean() / len(
                        mixed_cls)
                    loss += gram_loss_style * style_weight * cycle_weight_global
                    self._metrics['cycle_gram'] += gram_loss_style.item()

        return loss

    def _after_task_intensive(self, inc_dataset, train_generator=False):
        if self._task == self._n_tasks - 1:
            logger.LOGGER.info('M: Generating memory for eval.')
            super()._after_task_intensive(inc_dataset=inc_dataset)
        elif self._train_generator_config:
            if train_generator:
                logger.LOGGER.info('M: Generating memory for eval.')
                super()._after_task_intensive(inc_dataset=inc_dataset)
            else:
                logger.LOGGER.info('Memory exists, skip generating memory.')
        else:
            logger.LOGGER.info('Generating memory for eval.')
            super()._after_task_intensive(inc_dataset=inc_dataset)


def compute_gram_matrix(feature_maps, smooth_type='normalize'):
    # Unwrapping the tensor dimensions into respective variables i.e. batch size, distance, height and width
    n, c, h, w = feature_maps.size()
    # Reshaping data into a two dimensional of array or two dimensional of tensor
    if smooth_type == 'normalize':
        tensor = F.normalize(feature_maps.view(n, c, h * w), dim=-1)
    elif smooth_type == 'softmax':
        tensor = F.softmax(feature_maps.view(n, c, h * w), dim=-1)
    else:
        assert False
    # Multiplying the original tensor with its own transpose using torch.mm
    # tensor.t() will return the transpose of original tensor
    transposed_tensor = tensor.transpose(-1, -2)
    gram = torch.bmm(tensor, transposed_tensor)
    # Returning gram matrix
    return gram
