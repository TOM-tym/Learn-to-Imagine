import copy
import json
import logging
import os
import pickle
import random
import statistics
import sys
import time

import numpy as np
import torch

import yaml
from inclearn.lib import factory
# from inclearn.lib import logger as logger_lib
from inclearn.lib import metrics, results_utils, utils
from inclearn.lib.network import FeatureGenerator
from inclearn.lib.data.samplers import NPairSampler, AuxSampler
from copy import deepcopy

from inclearn.utils import LOGGER as logger
import pprint


def train(args):
    # logger_lib.set_logging_level(args["logging"])
    logger.LOGGER.setLevel(args["logging"].upper())
    autolabel = _set_up_options(args)
    if args["autolabel"]:
        args["label"] = autolabel

    if args["label"]:
        logger.LOGGER.info("Label: {}".format(args["label"]))
        try:
            os.system("echo '\ek{}\e\\'".format(args["label"]))
        except:
            pass
    if args["resume"] and not os.path.exists(args["resume"]):
        raise IOError(f"Saved model {args['resume']} doesn't exist.")

    if args["save_model"] != "never" and args["label"] is None:
        raise ValueError(f"Saving model every {args['save_model']} but no label was specified.")

    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    start_date = utils.get_date()
    results_folder = results_utils.get_save_folder(args["model"], start_date, args["label"])
    logger.add_file_headler(results_folder)

    orders = copy.deepcopy(args["order"])
    del args["order"]
    if orders is not None:
        assert isinstance(orders, list) and len(orders)
        assert all(isinstance(o, list) for o in orders)
        assert all([isinstance(c, int) for o in orders for c in o])
    else:
        orders = [None for _ in range(len(seed_list))]

    avg_inc_accs, last_accs, forgettings = [], [], []
    for i, seed in enumerate(seed_list):
        logger.LOGGER.warning("Launching run {}/{}".format(i + 1, len(seed_list)))
        args["seed"] = seed
        args["device"] = device

        start_time = time.time()

        for avg_inc_acc, last_acc, forgetting in _train(args, start_date, orders[i], i):
            yield avg_inc_acc, last_acc, forgetting, False

        avg_inc_accs.append(avg_inc_acc)
        last_accs.append(last_acc)
        forgettings.append(forgetting)

        logger.LOGGER.info("Training finished in {}s.".format(int(time.time() - start_time)))
        yield avg_inc_acc, last_acc, forgetting, True

    logger.LOGGER.info("Label was: {}".format(args["label"]))

    logger.LOGGER.info(
        "Results done on {} seeds: avg: {}, last: {}, forgetting: {}".format(
            len(seed_list), _aggregate_results(avg_inc_accs), _aggregate_results(last_accs),
            _aggregate_results(forgettings)
        )
    )
    logger.LOGGER.info("Individual results avg: {}".format([round(100 * acc, 2) for acc in avg_inc_accs]))
    logger.LOGGER.info("Individual results last: {}".format([round(100 * acc, 2) for acc in last_accs]))
    logger.LOGGER.info(
        "Individual results forget: {}".format([round(100 * acc, 2) for acc in forgettings])
    )

    logger.LOGGER.info(f"Command was {' '.join(sys.argv)}")


def _train(args, start_date, class_order, run_id):
    _set_global_parameters(args)
    inc_dataset, model = _set_data_model(args, class_order)
    results, results_folder = _set_results(args, start_date)

    memory, memory_val, pseudo_memory = None, None, None
    metric_logger = metrics.MetricLogger(
        inc_dataset.n_tasks, inc_dataset.n_classes, inc_dataset.increments
    )
    use_unlabeled = args.get("use_unlabeled", False)
    pseudo_same_class = model._generator_config.get("use_same_classes", False)
    print(f'use_unlabeled:{use_unlabeled}')
    for task_id in range(inc_dataset.n_tasks):
        pseudo_memory_n_samples = args.get("pseudo_memory_n_samples", 2)
        task_info, train_loader, val_loader, test_loader, aux_loader, pseudo_memory_loader, pure_new_data = \
            inc_dataset.new_task(memory, pseudo_memory, memory_val, pseudo_memory_n_samples=pseudo_memory_n_samples,
                                 pseudo_same_class=pseudo_same_class)
        if task_info["task"] == args["max_task"]:
            break
        model.set_task_info(task_info)

        # ---------------
        # 1. Prepare Task
        # ---------------
        model.eval()
        model.before_task(train_loader, val_loader if val_loader else test_loader)

        # -------------
        # 2. Train Task
        # -------------
        pseudo_memory, resume_from_chkpt = _train_task(args, model, train_loader, aux_loader, pseudo_memory,
                                                       pseudo_memory_loader, pure_new_data, inc_dataset, val_loader,
                                                       test_loader, run_id, task_id, task_info, results_folder)

        # ----------------
        # 3. Conclude Task
        # ----------------
        model.eval()
        _after_task(args, model, inc_dataset, run_id, task_id, results_folder)

        # ------------
        # 4. Eval Task
        # ------------
        logger.LOGGER.info("Eval on {}->{}.".format(0, task_info["max_class"]))
        if resume_from_chkpt:
            logger.LOGGER.info(f'Skipping eval on task {task_id}, because of loading from checkpoints.')
            path = os.path.join(args['resume'], "predictions_{}".format(run_id),
                                str(task_id).rjust(len(str(30)), "0") + ".pkl")
            try:
                with open(path, 'rb') as f:
                    ypreds, ytrue = pickle.load(f)
            except Exception as e:
                logger.LOGGER.warning(f'Error when loading predictions, {e}, re-calculate it.')
                ypreds, ytrue = model.eval_task(test_loader)
        else:
            ypreds, ytrue = model.eval_task(test_loader)
            if args["dump_predictions"] and args["label"]:
                os.makedirs(
                    os.path.join(results_folder, "predictions_{}".format(run_id)), exist_ok=True
                )
                with open(
                        os.path.join(
                            results_folder, "predictions_{}".format(run_id),
                            str(task_id).rjust(len(str(30)), "0") + ".pkl"
                        ), "wb+"
                ) as f:
                    pickle.dump((ypreds, ytrue), f)
        metric_logger.log_task(
            ypreds, ytrue, task_size=task_info["increment"], zeroshot=args.get("all_test_classes")
        )

        if args["dump_predictions"] and args["label"]:
            os.makedirs(
                os.path.join(results_folder, "predictions_{}".format(run_id)), exist_ok=True
            )
            with open(
                    os.path.join(
                        results_folder, "predictions_{}".format(run_id),
                        str(task_id).rjust(len(str(30)), "0") + ".pkl"
                    ), "wb+"
            ) as f:
                pickle.dump((ypreds, ytrue), f)

        if args["label"]:
            logger.LOGGER.info(args["label"])
        logger.LOGGER.info("Avg inc acc: {}.".format(metric_logger.last_results["incremental_accuracy"]))
        logger.LOGGER.info("Current acc: {}.".format(metric_logger.last_results["accuracy"]))
        logger.LOGGER.info(
            "Avg inc acc top5: {}.".format(metric_logger.last_results["incremental_accuracy_top5"])
        )
        logger.LOGGER.info("Current acc top5: {}.".format(metric_logger.last_results["accuracy_top5"]))
        logger.LOGGER.info("Forgetting: {}.".format(metric_logger.last_results["forgetting"]))
        logger.LOGGER.info("Cord metric: {:.2f}.".format(metric_logger.last_results["cord"]))
        if task_id > 0:
            logger.LOGGER.info(
                "Old accuracy: {:.2f}, mean: {:.2f}.".format(
                    metric_logger.last_results["old_accuracy"],
                    metric_logger.last_results["avg_old_accuracy"]
                )
            )
            logger.LOGGER.info(
                "New accuracy: {:.2f}, mean: {:.2f}.".format(
                    metric_logger.last_results["new_accuracy"],
                    metric_logger.last_results["avg_new_accuracy"]
                )
            )
        if args.get("all_test_classes"):
            logger.LOGGER.info(
                "Seen classes: {:.2f}.".format(metric_logger.last_results["seen_classes_accuracy"])
            )
            logger.LOGGER.info(
                "unSeen classes: {:.2f}.".format(
                    metric_logger.last_results["unseen_classes_accuracy"]
                )
            )

        results["results"].append(metric_logger.last_results)

        avg_inc_acc = results["results"][-1]["incremental_accuracy"]
        last_acc = results["results"][-1]["accuracy"]["total"]
        forgetting = results["results"][-1]["forgetting"]
        yield avg_inc_acc, last_acc, forgetting

        memory = model.get_memory()
        memory_val = model.get_val_memory()
    logger.LOGGER.info(
        "Average Incremental Accuracy: {}.".format(results["results"][-1]["incremental_accuracy"])
    )
    if args["label"] is not None:
        results_utils.save_results(
            results, args["label"], args["model"], start_date, run_id, args["seed"]
        )

    del model
    del inc_dataset


def get_pseudo_memory(aux_loader, model, pseudo_memory, load_folder, save_folder, run_id, task_id, re_mine=False,
                      n_classes_samples=100):
    unlabeled_data_save_path = os.path.join(save_folder, f'pseudo_memory_{task_id}_task_{run_id}.pth')
    unlabeled_data_load_path = None
    if load_folder is not None:
        unlabeled_data_load_path = os.path.join(load_folder, f'pseudo_memory_{task_id}_task_{run_id}.pth')
    if unlabeled_data_load_path is not None and os.path.exists(unlabeled_data_load_path):
        pseudo_memory = torch.load(unlabeled_data_load_path)
        logger.LOGGER.info(f'Loaded existing pseudo data form {unlabeled_data_load_path}.')
        new_data, new_label = pseudo_memory[0], pseudo_memory[1]
    else:
        if pseudo_memory is not None and not re_mine:
            existing_pseudo_mem_cls = torch.unique(pseudo_memory[1])
        else:
            existing_pseudo_mem_cls = None

        new_pseudo_memory = model.get_pseudo_memory(aux_loader, existing_cls=existing_pseudo_mem_cls,
                                                    n_classes_samples=n_classes_samples)

        if existing_pseudo_mem_cls is not None and not re_mine:
            new_data = np.concatenate((pseudo_memory[0], new_pseudo_memory[0]), axis=0)
            new_label = torch.cat((pseudo_memory[1], new_pseudo_memory[1]), dim=0).cpu()
            pseudo_memory = (new_data, new_label)
        else:
            pseudo_memory = new_pseudo_memory
            new_data = new_pseudo_memory[0]
            new_label = new_pseudo_memory[1]

        logger.LOGGER.info(f'Now unlabeled data: {len(pseudo_memory[0])}')
    if not os.path.exists(unlabeled_data_save_path):
        torch.save(pseudo_memory, unlabeled_data_save_path)
        logger.LOGGER.info(f'Saved pseudo memory to {unlabeled_data_save_path}.')
    return pseudo_memory, new_data, new_label


# ------------------------
# Lifelong Learning phases
# ------------------------


def _train_task(config, model, train_loader, aux_loader, pseudo_memory, pseudo_memory_loader, pure_new_data,
                inc_dataset, val_loader, test_loader, run_id, task_id, task_info, results_folder):
    retrain = False
    # pre-check the existence of checkpoints for current task_id
    checkpoints_existence = False
    resume_from_chkpt = True
    if config["resume"] is not None:
        path = os.path.join(config["resume"], f"net_{run_id}_task_{task_id}.pth")
        if os.path.exists(path):
            checkpoints_existence = True

    if config["resume"] is not None and checkpoints_existence and not (config["resume_first"] and task_id > 0):
        res = model.load_parameters(config["resume"], run_id, device=config['device'][0])
        model.load_metadata(config['resume'], run_id)
        logger.LOGGER.info(
            "Skipping training phase {} because reloading pretrained model.".format(task_id)
        )
        retrain = not res
    elif config["resume"] is not None and os.path.isfile(config["resume"]) and \
            os.path.exists(config["resume"]) and task_id == 0:
        # In case we resume from a single model file, it's assumed to be from the first task.
        # model.network = config["resume"]
        res = model.load_parameters(config["resume"], run_id, device=config['device'][0])
        model.load_metadata(config['resume'], run_id)
        logger.LOGGER.info(
            "Skipping initial training phase {} because reloading pretrained model.".
                format(task_id)
        )
    else:
        logger.LOGGER.info("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
        model.train()
        model.train_task(train_loader, pseudo_memory_loader, val_loader if val_loader else test_loader,
                         freeze_layers=task_id != 0)

    if config["label"] and (
            config["save_model"] == "task" or
            (config["save_model"] == "last" and task_id == inc_dataset.n_tasks - 1) or
            (config["save_model"] == "first" and task_id == 0)
    ):
        model.save_parameters(results_folder, run_id)
        # model.save_metadata(results_folder, run_id)
    # model.network.convnet.init_fake_BN()
    finetuning_config = config.get("finetuning_config")

    use_unlabeled = config.get('use_unlabeled', False)
    generator_config = config.get("generator_config", {})
    pseudo_same_class = generator_config.get("use_same_classes", False)
    train_generator_config = generator_config.get("train_config", {})
    batch_size = config.get("labeled_batch_size", 128)
    re_mined = config.get("pseudo_re_mined", False)
    n_classes_samples = config.get("pseudo_mem_n_classes_samples", 100)
    if task_id < task_info["max_task"] - 1 and use_unlabeled:
        p = get_pseudo_memory(aux_loader, model, pseudo_memory, config["resume"], results_folder, run_id, task_id,
                              re_mine=re_mined, n_classes_samples=n_classes_samples)
        pseudo_memory = p[0]
        new_pseudo_memory = (p[1], p[2])

        pseudo_memory_n_samples = train_generator_config.get('train_generator_unlabel_n_samples', 2)
        current_pseudo_class = torch.unique(new_pseudo_memory[1])

        # train the feature generator
        if generator_config:
            n_class_mem = train_generator_config.get("train_generator_memory_n_samples", 12)
            n_class_new = train_generator_config.get("train_generator_new_n_samples", 12)
            if not config["resume"] or (task_id > 0 and config["resume_first"]) or (
                    config['resume'] and model.get_memory()[0] is None):
                model.after_task_intensive(inc_dataset, train_generator=True)
            current_memory = model.get_memory()

            input_dim = generator_config.get("input_dim", 64)
            latent_dim = generator_config.get("latent_dim", 64)
            num_blocks = generator_config.get("n_blocks", 2)

            for cls in range(int(task_info['min_class']), int(task_info['max_class'])):
                cls_encoder = FeatureGenerator(input_dim, latent_dim=latent_dim, num_blocks=num_blocks).to(model.device)
                lr = generator_config.get("lr", 0.1)
                model.create_generator_optimizer(cls, cls_encoder.parameters(), lr=lr, task_id=task_id)
                model.class_encoders[cls] = cls_encoder
            res = False
            if task_id == 0 and config["resume"] is not None:
                generator_path = os.path.join(config["resume"], 'generators')
                if os.path.exists(generator_path):
                    res = load_generator_params(model, config, generator_path, run_id, task_info['min_class'],
                                                task_info['max_class'])
            use_generators = not config.get('softmax_ce_not_unlabeled', False)
            if config['resume']:
                next_stage_pth_exists = os.path.exists(
                    os.path.join(config['resume'], f"net_{run_id}_task_{model._task + 1}.pth"))
            else:
                next_stage_pth_exists = None
            if not next_stage_pth_exists and (not res and task_id < task_info["max_task"] - 1 and use_generators):
                nb_class = min(len(np.unique(pure_new_data[1])), int(batch_size / n_class_new))
                train_sampler = NPairSampler(y=pure_new_data[1], n_classes=nb_class, n_samples=n_class_new)
                train_loader_PK = inc_dataset.get_loader(*pure_new_data, memory_flags=np.zeros(len(pure_new_data[1])),
                                                         mode="train", sampler=train_sampler, sampler_init=False)

                mem_sampler = AuxSampler(current_memory[1], batch_size=n_class_mem * int(task_info['increment']),
                                         n_sample=n_class_mem, farther_dataset_batch_size=train_loader.batch_size,
                                         farther_dataset_idx=None,
                                         farther_dataset_targets=None,
                                         farther_dataset_mem_flag=None,
                                         same_class=True,
                                         farther_is_mem=False,
                                         father_classes_per_batch=train_sampler.classes_per_batch)

                memory_loader_PK = inc_dataset.get_loader(*current_memory,
                                                          memory_flags=np.zeros(current_memory[0].shape),
                                                          mode="train", sampler=mem_sampler, sampler_init=False)
                tmp_pseudo_memory_loader = inc_dataset.get_pseudo_memory_loader(pseudo_memory, pseudo_memory_n_samples,
                                                                                batch_size=pseudo_memory_n_samples * len(
                                                                                    current_pseudo_class),
                                                                                farther_dataset_batch_size=train_loader.batch_size,
                                                                                farther_dataset_idx=None,
                                                                                farther_dataset_targets=None,
                                                                                farther_dataset_mem_flag=None,
                                                                                same_class=pseudo_same_class,
                                                                                farther_is_mem=False,
                                                                                father_classes_per_batch=train_sampler.classes_per_batch)

                train_generator_data = {
                    'labeled_loader': train_loader_PK,
                    'memory_loader': memory_loader_PK,
                }
                model.train_task(train_loader, tmp_pseudo_memory_loader, val_loader if val_loader else test_loader,
                                 train_generator_data=train_generator_data, train_generator_config=train_generator_config)

    # fine-tune
    if finetuning_config and task_id > 0:
        model.fine_tune(pseudo_memory_loader, val_loader)
    return pseudo_memory, resume_from_chkpt


def save_generator_params(model, results_folder, run_id, min_class, max_class):
    for cls in range(min_class, max_class):
        e_save_path = os.path.join(results_folder, f'generator_encoders_run{run_id}_class{cls}.pth')
        torch.save({'state_dict': model.class_encoders[cls].state_dict()}, e_save_path)
        logger.LOGGER.info(f'Saved generator encoder for class {cls} to file {e_save_path}.')

    d_save_path = os.path.join(results_folder, f'generator_decoders_run{run_id}.pth')
    torch.save({'state_dict': model.universal_decoder.state_dict()}, d_save_path)
    logger.LOGGER.info(f'Saved universal decoder to file {d_save_path}.')


def load_generator_params(model, config, results_folder, run_id, min_class, max_class):
    for cls in range(min_class, max_class):
        e_save_path = os.path.join(results_folder, f'generator_encoders_run{run_id}_class{cls}.pth')
        try:
            state_dict_saved = torch.load(e_save_path, map_location=config['device'][0])
        except Exception as e:
            logger.LOGGER.warning(f'Loading file `{e_save_path}` failed. Try to train it again.')
            return False
        model.class_encoders[cls].load_state_dict(state_dict_saved['state_dict'])
        logger.LOGGER.info(f'Loaded generator encoder for class {cls} from file {e_save_path}.')

    d_save_path = os.path.join(results_folder, f'generator_decoders_run{run_id}.pth')
    try:
        state_dict_saved = torch.load(d_save_path, map_location=config['device'][0])
    except:
        logger.LOGGER.warning(f'Loading file `{d_save_path}` failed. Try to train it again.')
        return False
    model.universal_decoder.load_state_dict(state_dict_saved['state_dict'])
    logger.LOGGER.info(f'Loaded universal decoder from file {d_save_path}.')
    return True


def _after_task(config, model, inc_dataset, run_id, task_id, results_folder):
    regenerate = False
    if config["resume"] and os.path.isdir(config["resume"]) and not config["recompute_meta"] \
            and (config["resume_first"] and task_id == 0):
        loaded = model.load_metadata(config["resume"], run_id)
        regenerate = not loaded
    else:
        model.after_task_intensive(inc_dataset)
    if regenerate:
        logger.LOGGER.warning(f'loaded meta file failed! Regenerating..')
        model.after_task_intensive(inc_dataset, train_generator=True)
    model.after_task(inc_dataset)

    if config["label"] and (
            config["save_model"] == "task" or
            (config["save_model"] == "last" and task_id == inc_dataset.n_tasks - 1) or
            (config["save_model"] == "first" and task_id == 0)
    ):
        model.save_parameters(results_folder, run_id)
        model.save_metadata(results_folder, run_id)


# ----------
# Parameters
# ----------


def _set_results(config, start_date):
    if config["label"]:
        results_folder = results_utils.get_save_folder(config["model"], start_date, config["label"])
    else:
        results_folder = None

    if config["save_model"]:
        logger.LOGGER.info("Model will be save at this rythm: {}.".format(config["save_model"]))

    results = results_utils.get_template_results(config)

    return results, results_folder


def _set_data_model(config, class_order):
    inc_dataset = factory.get_data(config, class_order)
    config["classes_order"] = inc_dataset.class_order

    model = factory.get_model(config)
    model.inc_dataset = inc_dataset

    return inc_dataset, model


def _set_global_parameters(config):
    _set_seed(config["seed"], config["threads"], config["no_benchmark"], config["detect_anomaly"])
    factory.set_device(config)


def _set_seed(seed, nb_threads, no_benchmark, detect_anomaly):
    logger.LOGGER.info("Set seed {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if no_benchmark:
        logger.LOGGER.warning("CUDA algos are not determinists but faster!")
    else:
        logger.LOGGER.warning("CUDA algos are determinists but very slow!")
    torch.backends.cudnn.deterministic = not no_benchmark  # This will slow down training.
    torch.set_num_threads(nb_threads)
    if detect_anomaly:
        logger.LOGGER.info("Will detect autograd anomaly.")
        torch.autograd.set_detect_anomaly(detect_anomaly)


def _set_up_options(args):
    options_paths = args["options"] or []

    autolabel = []
    for option_path in options_paths:
        if not os.path.exists(option_path):
            raise IOError("Not found options file {}.".format(option_path))

        args.update(_parse_options(option_path))

        autolabel.append(os.path.splitext(os.path.basename(option_path))[0])

    return "_".join(autolabel)


def _parse_options(path):
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.load(f, Loader=yaml.FullLoader)
        elif path.endswith(".json"):
            return json.load(f)["config"]
        else:
            raise Exception("Unknown file type {}.".format(path))


# ----
# Misc
# ----


def _aggregate_results(list_results):
    res = str(round(statistics.mean(list_results) * 100, 2))
    if len(list_results) > 1:
        res = res + " +/- " + str(round(statistics.stdev(list_results) * 100, 2))
    return res
