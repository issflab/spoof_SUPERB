"""
Utilization functions
"""

import os
from collections.abc import Mapping
import random
import sys

import numpy as np
import torch


def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError('invalid truth value {}'.format(val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras-style"""
    return 1. / (1. + decay * step)


class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """SGD with restarts scheduler"""
    def __init__(self, optimizer, T0, T_mul, eta_min, last_epoch=-1):
        self.Ti = T0
        self.T_mul = T_mul
        self.eta_min = eta_min

        self.last_restart = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T_cur = self.last_epoch - self.last_restart
        if T_cur >= self.Ti:
            self.last_restart = self.last_epoch
            self.Ti = self.Ti * self.T_mul
            T_cur = 0

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + np.cos(np.pi * T_cur / self.Ti)) / 2
            for base_lr in self.base_lrs
        ]


def _get_optimizer(model_parameters, optim_config):
    """Defines optimizer according to the given config"""
    optimizer_name = optim_config['optimizer']

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=optim_config['base_lr'],
                                    momentum=optim_config['momentum'],
                                    weight_decay=optim_config['weight_decay'],
                                    nesterov=optim_config['nesterov'])
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=optim_config['base_lr'],
                                     betas=optim_config['betas'],
                                     weight_decay=optim_config['weight_decay'],
                                     amsgrad=str_to_bool(
                                         optim_config['amsgrad']))
    else:
        print('Un-known optimizer', optimizer_name)
        sys.exit()

    return optimizer


def _get_scheduler(optimizer, optim_config):
    """
    Defines learning rate scheduler according to the given config
    """
    if optim_config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay'])

    elif optim_config['scheduler'] == 'sgdr':
        scheduler = SGDRScheduler(optimizer, optim_config['T0'],
                                  optim_config['Tmult'],
                                  optim_config['lr_min'])

    elif optim_config['scheduler'] == 'cosine':
        total_steps = optim_config['epochs'] * \
            optim_config['steps_per_epoch']

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                optim_config['lr_min'] / optim_config['base_lr']))

    elif optim_config['scheduler'] == 'keras_decay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: keras_decay(step))
    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config):
    """Defines an optimizer and a scheduler"""
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader

    NOT currently wired into main.py's DataLoaders, which run with
    num_workers=8 and no worker_init_fn -- so the workers are seeded by torch's
    default, not by this. Passing it as worker_init_fn would change training
    results, so it is a deliberate decision rather than an oversight to fix in
    passing.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _toggle(config, name, default):
    """Read one cuDNN toggle from whatever `config` happens to be.

    Three shapes reach this, and they are not interchangeable:

      * ``None``                -- nothing configured; use `default`
      * an argparse ``Namespace`` -- attributes, already real booleans
      * a mapping                 -- items, historically strings from JSON

    The mapping case goes through `str_to_bool`, which calls ``.lower()`` and
    therefore raises on a real bool. That is exactly why this repo used to carry
    two seeding functions: the vendored one read attributes and the local one
    subscripted a dict through `str_to_bool`, so neither could take the other's
    argument. Only strings are converted.
    """
    if config is None:
        return default
    if hasattr(config, name):
        value = getattr(config, name)
    elif isinstance(config, Mapping) and name in config:
        value = config[name]
    else:
        return default
    return str_to_bool(value) if isinstance(value, str) else bool(value)


def set_seed(seed, config=None):
    """Seed every RNG a training run touches, and fix the cuDNN backend.

    Replaces `core_scripts.startup_config.set_random_seed`, which was the only
    thing this repo used out of a 38-file vendored copy of Xin Wang's
    project-NN-Pytorch-scripts. Behaviour is preserved exactly, including the
    two stdout notices, so a run seeded before and after the removal draws the
    same numbers.

    `config` may be None, an argparse Namespace, or a mapping -- see `_toggle`.
    With None the backend is set to the safe pair (deterministic on, benchmark
    off) rather than raising, because "seed everything and leave cuDNN alone"
    is a reasonable thing for a caller to ask for.

    Note on PYTHONHASHSEED: setting it here cannot affect THIS process, whose
    hash randomisation was fixed before the first line of Python ran. It is set
    because the original did, and because it does reach subprocesses. Making a
    run's `str` hashing reproducible requires the variable to be exported
    before the interpreter starts.
    """
    deterministic = _toggle(config, "cudnn_deterministic_toggle", True)
    benchmark = _toggle(config, "cudnn_benchmark_toggle", False)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if config is not None:
        if not deterministic:
            print("cudnn_deterministic set to False")
        if benchmark:
            print("cudnn_benchmark set to True")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
