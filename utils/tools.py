# -*- coding: utf-8 -*-
# Utility functions
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import importlib
import logging
import sys

import types
from datetime import timedelta
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union

from utils.__init__ import logger


# Helper timing functions
def start_timer() -> float:
    """Returns the number of seconds passed since epoch. The epoch is the point where the time starts,
    and is platform dependent.

    Returns:
        float:  The number of seconds passed since epoch
    """
    return timer()


def end_timer(start_time: float, timed_event: str = "Time usage") -> None:
    """Prints the time passed from start_time.


    Args:
        start_time (float): The number of seconds passed since epoch when the timer started
        timed_event (str, optional): A string describing the activity being monitored. Defaults to "Time usage".
    """
    logger.info(f"{timed_event}: {timedelta(seconds=timer() - start_time)}")


def module_exists(
    *names: Union[List[str], str],
    error: str = "ignore",
    warn_every_time: bool = False,
    __INSTALLED_OPTIONAL_MODULES: Dict[str, bool] = {},
) -> Optional[Union[Tuple[types.ModuleType, ...], types.ModuleType]]:
    """Try to import optional dependencies.
    Ref: https://stackoverflow.com/a/73838546/4900327

    Args:
        names (Union(List(str), str)): The module name(s) to import. Str or list of strings.
        error (str, optional): What to do when a dependency is not found:
                * raise : Raise an ImportError.
                * warn: print a warning.
                * ignore: If any module is not installed, return None, otherwise, return the module(s).
            Defaults to "ignore".
        warn_every_time (bool, optional): Whether to warn every time an import is tried. Only applies when error="warn".
            Setting this to True will result in multiple warnings if you try to import the same library multiple times.
            Defaults to False.
    Raises:
        ImportError: ImportError of Module

    Returns:
        Optional[ModuleType, Tuple[ModuleType...]]: The imported module(s), if all are found.
            None is returned if any module is not found and `error!="raise"`.
    """
    assert error in {"raise", "warn", "ignore"}
    if isinstance(names, (list, tuple, set)):
        names: List[str] = list(names)
    else:
        assert isinstance(names, str)
        names: List[str] = [names]
    modules = []
    for name in names:
        try:
            module = importlib.import_module(name)
            modules.append(module)
            __INSTALLED_OPTIONAL_MODULES[name] = True
        except ImportError:
            modules.append(None)

    def error_msg(missing: Union[str, List[str]]):
        if not isinstance(missing, (list, tuple)):
            missing = [missing]
        missing_str: str = " ".join([f'"{name}"' for name in missing])
        dep_str = "dependencies"
        if len(missing) == 1:
            dep_str = "dependency"
        msg = f"Missing optional {dep_str} {missing_str}. Use pip or conda to install."
        return msg

    missing_modules: List[str] = [
        name for name, module in zip(names, modules) if module is None
    ]
    if len(missing_modules) > 0:
        if error == "raise":
            raise ImportError(error_msg(missing_modules))
        if error == "warn":
            for name in missing_modules:
                # Ensures warning is printed only once
                if warn_every_time is True or name not in __INSTALLED_OPTIONAL_MODULES:
                    logger.warning(f"Warning: {error_msg(name)}")
                    __INSTALLED_OPTIONAL_MODULES[name] = False
        return None
    if len(modules) == 1:
        return modules[0]
    return tuple(modules)


def close_logger(logger: logging.Logger) -> None:
    """Closing a logger savely

    Args:
        logger (logging.Logger): Logger to close
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

    logger.handlers.clear()
    logging.shutdown()


class AverageMeter(object):
    """Computes and stores the average and current value

    Original-Code: https://github.com/facebookresearch/simsiam
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary and insert the sep to seperate keys

    Args:
        d (dict): dict to flatten
        parent_key (str, optional): parent key name. Defaults to ''.
        sep (str, optional): Seperator. Defaults to '.'.

    Returns:
        dict: Flattened dict
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = ".") -> dict:
    """Unflatten a flattened dictionary (created a nested dictionary)

    Args:
        d (dict): Dict to be nested
        sep (str, optional): Seperator of flattened keys. Defaults to '.'.

    Returns:
        dict: Nested dict
    """
    output_dict = {}
    for key, value in d.items():
        keys = key.split(sep)
        d = output_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return output_dict


def remove_parameter_tag(d: dict, sep: str = ".") -> dict:
    """Remove all paramter tags from dictionary

    Args:
        d (dict): Dict must be flattened with defined seperator
        sep (str, optional): Seperator used during flattening. Defaults to ".".

    Returns:
        dict: Dict with parameter tag removed
    """
    param_dict = {}
    for k, _ in d.items():
        unflattened_keys = k.split(sep)
        new_keys = []
        max_num_insert = len(unflattened_keys) - 1
        for i, k in enumerate(unflattened_keys):
            if i < max_num_insert and k != "parameters":
                new_keys.append(k)
        joined_key = sep.join(new_keys)
        param_dict[joined_key] = {}
    print(param_dict)
    for k, v in d.items():
        unflattened_keys = k.split(sep)
        new_keys = []
        max_num_insert = len(unflattened_keys) - 1
        for i, k in enumerate(unflattened_keys):
            if i < max_num_insert and k != "parameters":
                new_keys.append(k)
        joined_key = sep.join(new_keys)
        param_dict[joined_key][unflattened_keys[-1]] = v

    return param_dict

def get_size_of_dict(d: dict) -> int:
    size = sys.getsizeof(d)
    for key, value in d.items():
        size += sys.getsizeof(key)
        size += sys.getsizeof(value)
    return size