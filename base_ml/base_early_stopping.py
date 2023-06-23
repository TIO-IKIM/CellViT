# -*- coding: utf-8 -*-
# Base Machine Learning Experiment
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging

logger = logging.getLogger("__main__")
logger.addHandler(logging.NullHandler())

import wandb


class EarlyStopping:
    """Early Stopping Class

    Args:
        patience (int): Patience to wait before stopping
        strategy (str, optional): Optimization strategy.
            Please select 'minimize' or 'maximize' for strategy. Defaults to "minimize".
    """

    def __init__(self, patience: int, strategy: str = "minimize"):
        assert strategy.lower() in [
            "minimize",
            "maximize",
        ], "Please select 'minimize' or 'maximize' for strategy"

        self.patience = patience
        self.counter = 0
        self.strategy = strategy.lower()
        self.best_metric = None
        self.best_epoch = None
        self.early_stop = False

        logger.info(
            f"Using early stopping with a range of {self.patience} and {self.strategy} strategy"
        )

    def __call__(self, metric: float, epoch: int) -> bool:
        """Early stopping update call

        Args:
            metric (float): Metric for early stopping
            epoch (int): Current epoch

        Returns:
            bool: Returns true if the model is performing better than the current best model,
                otherwise false
        """
        if self.best_metric is None:
            self.best_metric = metric
            self.best_epoch = epoch
            return True
        else:
            if self.strategy == "minimize":
                if self.best_metric >= metric:
                    self.best_metric = metric
                    self.best_epoch = epoch
                    self.counter = 0
                    wandb.run.summary["Best-Epoch"] = epoch
                    wandb.run.summary["Best-Metric"] = metric
                    return True
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                    return False
            elif self.strategy == "maximize":
                if self.best_metric <= metric:
                    self.best_metric = metric
                    self.best_epoch = epoch
                    self.counter = 0
                    wandb.run.summary["Best-Epoch"] = epoch
                    wandb.run.summary["Best-Metric"] = metric
                    return True
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                    return False
