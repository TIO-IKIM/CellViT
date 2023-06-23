# -*- coding: utf-8 -*-
# Base CLI to parse Arguments
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import argparse
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union

import yaml
from pydantic import BaseModel


class ABCParser(ABC):
    """Blueprint for Argument Parser"""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_config(self) -> Tuple[Union[BaseModel, dict], logging.Logger]:
        """Load configuration and create a logger

        Returns:
            Tuple[PreProcessingConfig, logging.Logger]: Configuration and Logger
        """
        pass

    @abstractmethod
    def store_config(self) -> None:
        """Store the config file in the logging directory to keep track of the configuration."""
        pass


class ExperimentBaseParser:
    """Configuration Parser for Machine Learning Experiments"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Start an experiment with given configuration file.",
        )
        requiredNamed = parser.add_argument_group("required named arguments")
        requiredNamed.add_argument(
            "--config", type=str, help="Path to a config file", required=True
        )
        parser.add_argument("--gpu", type=int, help="Cuda-GPU ID")
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "--sweep",
            action="store_true",
            help="Starting a sweep. For this the configuration file must be structured according to WandB sweeping. "
            "Compare https://docs.wandb.ai/guides/sweeps and https://community.wandb.ai/t/nested-sweep-configuration/3369/3 "
            "for further information. This parameter cannot be set in the config file!",
        )
        group.add_argument(
            "--agent",
            type=str,
            help="Add a new agent to the sweep. "
            "Please pass the sweep ID as argument in the way entity/project/sweep_id, e.g., user1/test_project/v4hwbijh. "
            "The agent configuration can be found in the WandB dashboard for the running sweep in the sweep overview tab "
            "under launch agent. Just paste the entity/project/sweep_id given there. The provided config file must be a sweep config file."
            "This parameter cannot be set in the config file!",
        )
        group.add_argument(
            "--checkpoint",
            type=str,
            help="Path to a PyTorch checkpoint file. "
            "The file is loaded and continued to train with the provided settings. "
            "If this is passed, no sweeps are possible. "
            "This parameter cannot be set in the config file!",
        )

        self.parser = parser

    def parse_arguments(self) -> Tuple[Union[BaseModel, dict]]:
        """Parse the arguments from CLI and load yaml config

        Returns:
            Tuple[Union[BaseModel, dict]]: Parsed arguments
        """
        # parse the arguments
        opt = self.parser.parse_args()
        with open(opt.config, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)
            yaml_config_dict = dict(yaml_config)

        opt_dict = vars(opt)
        # check for gpu to overwrite with cli argument
        if "gpu" in opt_dict:
            if opt_dict["gpu"] is not None:
                yaml_config_dict["gpu"] = opt_dict["gpu"]

        # check if either training, sweep, checkpoint or start agent should be called
        # first step: remove such keys from the config file
        if "run_sweep" in yaml_config_dict:
            yaml_config_dict.pop("run_sweep")
        if "agent" in yaml_config_dict:
            yaml_config_dict.pop("agent")
        if "checkpoint" in yaml_config_dict:
            yaml_config_dict.pop("checkpoint")

        # select one of the options
        if "sweep" in opt_dict and opt_dict["sweep"] is True:
            yaml_config_dict["run_sweep"] = True
        else:
            yaml_config_dict["run_sweep"] = False
        if "agent" in opt_dict:
            yaml_config_dict["agent"] = opt_dict["agent"]
        if "checkpoint" in opt_dict:
            if opt_dict["checkpoint"] is not None:
                yaml_config_dict["checkpoint"] = opt_dict["checkpoint"]

        self.config = yaml_config_dict

        return self.config
