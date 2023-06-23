# -*- coding: utf-8 -*-
# Main entry point for patch-preprocessing
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import inspect
import logging
import os
import sys


logger = logging.getLogger()
logger.addHandler(logging.NullHandler())

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from preprocessing.patch_extraction.src.cli import PreProcessingParser
from preprocessing.patch_extraction.src.patch_extraction import PreProcessor
from utils.tools import close_logger

if __name__ == "__main__":
    configuration_parser = PreProcessingParser()
    configuration, logger = configuration_parser.get_config()
    configuration_parser.store_config()

    slide_processor = PreProcessor(slide_processor_config=configuration)
    slide_processor.sample_patches_dataset()

    logger.info("Finished Preprocessing.")
    close_logger(logger)
