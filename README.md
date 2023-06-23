[![Python 3.9.7](https://img.shields.io/badge/python-3.9.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Flake8 Status](./reports/flake8/flake8-badge.svg)](./reports/flake8/index.html)
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/></a>
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
___
<p align="center">
  <img src="./docs/figures/DigitalHistologyHub.png"/>
</p>

___



# DigitalHistologyHub
DigitalHistologyHub is a comprehensive repository that provides a complete solution for whole-slide image processing in the field of digital histology. It includes a set of tools and utilities for image preprocessing, specifically patching, to improve the quality of images and make them more suitable for analysis. Additionally, the repository provides a collection of state-of-the-art neural network models implemented using PyTorch for analyzing these images.

The repository is designed to be a one-stop-shop for digital histology image processing and analysis, making it a valuable resource for researchers and practitioners in the field. The user-friendly interface allows for easy and efficient access to the tools and models provided. In addition, the repository includes a variety of visualization techniques to enable the user to better understand and analyze the results.

Overall, DigitalHistologyHub is a valuable resource for anyone interested in digital histology and is intended to make the process of analyzing whole-slide images more efficient and accessible.

## Table of Contents
- [DigitalHistologyHub](#digitalhistologyhub)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Folder Structure](#folder-structure)
  - [Preprocessing](#preprocessing)
    - [Resulting Dataset Structure](#resulting-dataset-structure)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
    - [Structure](#structure)
      - [Classification using Multiple Instance Learning](#classification-using-multiple-instance-learning)
    - [Encoding](#encoding)
    - [Decoding](#decoding)
      - [CLI](#cli)
      - [Splitting and Filelists](#splitting-and-filelists)


## Installation
1.) Create a conda environment with Python 3.9.7 version and install conda requirements: `conda create --name pathology_env --file ./requirements_conda.txt python=3.9.7`

2.) Activate environment: `conda activate pathology_env`

3.) Install torch for for system, as described [here](https://pytorch.org/get-started/locally/). Preferred version is 1.13, see [optional_dependencies](./optional_dependencies.txt) for help.

4.) Install pip dependencies: `pip install -r requirements.txt`

---
Optional:

5.) Install optional dependencies `pip install -r optional_dependencies.txt` to get a speedup using [NVIDIA-Clara](https://www.nvidia.com/de-de/clara/) and [CuCIM](https://github.com/rapidsai/cucim)

---

## Folder Structure
We are currently using the following folder structure:

```bash
├── base_ml               # Basic Machine Learning Code: CLI, Trainer, Experiment, ...
├── classification        # Classification Machine Learning Code: Specific training for classifiers
├── configs               # Config files
│   ├── examples            # Example config files with explanations
│   ├── projects            # Project specific configuration file (e.g., Tumor detection)
│   └── python              # Python configuration file for global Python settings
├── datamodel             # Datamodels of WSI, Patientes etc. (not ML specific)
├── docs                  # Documentation files (in addition to this main README.md)
├── file_handling         # File handling code, e.g., for creating splits
├── models                # Machine Learning Models (PyTorch implementations)
│   ├── decoders            # Decoder networks (see ML structure below)
│   ├── encoders            # Encoder networks (see ML structure below)
│   ├── pretrained          # Checkpoint of important pretrained models
├── notebooks             # Folder for additional notebooks
├── preprocessing         # Preprocessing code
│   ├── encoding            # Encoding code to create embeddings of patches
│   └── patch_extraction    # Code to extract patches from WSI
├── test_database         # Test database for testing purposes
│   ├── examples            # Example files
│   ├── input               # WSI files and annotations
├── tests                 # Unittests
```


## Preprocessing
Whole-slide image preprocessing is an essential step in digital pathology that involves preparing digital images of entire tissue slides for analysis by pathologists and machine learning algorithms. Whole-slide images are typically high-resolution digital images of tissue slides, which can range in size from hundreds of megabytes to several gigabytes. The large size of these images presents a challenge for analysis, as processing them directly can be computationally intensive and time-consuming.

Preprocessing whole-slide images typically involves dividing the images into smaller, more manageable patches or tiles. These patches are typically square or rectangular, and can be selected using a regular grid pattern or using more sophisticated methods that take into account the content of the image. Once the patches are extracted, various techniques can be applied to improve the quality and consistency of the image data.

In our Pre-Processing pipeline, we are able to extract quadratic patches from detected tissue areas, load annotation files (`.json`) and apply color normlizations. We make use of the popular [OpenSlide](https://openslide.org/) library, but extended it with the [RAPIDS cuCIM](https://github.com/rapidsai/cucim) framework for an x8 speedup in patch-extraction.

The documentation for the preprocessing can be found [here](./docs/readmes/preprocessing.md).

### Resulting Dataset Structure
In general, the folder structure for a preprocessed dataset looks like this:
The aim of pre-processing is to create one dataset per WSI in the following structure:
```bash
WSI_Name
├── annotation_masks      # thumbnails of extracted annotation masks
│   ├── all_overlaid.png  # all with same dimension as the thumbnail
│   ├── tumor.png
│   └── ...  
├── context               # context patches, if extracted
│   ├── 2                 # subfolder for each scale
│   │   ├── WSI_Name_row1_col1_context_2.png
│   │   ├── WSI_Name_row2_col1_context_2.png
│   │   └── ...
│   └── 4
│   │   ├── WSI_Name_row1_col1_context_2.png
│   │   ├── WSI_Name_row2_col1_context_2.png
│   │   └── ...
├── masks                 # Mask (numpy) files for each patch -> optional folder for segmentation
│   ├── WSI_Name_row1_col1.npy
│   ├── WSI_Name_row2_col1.npy
│   └── ...
├── metadata              # Metadata files for each patch
│   ├── WSI_Name_row1_col1.yaml
│   ├── WSI_Name_row2_col1.yaml
│   └── ...
├── patches               # Patches as .png files
│   ├── WSI_Name_row1_col1.png
│   ├── WSI_Name_row2_col1.png
│   └── ...
├── thumbnails            # Different kind of thumbnails
│   ├── thumbnail_mpp_5.png
│   ├── thumbnail_downsample_32.png
│   └── ...
├── tissue_masks          # Tissue mask images for checking
│   ├── mask.png          # all with same dimension as the thumbnail
│   ├── mask_nogrid.png
│   └── tissue_grid.png
├── mask.png              # tissue mask with green grid  
├── metadata.yaml         # WSI metdata for patch extraction
├── patch_metadata.json   # Patch metadata of WSI merged in one file
└── thumbnail.png         # WSI thumbnail
```
For later usage with two-stage models (see [Encoding](#encoding) section), another folder is created for slide-embeddings:
```bash
WSI_Name
├── annotation_masks  
│   └── ...  
├── embeddings          # Embeddings of the WSI
│   ├── # Embedding vector for all patches as torch tensor with given backbone name and comment (defined during encoding phase)
│   ├── # Each embedding vector has the shape [num_patches, embedding_dimension]
│   ├── embedding_{backbone+comment}.pt  
│   ├── # Important metadata such like row, col position of each patch, wsi_metadata, intersected labels etc.
│   ├── embedding_{backbone+comment}_metadata.json
...
```

## Machine Learning Pipeline

### Structure
The Base Machine Learning Code can be found in the [`base_ml`](./base_ml) folder.
A standard ML-Pipeline experiment consists of a CLI parsing the argument ([`base_cli.py`](./base_ml/base_cli.py)). For each experiment type, a run script which is setting up the experiment needs to be defined (e.g., compare [`run_clam.py`](./classification/run_clam.py)). The run code for the training is always wrapped in an exepriment class. A base class is provided in the [`base_experiment.py`](./base_ml/base_experiment.py) file. The experiment is started using the `run_experiment` method. Inside the `run_experiment` method, a trainer object needs to be instantited, and the training is started using the `trainer.fit` method.

**Currently, the ML-Pipeline is as follows:**
<p align="center">
  <img src="./docs/figures/network.png" width="1500"/>
</p>
A WSI is preprocessed and the preprocessing results are stored for later usage. Then, for each patch, embeddings are calculated with an encoder network, which are subsequently aggregated using the Multiple-Instance-Learning paradigm in a specific Decoder network to a slide/patient embedding used for prediction. In a future release, thus structure is relaxed to a more general solution to also provide the opportunity to use Graph-Aggregation networks or combined Encoder-Decoder networks (e.g, stacked Vision Transformers).

#### Classification using Multiple Instance Learning
Classification using Multiple Instance Learning (MIL) is based on the described encoder-decoder structure.

Currently, the following encoder networks are available:

* ResNet50
* ResNet50Bottleneck (recommended, compare https://github.com/mahmoodlab/CLAM)
* ResNet34
* ResNet18

For the decoder, the following networks are available:

* CLAM (https://github.com/mahmoodlab/CLAM)
* DeepMIL (https://proceedings.mlr.press/v80/ilse18a.html)
* Baseline Mean and Max Pooling

More networks are continuously integrated.

### Encoding
TBD

### Decoding
Note: This part assumes that embeddings have already been generated
#### CLI
The CLI for a ML-experiment is as follows (here the [```run_clam.py```](./classification/run_clam.py) script is used):
```bash
usage: run_clam.py [-h] --config CONFIG [--gpu GPU] [--sweep | --agent AGENT | --checkpoint CHECKPOINT]

Start an experiment with given configuration file.

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             Cuda-GPU ID (default: None)
  --sweep               Starting a sweep. For this the configuration file must be structured according to WandB sweeping. Compare
                        https://docs.wandb.ai/guides/sweeps and https://community.wandb.ai/t/nested-sweep-configuration/3369/3 for further
                        information. This parameter cannot be set in the config file! (default: False)
  --agent AGENT         Add a new agent to the sweep. Please pass the sweep ID as argument in the way entity/project/sweep_id, e.g.,
                        user1/test_project/v4hwbijh. The agent configuration can be found in the WandB dashboard for the running sweep in
                        the sweep overview tab under launch agent. Just paste the entity/project/sweep_id given there. The provided config
                        file must be a sweep config file.This parameter cannot be set in the config file! (default: None)
  --checkpoint CHECKPOINT
                        Path to a PyTorch checkpoint file. The file is loaded and continued to train with the provided settings. If this is
                        passed, no sweeps are possible. This parameter cannot be set in the config file! (default: None)

required named arguments:
  --config CONFIG       Path to a config file (default: None)
```

The important file is the configuration file, in which all paths are set, the model configuration is given and the hyperparameters or sweeps are defined. For each specific run file, there exists an example file in the [./configs/examples/classification](./configs/examples/classification) folder with the same naming as well as a configuration file that explains how to run WandB sweeps for hyperparameter search. A general configuration for all experiments is given one section below.

All metrics defined in your trainer are logged to WandB. The WandB configuration needs to be set up in the configuration file.

**Configuration Structure**
```yaml
# Base configuration for all ML experiments

# comment and project setup for wandb
logging:
  mode:                     # "online" or "offline" [str]
  project:                  # Name of project to use [str]
  notes:                    # Notes about the run, verbose description [str]
  log_comment:              # Comment to add to name the local logging folder [str]
  tags:                     # List of tags, e.g., ["baseline", "run1"] [str]
    - "tag1"
    - "tag2"
    - "..."
  wandb_dir:                # Direcotry to store the wandb file. CAREFUL: Directory must exists [str]
  log_dir:                  # Direcotry to store all logging related files and outputs [str]
  level:                    # Level of logging must be either ["critical", "error", "warning", "info", "debug"] [str]

# seeding
random_seed: 19             # Seed for numpy, pytorch etc. [int]

# hardware
gpu:                        # Number of GPU to run experiment on [int]

# setting paths and dataset
wsi_paths:                  # Path to WSI files [str]
patch_dataset_path:         # Path to patched dataset (parent path of dataset) [str]
filelist:                   # Filelists with all files (not splitted). Required columns are "Patient", "Filename" and one type of label as defined in the label key.
                            #   An example filelist is provided here: ./test_database/examples/filelist.csv and below [str]
split_path:                 # Path to splitting filelist. Either parent path of a train-val-test split or parent path of folding with fold1 ... fold n as subfolders
                            #   Each subfolder must contain the following files: test_split, train_split and val_split
                            #   An example split is provided here: ./test_database/examples/split and here:
                            #   [str]
label:                      # Training label name, must be a column name that is apparent in the splits and the filelist. [str]
label_map:                  # Verbose label map, below is an example given [dict]
  # e.g.,
  # Healthy: 0
  # Tumor: 1
  # ...

# model options
model:
  # some model options, specific for experiment
  pretrained:               # Path to a pretrained model (.pt file) [str, default None]

embeddings:
  feature_dimensions:       # Input embedding dimensions. e.g., 512 for ResNet34, 1024 for ResNet50Bottleneck and 2048 for ResNet50 [int] [Optional, default 1024]
  backbone:                 # Name of backbone used, necessary to load embeddings from the patched folder. Embeddings must be calculated in advance [str]
  embedding_comment:        # [Optional, defaults to None] If an embedding comment has been used during encoding, please also give it here [str]

# training options
training:
  # loss, dropout, scheduling are specific for experiments
  epochs:                   # Number of Training Epochs to use
  accumulation_steps:       # Gradient accumulation steps. Used for gradient accumulation, because batch-size is always 1 in training. 1 means no accumulation.
                            # See here: https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa
                            # [int] [Optional, default 1]
  optimizer:                # Pytorch Optimizer Name. All pytorch optimizers (v1.13) are supported. [str]
  optimizer_hyperparameter: # Hyperparamaters for the optimizers, must be named exactly as in the pytorch documation given for the selected optimizer
  early_stopping_patience:  # Number of epochs before applying early stopping after metric has not been improved. Metric used is AUC. [int]

```

#### Splitting and Filelists
**Train-Val-Test-Split**
The folder structure for a simple train-val-test split should be like this
```bash
├── test_split.csv
├── train_split.csv
└── val_split.csv
```
with each file beeing structured the following way:
```csv
Patient,Filename,Label1,Label2,Label3...  # Header
1,wsi_name.csv,label1,label2,label3       # Each row is one WSI file
```
The Labels (1...N) can be labelled as you want, and can always be defined in the configurations
If no external test set is used, just create a .csv file with the header and no rows.

**K-Fold/MCCV**
For K-Fold Cross-Validation and Monte-Carlo Cross-Validation, the folder structure consists of one folder for each fold and one main `test_split.csv` file:
```bash
├── fold0
│   ├── train_split.csv
│   └── val_split.csv
├── fold1
│   ├── train_split.csv
│   └── val_split.csv
...
├── foldK
│   ├── train_split.csv
│   └── val_split.csv
└── test_split.csv
```

Further examples are provided in the [./tests/static_test_files/splitting](./tests/static_test_files/splitting) folder.

**Further Documentation**:

Further documentation for file splitting and how to create splits is given [here](./docs/readmes/splitting.md).
