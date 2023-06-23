[![Python 3.9.7](https://img.shields.io/badge/python-3.9.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Flake8 Status](./reports/flake8/flake8-badge.svg)](./reports/flake8/index.html)
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/></a>
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
___
<p align="center">
  <img src="./docs/figures/banner.png"/>
</p>

___

# CellViT: Expanding Horizons with Vision Transformers for Precise Cell Segmentation and Classification
<div align="center">

[Key Features](#key-features) • [Installation](#installation) • [Usage](#usage) • [Training](#training) • [Inference](#inference)

</div>
This is the official PyTorch implementation of the cell detection and instance segmentation algorithm using a comination of Vision Transformer image encoder and U-Net network structure, titled: "CellViT: Expanding Horizons with Vision Transformers for Precise Cell Segmentation and Classification" (Fabian Hörst, Moritz Rempe, Lukas Heine, Constantin Seibold, Julius Keyl, Giulia Baldini, Selma Ugurel, Jens Siveke, Barbara Grünwald, Jan Egger, and Jens Kleesiek, 2023)

This repository contains the code implementation of CellViT, a deep learning-based method for automated instance segmentation of cell nuclei in digitized tissue samples. CellViT utilizes a Vision Transformer architecture and achieves state-of-the-art performance on the PanNuke dataset, a challenging nuclei instance segmentation benchmark.

<p align="center">
  <img src="./docs/figures/network_large.png"/>
</p>


## Key Features
- Utilizes Vision Transformer (ViT) for nuclei instance segmentation.
- Trained and evaluated on the PanNuke dataset, a challenging nuclei segmentation benchmark.
- We provide a fast inference pipeline with connection to current Viewing Software such as *QuPath*
– Achieves state-of-the-art performance on the PanNuke dataset:
  - Mean panoptic quality: 0.51
  - F1-detection score: 0.83

#### Visualization
<div align="center">

![Example](docs/figures/qupath.gif)

</div>


## Installation

1. Clone the repository:
  `git clone https://github.com/TIO-IKIM/CellViT.git`
2. Create a conda environment with Python 3.9.7 version and install conda requirements: `conda create --name pathology_env --file ./requirements_conda.txt python=3.9.7`
3. Activate environment: `conda activate pathology_env`
4. Install torch for for system, as described [here](https://pytorch.org/get-started/locally/). Preferred version is 1.13, see [optional_dependencies](./optional_dependencies.txt) for help.
5. Install pip dependencies: `pip install -r requirements.txt`
---
Optional:
Install optional dependencies `pip install -r optional_dependencies.txt` to get a speedup using [NVIDIA-Clara](https://www.nvidia.com/de-de/clara/) and [CuCIM](https://github.com/rapidsai/cucim)
for preprocessing during inference.


## Usage:

### Project Structure

We are currently using the following folder structure:

```bash
├── base_ml               # Basic Machine Learning Code: CLI, Trainer, Experiment, ...
├── cell_segmentation     # Cell Segmentation training and inference files
├── configs               # Config files
│   ├── examples          # Example config files with explanations
│   └── python            # Python configuration file for global Python settings
├── datamodel             # Datamodels of WSI, Patientes etc. (not ML specific)
├── docs                  # Documentation files (in addition to this main README.md)
├── models                # Machine Learning Models (PyTorch implementations)
│   ├── encoders          # Encoder networks (see ML structure below)
│   ├── pretrained        # Checkpoint of important pretrained models (needs to be downloaded from Google drive)
│   └── segmentation      # CellViT Code
├── preprocessing         # Preprocessing code
│   └── patch_extraction  # Code to extract patches from WSI
```


### Training
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

!!!! Insert link to example yaml file!!!!!

### Inference

Model checkpoints can be downloaded here:

- [CellViT-SAM-H](https://drive.google.com/uc?export=download&id=1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV) 🚀
- [CellViT-256](https://drive.google.com/uc?export=download&id=1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q)
- [CellViT-SAM-H-x20](https://drive.google.com/uc?export=download&id=1wP4WhHLNwyJv97AK42pWK8kPoWlrqi30)
- [CellViT-256-x20](https://drive.google.com/uc?export=download&id=1w99U4sxDQgOSuiHMyvS_NYBiz6ozolN2)

License: [Apache 2.0 with Commons Clause](./LICENSE)

Pre-trained ViT models for training initialization can be downloaded here: [ViT-Models](https://drive.google.com/drive/folders/1zFO4bgo7yvjT9rCJi_6Mt6_07wfr0CKU?usp=sharing)
Please check out the corresponding licenses before distribution and further usage!



### Preprocessing
In our Pre-Processing pipeline, we are able to extract quadratic patches from detected tissue areas, load annotation files (`.json`) and apply color normlizations. We make use of the popular [OpenSlide](https://openslide.org/) library, but extended it with the [RAPIDS cuCIM](https://github.com/rapidsai/cucim) framework for an x8 speedup in patch-extraction. The documentation for the preprocessing can be found [here](./docs/readmes/preprocessing.md).

Preprocessing is necessary to extract patches for our inference pipeline. We use squred patches of size 1024 pixels with an overlap of 64 px.

**Please make sure that you select the following properties for our CellViT inference**
| Parameter     	| Value 	|
|---------------	|-------	|
| patch_size    	| 1024  	|
| patch_overlap 	| 6.25  	|

#### Resulting Dataset Structure
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

The cell detection and segmentation results are stored in a newly created `cell_detection` folder for each WSI.



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
