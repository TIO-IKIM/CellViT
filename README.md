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

[Key Features](#key-features) • [Installation](#installation) • [Usage](#usage) • [Training](#training) • [Inference](#inference) • [Examples](#examples)

</div>
This is the official PyTorch implementation of the cell detection and instance segmentation algorithm using a comination of Vision Transformer image encoder and U-Net network structure, titled: "CellViT: Expanding Horizons with Vision Transformers for Precise Cell Segmentation and Classification" (Fabian Hörst, Moritz Rempe, Lukas Heine, Constantin Seibold, Julius Keyl, Giulia Baldini, Selma Ugurel, Jens Siveke, Barbara Grünwald, Jan Egger, and Jens Kleesiek, 2023)

This repository contains the code implementation of CellViT, a deep learning-based method for automated instance segmentation of cell nuclei in digitized tissue samples. CellViT utilizes a Vision Transformer architecture and achieves state-of-the-art performance on the PanNuke dataset, a challenging nuclei instance segmentation benchmark.

<p align="center">
  <img src="./docs/figures/network_large.png"/>
</p>


## Key Features
  - State-of-the-Art Performance: CellViT outperforms existing methods for nuclei instance segmentation by a substantial margin, delivering superior results on the PanNuke dataset:
    - Mean panoptic quality: 0.51
    - F1-detection score: 0.83
  - Vision Transformer Encoder: The project incorporates pre-trained Vision Transformer (ViT) encoders, which are known for their effectiveness in various computer vision tasks. This choice enhances the segmentation performance of CellViT.
  - U-Net Architecture: CellViT adopts a U-Net-shaped encoder-decoder network structure, allowing for efficient and accurate nuclei instance segmentation. The network architecture facilitates both high-level and low-level feature extraction for improved segmentation results.
  - Weighted Sampling Strategy: To enhance the performance of CellViT, a novel weighted sampling strategy is introduced. This strategy improves the representation of challenging nuclei instances, leading to more accurate segmentation results.
  - Fast Inference on Gigapixel WSI: The framework provides fast inference results by utilizing a large inference patch size of $1024 \times 1024$ pixels, in contrast to the conventional $256$-pixel-sized patches. This approach enables efficient analysis of Gigapixel Whole Slide Images (WSI) and generates localizable deep features that hold potential value for downstream tasks. We provide a fast inference pipeline with connection to current Viewing Software such as *QuPath*


#### Visualization
<div align="center">

![Example](docs/figures/qupath.gif)

</div>


## Installation

1. Clone the repository:
  `git clone https://github.com/TIO-IKIM/CellViT.git`
2. Create a conda environment with Python 3.9.7 version and install conda requirements: `conda env create -f environment.yml`. You can change the environment name by editing the `name` tag in the environment.yaml file.
This step is necessary, as we need to install `Openslide` with binary files. This is easier with conda. Otherwise, installation from [source](https://openslide.org/api/python/) needs to be performed and packages installed with pi
3. Activate environment: `conda activate cellvit_env`
4. Install torch for for system, as described [here](https://pytorch.org/get-started/locally/). Preferred version is 1.13, see [optional_dependencies](./optional_dependencies.txt) for help. You can find all version here: https://pytorch.org/get-started/previous-versions/

5. Install optional dependencies `pip install -r optional_dependencies.txt` to get a speedup using [NVIDIA-Clara](https://www.nvidia.com/de-de/clara/) and [CuCIM](https://github.com/rapidsai/cucim) for preprocessing during inference. Please select your cude versions. Help for installing cucim can be found [online](https://github.com/rapidsai/cucim).
**Note: cannot import name CuImage from cucim**
If you get this error, install cucim from conda to get all binary files.
First remove your previous dependeny with `pip uninstall cupy-cuda117` and reinstall with `
conda install -c rapidsai cucim` inside your conda environment. Also follow their [official guideline](https://github.com/rapidsai/cucim).

## Usage:

### Project Structure

We are currently using the following folder structure:

```bash
├── base_ml               # Basic Machine Learning Code: CLI, Trainer, Experiment, ...
├── cell_segmentation     # Cell Segmentation training and inference files
│   ├── datasets          # Datasets (PyTorch)
│   ├── experiments       # Specific Experiment Code for different experiments
│   ├── inference         # Inference code for experiment statistics and plots
│   ├── trainer           # Trainer functions to train networks
│   ├── utils             # Utils code
│   └── run_xxx.py        # Run file to start an experiment
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
The CLI for a ML-experiment to train the CellViT-Network is as follows (here the [```run_cellvit.py```](cell_segmentation/run_cellvit.py) script is used):
```bash
usage: run_cellvit.py [-h] --config CONFIG [--gpu GPU] [--sweep | --agent AGENT | --checkpoint CHECKPOINT]

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

The important file is the configuration file, in which all paths are set, the model configuration is given and the hyperparameters or sweeps are defined. For each specific run file, there exists an example file in the [./configs/examples/classification](configs/examples/cell_segmentation) folder with the same naming as well as a configuration file that explains how to run WandB sweeps for hyperparameter search. All metrics defined in your trainer are logged to WandB. The WandB configuration needs to be set up in the configuration file, but also turned off by the user.


An example config file is given [here](configs/examples/cell_segmentation/train_cellvit.yaml) with explanations [here](docs/readmes/example_train_config.md):

### Inference

Model checkpoints can be downloaded here:

- [CellViT-SAM-H](https://drive.google.com/uc?export=download&id=1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV) 🚀
- [CellViT-256](https://drive.google.com/uc?export=download&id=1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q)
- [CellViT-SAM-H-x20](https://drive.google.com/uc?export=download&id=1wP4WhHLNwyJv97AK42pWK8kPoWlrqi30)
- [CellViT-256-x20](https://drive.google.com/uc?export=download&id=1w99U4sxDQgOSuiHMyvS_NYBiz6ozolN2)

License: [Apache 2.0 with Commons Clause](./LICENSE)

Pre-trained ViT models for training initialization can be downloaded here: [ViT-Models](https://drive.google.com/drive/folders/1zFO4bgo7yvjT9rCJi_6Mt6_07wfr0CKU?usp=sharing)
Please check out the corresponding licenses before distribution and further usage!

##### Steps
The following steps are necessary for preprocessing:
1. Prepare WSI with our preprocessing pipeline
2. Run inference with the [`inference/cell_detection.py`](/cell_segmentation/inference/cell_detection.py) script

Results are stored at preprocessing locations

#### 1. Preprocessing
In our Pre-Processing pipeline, we are able to extract quadratic patches from detected tissue areas, load annotation files (`.json`) and apply color normlizations. We make use of the popular [OpenSlide](https://openslide.org/) library, but extended it with the [RAPIDS cuCIM](https://github.com/rapidsai/cucim) framework for an x8 speedup in patch-extraction. The documentation for the preprocessing can be found [here](/docs/readmes/preprocessing.md).

Preprocessing is necessary to extract patches for our inference pipeline. We use squared patches of size 1024 pixels with an overlap of 64 px.

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

#### 2. Cell detection script
If the data is prepared, use the [`cell_detection.py`](inference/cell_detection.py) script inside the `cell_segmentation/inference` folder to perform inference:

```bash
usage: cell_detection.py --model MODEL [--gpu GPU] [--magnification MAGNIFICATION] [--outdir_subdir OUTDIR_SUBDIR]
                          [--geojson] {process_wsi,process_dataset} ...

Perform CellViT inference for given run-directory with model checkpoints and logs

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             Cuda-GPU ID for inference. Default: 0 (default: 0)
  --magnification MAGNIFICATION
                        Network magnification. Is used for checking patch magnification such that
                        we use the correct resolution for network. Default: 40 (default: 40)
  --outdir_subdir OUTDIR_SUBDIR
                        If provided, a subdir with the given name is created in the cell_detection folder
                        where the results are stored. Default: None (default: None)
  --geojson             Set this flag to export results as additional geojson files for
                        loading them into Software like QuPath. (default: False)

required named arguments:
  --model MODEL         Model checkpoint file that is used for inference (default: None)

subcommands:
  Main run command for either performing inference on single WSI-file or on whole dataset

  {process_wsi,process_dataset}
```
##### Single WSI
For processing a single WSI file, you need to select the `process_wsi` (`python3 cell_detection.py process_wsi`) subcommand with the following structure:
```bash
usage: cell_detection.py process_wsi --wsi_path WSI_PATH --patched_slide_path PATCHED_SLIDE_PATH

Process a single WSI file

arguments:
  -h, --help            show this help message and exit
  --wsi_path WSI_PATH   Path to WSI file
  --patched_slide_path PATCHED_SLIDE_PATH
                        Path to patched WSI file (specific WSI file, not parent path of patched slide dataset)
```
##### Multiple WSI
To process an entire dataset, select `process_dataset` (`python3 cell_detection.py process_dataset`):
```bash
usage: cell_detection.py process_dataset  --wsi_paths WSI_PATHS --patch_dataset_path PATCH_DATASET_PATH [--filelist FILELIST] [--wsi_extension WSI_EXTENSION]

Process a whole dataset

arguments:
  -h, --help            show this help message and exit
  --wsi_paths WSI_PATHS
                        Path to the folder where all WSI are stored
  --patch_dataset_path PATCH_DATASET_PATH
                        Path to the folder where the patch dataset is stored
  --filelist FILELIST   Filelist with WSI to process. Must be a .csv file with one row denoting the filenames (named 'Filename').
                        If not provided, all WSI files with given ending in the WSI folder are processed. (default: 'None')
  --wsi_extension WSI_EXTENSION
                        The extension types used for the WSI files, see configs.python.config (WSI_EXT). (default: 'svs')
```

#### 3. Example
We provide an example TCGA file to show the performance and usage of our algorithms.
Files and scripts can be found in the [example](example) folder.

**Preprocessing:**
```bash
python3 ./preprocessing/patch_extraction/main_extraction.py --config ./example/preprocessing_example.yaml
```

Output is stored inside `./example/output/preprocessing`

**Inference:**
Download the models and store them in `` or on your preferred location and change the model parameter.

```bash
python3 ./cell_segmentation/inference/cell_detection.py \
  --model ./models/pretrained/CellViT/CellViT-SAM-H-x40.pth\
  --gpu 1 \
  --geojson \
  process_wsi \
  --wsi_path ./example/TCGA-V5-A7RE-11A-01-TS1.57401526-EF9E-49AC-8FF6-B4F9652311CE.svs \
  --patched_slide_path ./example/output/preprocessing/TCGA-V5-A7RE-11A-01-TS1.57401526-EF9E-49AC-8FF6-B4F9652311CE
```
You can import your results (.geojson files) into [QuPath](https://qupath.github.io/).

## Docker Image (Coming Soon) 🐳

In a future release, we will provide a Docker image that contains all the necessary dependencies and configurations pre-installed. This Docker image will ensure reproducibility and simplify the setup process, allowing for easy installation and usage of the project.

Stay tuned for updates on the availability of the Docker image, as we are actively working on providing this convenient packaging option for our project. 🚀
