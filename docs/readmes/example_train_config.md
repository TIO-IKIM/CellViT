


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
