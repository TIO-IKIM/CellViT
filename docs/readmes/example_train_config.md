


**Configuration Structure**
```yaml
# Example configuration for HoverNet-Cell-Segmentation

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
  log_images:               # If images should be logged to WandB for this run. [bool] [Optional, defaults to False]
  group:                    # WandB group tag [str] [Optional, defaults to None]

# seeding
random_seed: 19             # Seed for numpy, pytorch etc. [int]

# hardware
gpu:                        # Number of GPU to run experiment on [int]

# setting paths and dataset
data:
  dataset:                  # Name of dataset, currently supported: PanNuke [str]
  dataset_path:             # Path to dataset, compare ./docs/readmes/cell_segmentation.md for further details [str]
  train_folds: [...]        # List of fold Numbers to use for training [list[int]]
  val_split:                # Percentage of training set that should be used for validation. Either val_split or val_fold must be provided, not both. [float]
  val_folds:                # List of fold Numbers to use for validation [list[int]]
  test_folds:               # List of fold Numbers to use for final testing [list[int]]
  num_nuclei_classes:       # Number of different nuclei classes (including background!, e.g. 5 nuclei classes + background = 6) [int]
  input_shape:              # Input shape of data. [int] [Optional, defaults to 256]

# model options
model:
  backbone:                 # Backbone Type: Options are: default, ViT256, SAM-B, SAM-L, SAM-H
  pretrained_encoder:       # Set path to a pretrained encoder [str]
  pretrained:               # Path to a pretrained model (.pt file) [str, default None]
  embed_dim:                # Embedding dimension for ViT - typical values are 384 (ViT-S), 768 (ViT-B), 1024 (ViT-L), 1280 (ViT-H) [int]
  input_channels:           # Number of input channels, usually 3 for RGB [int, default 3]
  depth:                    # Number of Transformer Blocks to use - typical values are 12 (ViT-S), 12 (ViT-B), 24 (ViT-L), 32 (ViT-H) [int]
  num_heads:                # Number of attention heads for MHA - typical values are 6 (ViT-S), 12 (ViT-B), 16 (ViT-L), 16 (ViT-H) [int]
  extract_layers:           # List of layers to extract for skip connections - starting from 1 with a maximum value equals the depth [int]
  shared_decoders:          # If decoder networks should be shared except for the heads. [bool] [Optional, defaults to False]
  regression_loss:          # If regression loss should be used for binary prediction head. [bool] [Optional, defaults to False]


# loss function settings (best shown by an example). See all implemented loss functions in base_ml.base_loss module
loss:
  nuclei_binary_map:        # Name of the branch. Possible branches are "nuclei_binary_map", "hv_map", "nuclei_type_map", "tissue_types". If not defined default HoverNet settings are used [str]
    bce:                    # Name of the loss [str]
      loss_fn: xentropy_loss  # Loss_fn, name is defined in LOSS_DICT in base_ml.base_loss module [str]
      weight: 1               # Weight parameter [int] [Optional, defaults to 1]
      args:                   # Parameters for the loss function if necessary. Does not need to be set, can also be empty. Just given as an example here
        arg1:                 # 1. Parameter etc. (Name must match to the initialization parameter given for the loss)
    dice:                   # Name of the loss [str]
      loss_fn: dice_loss    # Loss_fn, name is defined in LOSS_DICT in base_ml.base_loss module [str]
      weight: 1             # Weight parameter [int] [Optional, defaults to 1]
    focaltverskyloss:       # Name of the loss [str]
      loss_fn: FocalTverskyLoss # Loss_fn, name is defined in LOSS_DICT in base_ml.base_loss module [str]
      weight: 1             # Weight parameter [int] [Optional, defaults to 1]
      # optional parameters, not implemented yet
  # hv_map:                   another branch
  # nuclei_type_map:          another branch
  # tissue_types:             another branch


# training options
training:
  batch_size: 32            # Training Batch size [int]
  epochs: 100               # Number of Training Epochs to use [int]
  unfreeze_epoch:           # Epoch Number to unfreeze backbone [int]
  drop_rate:                # Dropout rate [float] [Optional, defaults to 0]
  attn_drop_rate:           # Dropout rate in attention layer [float] [Optional, defaults to 0]
  drop_path_rate:           # Dropout rate in paths [float] [Optional, defaults to 0]
  optimizer:                # Pytorch Optimizer Name. All pytorch optimizers (v1.13) are supported. [str]
  optimizer_hyperparameter: # Hyperparamaters for the optimizers, must be named exactly as in the pytorch documation given for the selected optimizer
    lr: 0.001               # e.g. learning-rate for Adam
    betas: [0.85, 0.9]      # e.g. betas for Adam
  early_stopping_patience:  # Number of epochs before applying early stopping after metric has not been improved. Metric used is total loss. [int]
  scheduler:                # Learning rate scheduler. If no scheduler is selected here, then the learning rate stays constant
    scheduler_type:         # Name of learning rate scheduler. Currently implemented: "constant", "exponential", "cosine". [str]
    # hyperparameters       # gamma [default 0.95] for "exponential", "eta_min" [default 1e-5] for CosineAnnealingLR
  sampling_strategy:        # Sampling strategy. Implemented are "random", "cell", "tissue" and "cell+tissue" [str] [Optional, defaults to "random"]
  sampling_gamma:           # Gamma for balancing sampling. Must be between 0 (equal weights) and 1 (100% oversampling) [float] [Optional, defaults to 1]

# transformations, here all options are given. Remove transformations by removing them from this section
transformations:
  randomrotate90:           # RandomRotation90
    p:                      # Probability [float, between 0 and 1]
  horizontalflip:           # HorizontalFlip
    p:                      # Probability [float, between 0 and 1]
  verticalflip:             # VerticalFlip
    p:                      # Probability [float, between 0 and 1]
  downscale:                # Downscaling
    p:                      # Probability [float, between 0 and 1]
    scale:                  # Scaling factor, maximum should be 0.5. Must be smaller than 1 [float, between 0 and 1]
  blur:                     # Blur
    p:                      # Probability [float, between 0 and 1]
    blur_limit:             # Bluring limit, maximum should be 10, recommended 10 [float]
  gaussnoise:               # GaussianNoise
    p:                      # Probability [float, between 0 and 1]
    var_limit:              # Variance limit, maxmimum should be 50, recommended 10 [float]
  colorjitter:              # ColorJitter
    p:                      # Probability [float, between 0 and 1]
    scale_setting:          # Scaling for contrast and brightness, recommended 0.25 [float]
    scale_color:            # Scaling for hue and saturation, recommended 0.1 [float]
  superpixels:              # SuperPixels
    p:                      # Probability [float, between 0 and 1]
  zoomblur:                 # ZoomBlur
    p:                      # Probability [float, between 0 and 1]
  randomsizedcrop:          # RandomResizeCrop
    p:                      # Probability [float, between 0 and 1]
  elastictransform:         # ElasticTransform
    p:                      # Probability [float, between 0 and 1]
  normalize:                # Normalization
    mean:                   # Mean for Normalizing, default to (0.5, 0.5, 0.5) [list[float], between 0 and 1 for each entry]
    std:                    # STD for Normalizing, default to (0.5, 0.5, 0.5) [list[float], between 0 and 1 for each entry]

eval_checkpoint:            # Either select "best_checkpoint.pth", "latest_checkpoint.pth" or one of the intermediate checkpoint names, e.g., "checkpoint_100.pth"

```
