## PanNuke Preparation
The original PanNuke dataset has the following style using just one big array for each dataset split:

```bash
├── fold0
│   ├── images.npy
│   ├── masks.npy
│   └── types.npy
├── fold1
│   ├── images.npy
│   ├── masks.npy
│   └── types.npy
└── fold2
    ├── images.npy
    ├── masks.npy
    └── types.npy
```

For memory efficieny and to make us of multi-threading dataloading with our augmentation pipeline, we reassemble the dataset to the following structure:
```bash
├── fold0
│   ├── cell_count.csv      # cell-count for each image to be used in sampling
│   ├── images              # H&E Image for each sample as .png files
│   ├── images
│   │   ├── 0_0.png
│   │   ├── 0_1.png
│   │   ├── 0_2.png
...
│   ├── labels              # label as .npy arrays for each sample
│   │   ├── 0_0.npy
│   │   ├── 0_1.npy
│   │   ├── 0_2.npy
...
│   └── types.csv           # csv file with type for each image
├── fold1
│   ├── cell_count.csv
│   ├── images
│   │   ├── 1_0.png
...
│   ├── labels
│   │   ├── 1_0.npy
...
│   └── types.csv
├── fold2
│   ├── cell_count.csv
│   ├── images
│   │   ├── 2_0.png
...  
│   ├── labels  
│   │   ├── 2_0.npy
...  
│   └── types.csv  
├── dataset_config.yaml     # dataset config with dataset information
└── weight_config.yaml      # config file for our sampling
```

We provide all configuration files for the PanNuke dataset in the [`configs/datasets/PanNuke`](configs/datasets/PanNuke) folder. Please copy them in your dataset folder. Images and masks have to be extracted using the [`cell_segmentation/datasets/prepare_pannuke.py`](cell_segmentation/datasets/prepare_pannuke.py) script.
