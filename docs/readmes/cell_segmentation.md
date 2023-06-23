# Cell Segmentation

## Training

The data structure used to train cell segmentation networks is different than to train classification networks on WSI/Patient level. Cureently, due to the massive amount of cells inside a WSI, all famous cell segmentation datasets (such like [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke), https://doi.org/10.48550/arXiv.2003.10778) provide just patches with cell annotations. Therefore, we use the following dataset structure (with k folds):

```bash
dataset  
├── dataset_config.yaml  
├── fold0  
│   ├── images  
|   |   ├── 0_imgname0.png
|   |   ├── 0_imgname1.png
|   |   ├── 0_imgname2.png
...
|   |   └── 0_imgnameN.png  
│   ├── labels
|   |   ├── 0_imgname0.npy
|   |   ├── 0_imgname1.npy
|   |   ├── 0_imgname2.npy
...
|   |   └── 0_imgnameN.npy  
|   └── types.csv
├── fold1  
│   ├── images  
|   |   ├── 1_imgname0.png
|   |   ├── 1_imgname1.png
...
│   ├── labels
|   |   ├── 1_imgname0.npy
|   |   ├── 1_imgname1.npy
...
|   └── types.csv
...
└── foldk  
│   ├── images  
    |   ├── k_imgname0.png
    |   ├── k_imgname1.png
...
    ├── labels
    |   ├── k_imgname0.npy
    |   ├── k_imgname1.npy
    └── types.csv
```

Each type csv should have the following header:
```csv
img,type                            # Header
foldnum_imgname0.png,SetTypeHeare   # Each row is one patch with tissue type
```

The labels are numpy masks with the following structure:
TBD

## Add a new dataset
add to dataset coordnator.

All settings of the dataset must be performed in the correspondinng yaml file, under the data section

dataset name is **not** case sensitive!
