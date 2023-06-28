## MoNuSeg Preparation
The original PanNuke dataset has the following style using .xml annotations and .tiff files with a size of $1000 \times 1000$ pixels:

```bash
├── testing
│   ├── images
│   │   ├── TCGA-2Z-A9J9-01A-01-TS1.tif
│   │   ├── TCGA-44-2665-01B-06-BS6.tif
...
│   └── labels
│       ├── TCGA-2Z-A9J9-01A-01-TS1.xml
│       ├── TCGA-44-2665-01B-06-BS6.xml
...
└── training
    ├── images
    └── labels
```
For our experiments, we resized the dataset images to $1024 \times 1024$ pixels and convert the .xml annotations to binary masks:
```bash
├── testing
│   ├── images
│   │   ├── TCGA-2Z-A9J9-01A-01-TS1.png
│   │   ├── TCGA-44-2665-01B-06-BS6.png
...
│   └── labels
│   │   ├── TCGA-2Z-A9J9-01A-01-TS1.npy
│   │   ├── TCGA-44-2665-01B-06-BS6.npy
...
└── training
    ├── images
    └── labels
```

Everythin can be extracted using the [`cell_segmentation/datasets/prepare_monuseg.py`](cell_segmentation/datasets/prepare_monuseg.py) script.
