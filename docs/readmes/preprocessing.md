# Preprocessing

In our Pre-Processing pipeline, we are able to extract quadratic patches from detected tissue areas, load annotation files (`.json`) and apply color normlizations. We make use of the popular [OpenSlide](https://openslide.org/) library, but extended it with the [RAPIDS cuCIM](https://github.com/rapidsai/cucim) framework for a speedup in patch-extraction.

The CLI of the main script for patch extraction ([main_extraction](preprocessing/main_extraction.py)) is as follows:

```bash
python3 main_extraction.py [-h]
                          [--wsi_paths WSI_PATHS]
                          [--wsi_filelist WSI_FILELIST]
                          [--output_path OUTPUT_PATH]
                          [--wsi_extension {svs}]
                          [--config CONFIG]
                          [--patch_size PATCH_SIZE]
                          [--patch_overlap PATCH_OVERLAP]
                          [--target_mpp TARGET_MPP]
                          [--target_mag TARGET_MAG]
                          [--downsample DOWNSAMPLE]
                          [--level LEVEL]
                          [--context_scales [CONTEXT_SCALES ...]]
                          [--check_resolution CHECK_RESOLUTION]
                          [--processes PROCESSES]
                          [--overwrite]
                          [--annotation_paths ANNOTATION_PATHS]
                          [--annotation_extension {json,xml}]
                          [--incomplete_annotations]
                          [--label_map_file LABEL_MAP_FILE]
                          [--save_only_annotated_patches]
                          [--exclude_classes EXCLUDE_CLASSES]
                          [--store_masks]
                          [--overlapping_labels]
                          [--normalize_stains]
                          [--normalization_vector_json NORMALIZATION_VECTOR_JSON]
                          [--min_intersection_ratio MIN_INTERSECTION_RATIO]
                          [--tissue_annotation TISSUE_ANNOTATION]
                          [--masked_otsu]
                          [--otsu_annotation OTSU_ANNOTATION]
                          [--filter_patches FILTER_PATCHES]
                          [--apply_prefilter APPLY_PREFILTER]
                          [--log_path LOG_PATH]
                          [--log_level {critical,error,warning,info,debug}]
                          [--hardware_selection {cucim,openslide}]
                          [--wsi_properties DICT]

optional arguments:
  -h, --help            show this help message and exit
  --wsi_paths WSI_PATHS
                        Path to the folder where all WSI are stored or path to a single WSI-file. (default: None)
  --wsi_filelist WSI_FILELIST
                        Path to a csv-filelist with WSI files (separator: `,`), if provided just these files are
                        used.Must include full paths to WSIs, including suffixes.Can be used as an replacement for
                        the wsi_paths option.If both are provided, yields an error. (default: None)
  --output_path OUTPUT_PATH
                        Path to the folder where the resulting dataset should be stored. (default: None)
  --wsi_extension {svs}
                        The extension types used for the WSI files, the options are: ['svs'] (default: None)
  --config CONFIG       Path to a config file. The config file can hold the same parameters as the CLI. Parameters
                        provided with the CLI are always having precedence over the parameters in the config file.
                        (default: None)
  --patch_size PATCH_SIZE
                        The size of the patches in pixel that will be retrieved from the WSI, e.g. 256 for 256px
                        (default: None)
  --patch_overlap PATCH_OVERLAP
                        The percentage amount pixels that should overlap between two different patches. Please
                        Provide as integer between 0 and 100, indicating overlap in percentage. (default: None)
  --downsample DOWNSAMPLE
                        Each WSI level is downsampled by a factor of 2, downsample expresses which kind of
                        downsampling should be used with respect to the highest possible resolution. Medium
                        priority, gets overwritten by target_mag if provided, but overwrites level. (default:
                        None)
  --target_mpp TARGET_MPP
                        If this parameter is provided, the output level of the WSI corresponds to the level that
                        is at the target microns per pixel of the WSI. Alternative to target_mag, downsaple and level.
                        Highest priority,
                        overwrites target_mag, downsample and level if provided. (default: None)
  --target_mag TARGET_MAG
                        If this parameter is provided, the output level of the WSI corresponds to the level that
                        is at the target magnification of the WSI. Alternative to target_mpp, downsaple and level.
                        High priority, just target_mpp has a higher priority,
                        overwrites downsample and level if provided. (default: None)
  --level LEVEL         The tile level for sampling, alternative to downsample. Lowest priority, gets overwritten
                        by target_mag and downsample if they are provided. (default: None)
  --context_scales [CONTEXT_SCALES ...]
                        Define context scales for context patches. Context patches are centered around a central
                        patch. The context-patch size is equal to the patch-size, but downsampling is different
                        (default: None)
  --check_resolution CHECK_RESOLUTION
                        If a float value is supplies, the program checks whether the resolution of all images
                        corresponds to the given value (default: None)
  --processes PROCESSES
                        The number of processes to use. (default: None)
  --overwrite           Overwrite the patches that have already been created in case they already exist. Removes
                        dataset. Handle with care! (default: None)
  --annotation_paths ANNOTATION_PATHS
                        Path to the subfolder where the XML/JSON annotations are stored or path to a file
                        (default: None)
  --annotation_extension {json,xml}
                        The extension types used for the annotation files, the options are: ['json', 'xml']
                        (default: None)
  --incomplete_annotations
                        Set to allow WSI without annotation file (default: None)
  --label_map_file LABEL_MAP_FILE
                        The path to a json file that contains the mapping between the annotation labels and some
                        integers; an example can be found in examples (default: None)
  --save_only_annotated_patches
                        If true only patches containing annotations will be stored (default: None)
  --exclude_classes EXCLUDE_CLASSES
                        Can be used to exclude annotation classes (default: None)
  --store_masks         Set to store masks per patch. Defaults to false (default: None)
  --overlapping_labels  Per default, labels (annotations) are mutually exclusive. If labels overlap, they are
                        overwritten according to the label_map.json ordering (highest number = highest priority
                        (default: None)
  --normalize_stains    Uses Macenko normalization on a portion of the whole slide image (default: None)
  --normalization_vector_json NORMALIZATION_VECTOR_JSON
                        The path to a JSON file where the normalization vectors are stored (default: None)
  --adjust_brightness   Normalize brightness in a batch by clipping to 90 percent. Not recommended, but kept for legacy reasons (default: None)
  --min_intersection_ratio MIN_INTERSECTION_RATIO
                        The minimum intersection between the tissue mask and the patch. Must be between 0 and 1. 0
                        means that all patches are extracted. (default: None)
  --tissue_annotation TISSUE_ANNOTATION
                        Can be used to name a polygon annotation to determine the tissue area. If a tissue
                        annotation is provided, no Otsu-thresholding is performed (default: None)
  --masked_otsu         Use annotation to mask the thumbnail before otsu-thresholding is used (default: None)
  --otsu_annotation OTSU_ANNOTATION
                        Can be used to name a polygon annotation to determine the area for masked otsu
                        thresholding. Seperate multiple labels with ' ' (whitespace) (default: None)
  --filter_patches FILTER_PATCHES
                        Post-extraction patch filtering to sort out artefacts, marker and other non-tissue patches with a DL model. Time consuming.
                        (default: False)
  --apply_prefilter APPLY_PREFILTER
                        Pre-extraction mask filtering to remove marker from mask before applying otsu
                        (default: False)
  --log_path LOG_PATH   Path where log files should be stored. Otherwise, log files are stored in the output
                        folder (default: None)
  --log_level {critical,error,warning,info,debug}
                        Set the logging level. Options are ['critical', 'error', 'warning', 'info', 'debug']
                        (default: None)
  --hardware_selection {cucim,openslide}
                        Select hardware device (just if available, otherwise always cucim). Defaults to cucim.)
  --wsi_properties WSI_PROPERTIES
                        Dictionary with manual WSI metadata, but just applies if metadata cannot be derived from OpenSlide (e.g., for .tiff files). Supported keys are slide_mpp and magnification
                        (default: None)
```

**Label-Map**:

An exemplary `label_map.json` file is shown below. It is important that the background label always has a 0 assigned as integer value

Example:
```json
{
    "Background": 0,
    "Tissue-Annotation": 1,
    "Tumor": 2,
    "Stroma": 3,
    "Necrosis": 4
}
```
**Precedence of Target-Magnification, Downsampling and Level**

Target_mpp has the highest priority. If all four are passed, always the target mpp is used for output. Level has the lowest priority.
Sorted by priority:

- Target microns per pixel: Overwrites all other selections
- Target magnification: Overwrites downsampling and level
- Downsampling: Overwrites level
- Level: Lowest priority, default used when neither target magnification nor downsampling is passed


### CLI

A CLI is used to start the preprocessing. The entry-point is the [main_extraction.py](preprocessing/main_extraction.py) file. In addition to the CLI, also a configuration file can be passed via
```bash
python3 main_extraction.py --config path/to/config.yaml
```
Exemplary configuration file: [patch_extraction.yaml](/configs/examples/preprocessing/patch_extraction/patch_extraction.yaml)

### WSI-Properties Dictionary:
Per default, WSI metadata are derived automatically using OpenSlide. The WSI property dict just applies if the metadata cannot be extracted automatically (e.g., .tiff files). The following keys are necessary:
```
wsi_properties:
  - slide_mpp: # Microns per pixel, e.g., 0.25
  - magnification # Objective power during scanning, e.g., 20 or 40
```
