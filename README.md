# Amodal Panoptic Segmentation Evaluation

This repository contains scripts for evaluating amodal panoptic segmentation task results. It provides functions to compute amodal panoptic quality metrics: **Amodal Panoptic Quality (APQ)** and **Amodal Parsing Coverage (APC)** as presented in the [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Mohan_Amodal_Panoptic_Segmentation_CVPR_2022_paper.pdf).

## Prerequisites

- Python 3.x
- Dependencies listed in `requirements.txt`


## Installation

1. Clone the repository:

    ```shell
    git clone https://github.com/robot-learning-freiburg/amodal-panoptic.git
    ```

2. Install

    ```shell
    cd amodal_panoptic_eval
    pip install .
    ```

## Dataset Format

This section describes the expected data format for the evaluation script.

### PNG Encoding

The dataset employs 1-channel uint16 PNG files (*_ampano.png) to encode visible regions of objects and stuff in the scene. The encoding scheme is detailed below:

- **Stuff Classes**: Pixels corresponding to "stuff" classes are encoded directly with their `semantic_class_id`, uniquely identifying the stuff class of each pixel.

- **Thing Classes**: Pixels associated with "thing" classes are encoded with `semantic_class_id*1000 + instance_id`. 

### JSON Metadata

Each PNG file is accompanied by a corresponding JSON file (`*_ampano.json`). The JSON file is structured as a dictionary, where each key represents a unique object identified by `semantic_class_id*1000 + instance_id`, and contains the following information:

- **occlusion_mask**: An optional occlusion mask for the object, encoded in Run-Length Encoding (RLE). This mask indicates parts of the object occluded by others.

- **amodal_mask**: The amodal mask for the object, also in RLE. It represents the complete shape of the object, including both visible and occluded regions.

Note: Semantic class ids refer to 'id' and not 'trainId'.

## Usage

To evaluate amodal panoptic segmentation results, follow these steps:

1. Prepare the ground truth and prediction folders:
    - Ground Truth: Place it in a folder named **amodal_panoptic_seg**, located within any base directory. While subfolders under amodal_panoptic_seg can vary, each must include *_ampano.png and *_ampano.json files for amodal panoptic segmentation ground truth.
    - Prediction: The prediction folder must also contain a subfolder named **amodal_panoptic_seg**, mirroring the corresponding folder structure and including prediction files.

For example:
```
    /gt_directory
        /amodal_panoptic_seg
            /subfolder1
                image11_ampano.png
                image11_ampano.json
                image12_ampano.png
                image12_ampano.json
                ...

            /subfolder2
                image21_ampano.png
                image21_ampano.json
                image22_ampano.png
                image22_ampano.json
                ...

    /prediction_directory
        /amodal_panoptic_seg
            /subfolder1
                image11_ampano.png
                image11_ampano.json
                image12_ampano.png
                image12_ampano.json
                ...

            /subfolder2
                image21_ampano.png
                image21_ampano.json
                image22_ampano.png
                image22_ampano.json
                ...
```


2. Run the evaluation script:

    ```shell
    python -m amodal_panoptic_eval.eval --gt-folder /path/to/ground_truth/amodal_panoptic_seg --prediction-folder /path/to/predictions/amodal_panoptic_seg
    ```

    Optional arguments:
    - `--results-file`: File to store computed panoptic quality. Default: `resultAmodalPanoptic.json`
    - `--dataset-name`: Name of the dataset. Default: `amodalSynthDrive`
    - `--cpu-num`: Number of CPU cores to use for evaluation. Default: -1 (use all available cores)

3. The evaluation results will be printed to the console and saved in the specified results file.

## License

This project is licensed under the GNU General Public License v3 (GPLv3) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We have used utility functions from other open-source projects. We especially thank the authors of [Cityscapes](https://github.com/mcordts/cityscapesScripts).







