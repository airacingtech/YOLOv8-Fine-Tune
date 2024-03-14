# YOLOv8 Instance Segmentation Fine-Tuning
Fine-tuning pipeline for YOLOv8-seg using ultralytics.

## Install
1. Clone this repository.
2. Install CUDA Toolkit 12.1 (do not install the low level CUDA drivers as you will likely black screen your system and have to reinstall your graphics card drivers from the CLI).
3. Using conda, create a new environment by running the following in the ROOT directory of this repository:
```bash
conda env create -f environment.yml
```

## Data
All your data must be combined into one unified directory following the YOLOv8 segmentation format. The directory structure should look like this:
```bash
data/
    ├── test/
    │   ├── images/
    │   └── labels/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    data.yaml
```
where the `images/` directories contain images with each image having a corresponding txt to represent the segmentation mask in the `labels/` directory. An example of an `image.txt` file is shown below:
```bash
0 0 0.9973958333333334 0.18992248062015504 0.9921875 0.998062015503876 ...
1 0.3333333333333333 0.5234375 0.39728682170542634 0.5234375 0.4050387596899225 ...
0 0 0.9947916666666666 0.998062015503876 0.9921875 0.9961240310077519 0.8932291666666666 ...
1 0.33527131782945735 0.5234375 0.39728682170542634 0.5234375 0.40310077519379844 ...
```
where each line repsents a different segmentation instance. The first number in the line is the class label and the rest of the numbers define the contour of the segmentation mask.

The `data.yaml` file should look like this:
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['drivable-area', 'opponent-car'] # class names
...
```

## Usage
### Parameters
In the first cell of `/src/fine_tune.py` change the parameters to fit your needs (e.g. `EPOCHS`, `IMG_SIZE`, etc.). Then run all the cells in the notebook to:
1. Fine-tune the YOLOv8n-seg model.
2. Perform a hyperparameter sweep / tune on the model.
3. Evaluate the model on the test set and save the results to a directory.
4. Export the model to the `models/` directory in the `ONNX` format.
