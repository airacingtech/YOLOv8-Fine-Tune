# YOLOv8 Instance Segmentation Fine-Tuning
Fine-tuning pipeline for YOLOv8-seg using ultralytics.

## Install
1. Clone this repository.
2. Install CUDA Toolkit 12.1 (do not install the low level CUDA drivers as you will likely black screen your system and have to reinstall your graphics card drivers from the CLI).
3. Using conda, create a new environment by running the following in the ROOT directory of this repository - make sure to use `environment.yml` and NOT `OLD_env.yml`:
```bash
conda env create -f environment.yml
```
## SAM2 FineTuning:
`sam2_finetune.py` contains an end-to-end script for finetuning a model from sam2 labeled images. It is heavily based on `fine_tune.ipynb` and `sam2_format.ipynb`. Data is copied below for clarity.

This script performs the following:
1. Converts the SAM2 segmasks into a YOLO-compatible labels, where the first integer `2` is the COCO-dataset index for "car", and the following floats are normalized (x,y) coordinate pairs outlining the contour of the mask. 
```bash
2 0.9973958333333334 0.18992248062015504 0.9921875 0.998062015503876 ...
2 0.3333333333333333 0.5234375 0.39728682170542634 0.5234375 0.4050387596899225 ...
2 0.9947916666666666 0.998062015503876 0.9921875 0.9961240310077519 0.8932291666666666 ...
2 0.33527131782945735 0.5234375 0.39728682170542634 0.5234375 0.40310077519379844 ...
```
2. Creates new YOLO-compatible dataset
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
3. Finetunes YOLO model
4. Exports YOLO model as .onnx with a batch size of `1` and an img size of `1056` (or another image size, depending on the finetuned model size)
   
### Running SAM2 Finetuning
1. Create a directory that holds SAM2-labeled ROSbags of images and masks:
```bash
sam2_labeled_data/
    ├── bag1/
    │   ├── images/
    │   └── masks/
    ├── bag2/
    │   ├── images/
    │   └── masks/
    ├── bag3/
    │   ├── images/
    │   └── masks/
```
Masks are expected to be 2D-segmasks that match the dimensions of the corresponding image. They should be `.json` files in the form:
```json
{"1": [[0, 0, ..., 0],
       [0, 0, ..., 0],
              ...
       [1, 1, ..., 0]]}
```
Where the number represents the object id of the sam2 label - this is SEPARATE from the object label of `2` to represent car, object id represents the number of objects within that image.

2. Ensure the `data.yaml` contains the correct paths. all paths should be from the `src/` directory to the location of the desired `data/` directory. 

The `data.yaml` file should look like something this (make sure to alter paths to where the actual train, val, test folders are):
```yaml
train: ../../data/train/images
val: ../../data/valid/images
test: ../../data/test/images

nc: 3

names:
  0: person
  1: bicycle
  2: car
...
```

3. Run the `sam2_finetune.py` script while inside the `src/` directory. The script requires the packages in the conda env. `sam2_finetune.py` has the following cli arguments:

| Argument            | Type    | Required | Description                                                                 |
|---------------------|---------|----------|-----------------------------------------------------------------------------|
| `dataset_dir`       | `str`   | ✅ Yes   | Path to the root directory of the SAM2 dataset (the `sam2_labeled_data` folder in the schematic above)                  |
| `data_yaml`         | `str`   | ✅ Yes   | Path to the (manually created) `data.yaml` file that defines dataset structure and classes.   |
| `data_dest_dir`     | `str`   | ✅ Yes   | Path to the destination directory where the formatted dataset `data/` will be saved. |
| `--resume`          | `flag`  | ❌ No    | If provided, training will resume from the checkpoint defined by `--resume_path`. |
| `--resume_path`     | `str`   | ❌ No    | Path to the YOLOv8 checkpoint to resume training from. Only used if `--resume` is set. |
| `--model_size`      | `str`   | ❌ No    | Defines the size variant of the YOLOv8 model to use. Options: `'n'`, `'s'`, `'m'`, `'l'`, `'x'`. Default is `'n'`. |

Example usage:
```bash
python3 sam2_finetune.py ../../sam2_labeled_data/ ../data.yaml ../../
```

Specifying model size:
```bash
python3 sam2_finetune.py ../../sam2_labeled_data/ ../data.yaml ../../ --model_size x
```


For resuming training:
```bash
python3 sam2_finetune.py ../../sam2_labeled_data/ ../data.yaml ../../ --resume --resume_path runs/segment/yolov8n-img_size_1032_layers_frozen_0_2025-04-07-01-33-24/weights/best.pt
```

## Data (SAM-1 Labeling)
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
train: ../../data/train/images
val: ../../data/valid/images
test: ../../data/test/images

nc: 3

names:
  0: person
  1: bicycle
  2: car
...
```

## Usage
### Parameters
In the first cell of `/src/fine_tune.py` change the parameters to fit your needs (e.g. `EPOCHS`, `IMG_SIZE`, etc.). Then run all the cells in the notebook to:
1. Fine-tune the YOLOv8n-seg model.
2. Perform a hyperparameter sweep / tune on the model.
3. Evaluate the model on the test set and save the results to a directory.
4. Export the model to the `models/` directory in the `ONNX` format.



