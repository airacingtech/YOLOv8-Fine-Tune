import os
import cv2
import json
import numpy as np
import ultralytics
from ultralytics import YOLO
import time
import torch
import random
import shutil
import tqdm
import argparse


# ========== DIRECTORIES ========== #

CURR_DIR = os.getcwd()
# When run from src/ as documented, CURR_DIR is .../YOLOv8-Fine-Tune/src
# and WORKSPACE_DIR resolves to .../YOLOv8-Fine-Tune (the repo root).
WORKSPACE_DIR = os.path.dirname(CURR_DIR)


# ========== YOLO PARAMS ========== #

# Change these parameters to fit your needs
EPOCHS = 50           # Reduced from 100; sufficient for fine-tuning from pretrained weights
IMG_SIZE = 1056       # Must be divisible by 32; matches ART camera resolution
LAYER_FREEZE = 10     # Freeze the first 10 layers (backbone) to preserve pretrained features

# Amount to use different data augmentations
HSV_H = 0.1
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 10.0        # Reduced from 45.0: extreme rotation produces geometrically invalid samples,
                      # especially compounding barrel distortion on fisheye frames.
TRANSLATE = 0.3
SCALE = 0.5
SHEAR = 0.01
PERSPECTIVE = 0.0     # Disabled: perspective warp applied on top of fisheye distortion is nonphysical.
FLIPUD = 0.0          # Cars don't appear upside down.
FLIPLR = 0.5          # Symmetric left/right track views — 50% horizontal flip is appropriate.
BGR = 0.0             # Disabled: channel-order inversion teaches tolerance for color-space bugs
                      # that are not expected in the controlled ART deployment pipeline.
MIXUP = 0.0           # Disabled: blending two images creates ghostly car overlaps that are not
                      # physically plausible. Ultralytics default is 0.0.
COPY_PASTE = 0.1      # Reduced from 0.4: cars are position-constrained (track surface only);
                      # pasting at high probability places them in physically impossible locations.
ERASING = 0.2
CROP_FRACTION = 0.1

# NOTE: MOSAIC is silently disabled by Ultralytics when rect=True (used below for efficient
# landscape batching). rect=True requires uniform aspect-ratio batching, which is incompatible
# with mosaic's 2x2 square-tile grid. MOSAIC and CLOSE_MOSAIC are therefore not passed to
# model.train() and have no effect on training.

LR0 = 0.001           # Lower initial LR for fine-tuning from pretrained weights
LRF = 0.01            # Final LR factor
WARMUP_EPOCHS = 3     # Gradual warmup to avoid destabilizing pretrained features early

# Dictionary to weight dataset (randomly removed images with given probability
# or duplicate images).
# WARNING: Any bag whose directory name contains a key substring will be
# silently downsampled or duplicated. For example, 'large': 0.1 drops 90% of
# frames from any bag named e.g. "large_oval_track". Update or clear this dict
# whenever you add new bags to avoid silent data loss.
DATASET_WEIGHTS = {
    'large': 0.1  # Drop 90% of frames from bags whose name contains 'large'
}

# Whether or not to use hyperparameter tuning
HYPERPARAMETER_TUNING = False
# Whether or not to use ray tune for hyperparameter sweep / tuning
USE_RAY_TUNE = False
# Number of iterations for hyperparameter sweep / tuning
TUNE_ITERS = 5

# ========== TRAINING PARAMS ========== #

# Percentage of dataset to use for training
TRAIN_PERCENTAGE = 1.0

# Whether or not to keep empty frames (frames with no labels) in the dataset
KEEP_EMPTY_FRAMES = True
# Percentage of empty frames to keep in the dataset if KEEP_EMPTY_FRAMES is True (randomly sampled)
PERCENTAGE_EMPTY_FRAMES_TO_KEEP = 0.8

# This line prevents the Kernel from crashing when running model.train() which calls a plotting function
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Batch size for TensorRT / ONNX export
ONNX_BATCH_SIZE = 1

# Timestamp used in run and export naming
DATE = time.strftime('%Y-%m-%d-%H-%M-%S')


# ========== SAM2 DATASET PROCESSING ========== #

# Converts a binary mask (2D array of 0s and 1s) to normalized YOLO polygon(s).
# Supports multiple detections per frame — each contour becomes a separate polygon entry.
def mask_to_polygon(mask):
    img_width = mask.shape[1]
    img_height = mask.shape[0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if len(contours) == 0:
        return []
    ret_contours = []
    for raw_contour in contours:
        if len(raw_contour) == 0:
            continue
        # Simplify contour to reduce vertex count while preserving car shape.
        # epsilon = 0.1% of perimeter keeps the outline tight without hundreds of redundant
        # points, which would slow YOLO's label parser and waste disk space.
        epsilon = 0.001 * cv2.arcLength(raw_contour, True)
        simplified = cv2.approxPolyDP(raw_contour, epsilon, True)
        contour = np.squeeze(simplified, axis=1)
        if len(contour) < 3:
            continue
        contour = np.array(contour, dtype=float)
        contour[:, 0] = contour[:, 0] / img_width
        contour[:, 1] = contour[:, 1] / img_height
        ret_contours.append(contour)
    return ret_contours

# to ensure the labels have the same format as the images, we map the jsons to the images
def create_json_to_img_mapping(images_dir):
    mapping = {}
    for img_name in os.listdir(images_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            name_wo_ext = os.path.splitext(img_name)[0]
            if name_wo_ext.startswith('frame_'):
                # frame_000007.PNG → 7.json
                frame_num = str(int(name_wo_ext.split('_')[1]))
                mapping[frame_num + '.json'] = img_name
            else:
                # 00005.jpg → 00005.json
                frame_num = str(int(name_wo_ext.split('.')[0]))
                mapping[frame_num + '.json'] = img_name
    return mapping

# Creates directory of labels from the masks, retaining naming format of the images
def format_sam2_labels_dir(images_dir, masks_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # creates mapping so we can just iterate over the masks
    json_to_img = create_json_to_img_mapping(images_dir)

    for filename in sorted(os.listdir(masks_dir)):
        if not filename.endswith('.json'):
            continue
        if filename not in json_to_img:
            print(f"Warning: No matching image for mask {filename}")
            continue

        image_name = json_to_img[filename]
        label_filename = os.path.splitext(image_name)[0] + '.txt'
        dest_filepath = os.path.join(dest_dir, label_filename)

        with open(os.path.join(masks_dir, filename), "r") as file:
            data = json.load(file)

        with open(dest_filepath, "w") as file:
            string_to_write = ""
            for key in data.keys():
                mask_array = np.array(data[key], dtype=np.uint8)
                polygons = mask_to_polygon(mask_array)
                for polygon in polygons:
                    if len(polygon) < 3:
                        continue
                    string_to_write += "0"  # class 0 = car (single-class)
                    for point in polygon:
                        string_to_write += f" {point[0]} {point[1]}"
                    string_to_write += "\n"
            file.write(string_to_write)

# given a directory of bags, create labels from the masks
def format_sam2_labels(datasets_dir):
    # get all the bags in datasets_dir
    bags = os.listdir(datasets_dir)
    # now process all
    for bag in bags:
        bag_dir = os.path.join(datasets_dir, bag)
        # verify its a directory
        if os.path.isdir(bag_dir):
            print(f"Processing bag: {bag}")
            dataset_dir = os.path.join(bag_dir, 'masks')
            images_dir = os.path.join(bag_dir, 'images')
            dest_dir = os.path.join(bag_dir, 'labels')
            format_sam2_labels_dir(images_dir, dataset_dir, dest_dir)
        else:
            print(f"Skipping {bag} as it is not a directory")


# ========== DATASET PROCESSING FOR YOLO ========== #

def generate_empty_label(label_dst : os.PathLike) -> None:
    '''
    Generates an empty label file with the correct format to handle empty frames.
    '''
    with open(label_dst, 'w') as f:
        f.write("")

def choose_dataset_weight(img_src : os.PathLike, dataset_weight : dict, weighted_frames : dict, frames_removed : dict) -> int:
    '''
    Returns a dataset weight to use dataset_weights dictionary. Uses the file path to determine the dataset.
    '''
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(img_src)))
    for dataset, weight in dataset_weight.items():
        if dataset in dataset_name:
            # If weight is greater than 1, keep the frame and create additional frames with the same image and label
            if weight > 1:
                weighted_frames[dataset] = weighted_frames.get(dataset, 0) + (weight - 1)
                return weight
            # If weight is less than 1, randomly choose to keep the frame or not
            else:
                # Keep the frame with probability weight
                if random.random() < weight:
                    return 1
                # Discard the frame with probability 1 - weight
                else:
                    frames_removed[dataset] = frames_removed.get(dataset, 0) + 1
                    return 0
    return 1

def copy_data_yaml(label_src : os.PathLike, img_src : os.PathLike, label_dst : os.PathLike, img_dst : os.PathLike, empty_frames_kept : list, weighted_frames : dict, frames_removed : dict) -> None:
    '''
    Copies the images and labels from one directory to another. Also handles the case where the label file is empty (i.e. does not exist).
    '''
    if not os.path.exists(label_src):
        if KEEP_EMPTY_FRAMES:
            if random.random() < PERCENTAGE_EMPTY_FRAMES_TO_KEEP:
                empty_frames_kept[0] += 1
                for i in range(choose_dataset_weight(img_src, DATASET_WEIGHTS, weighted_frames, frames_removed)):
                    stem = os.path.splitext(img_dst)[0]
                    shutil.copy(img_src, stem + f"_{i}.jpg")
                    generate_empty_label(stem + f"_{i}.txt")
    else:
        # Normalize every label row to class id 0 (single-class car) in memory.
        # We never rewrite the source file — that would silently mutate the
        # user's labeled data in place across runs.
        with open(label_src, 'r') as f_in:
            lines = f_in.readlines()
        normalized = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            parts[0] = '0'
            normalized.append(' '.join(parts) + '\n')

        for i in range(choose_dataset_weight(img_src, DATASET_WEIGHTS, weighted_frames, frames_removed)):
            stem = os.path.splitext(img_dst)[0]
            shutil.copy(img_src, stem + f"_{i}.jpg")
            with open(stem + f"_{i}.txt", 'w') as f_out:
                f_out.writelines(normalized)

def format_datasets(datasets_path: os.PathLike, data_dest_dir: os.PathLike) -> None:
    """
    Takes in a directory of bag datasets (each bag contains images/ and labels/
    subdirectories) and formats them into a single dataset that YOLOv8 can use for
    training, placed in <data_dest_dir>/data/.

    This function deletes and recreates <data_dest_dir>/data/ on each run.
    A data.yaml is auto-generated with nc=1 and class name 'car'.

    IMPORTANT: The dataset is split at the BAG level (not frame level). Adjacent
    frames within a ROS bag are temporally near-identical (same lighting, same car
    positions, often < 100 ms apart). A per-frame shuffle would allow near-duplicate
    pairs across train and val, inflating val mAP by 5-15 points without reflecting
    real generalization. Bag-level splitting ensures the model is evaluated on footage
    it has never seen at all.
    """
    assert os.path.exists(datasets_path), f"The dataset path, {datasets_path}, does not exist."

    print("Formatting dataset from:", datasets_path)

    # Group frames by bag. Skip non-directory entries (.DS_Store, stray .yaml files, etc.)
    # and bags that don't have an images/ subdirectory.
    bags = {}
    for dataset_path in sorted(os.listdir(datasets_path)):
        full_path = os.path.join(datasets_path, dataset_path)
        if not os.path.isdir(full_path):
            continue
        img_dir = os.path.join(full_path, "images")
        if not os.path.isdir(img_dir):
            print(f"Skipping {dataset_path}: no images/ subdirectory found")
            continue
        bags[dataset_path] = [
            [os.path.join(full_path, "images", img_file),
             os.path.join(full_path, "labels", os.path.splitext(img_file)[0] + ".txt"),
             dataset_path]
            for img_file in sorted(os.listdir(img_dir))
        ]

    # Split at the BAG level to avoid temporal leakage between train / val / test.
    bag_names = sorted(bags.keys())
    random.seed(0)
    random.shuffle(bag_names)

    n = len(bag_names)
    train_bags = bag_names[:n * 7 // 10]
    val_bags   = bag_names[n * 7 // 10 : n * 9 // 10]
    test_bags  = bag_names[n * 9 // 10 :]

    # Guarantee at least 1 bag in val and test even with very small datasets.
    if not val_bags and len(train_bags) > 2:
        val_bags = [train_bags.pop()]
    if not test_bags and len(train_bags) > 1:
        test_bags = [train_bags.pop()]

    print(f"Train bags ({len(train_bags)}): {train_bags}")
    print(f"Val bags   ({len(val_bags)}):   {val_bags}")
    print(f"Test bags  ({len(test_bags)}):  {test_bags}")

    # Flatten bags into frame lists, then shuffle within each split.
    training_data = [f for b in train_bags for f in bags[b]]
    valid_data    = [f for b in val_bags   for f in bags[b]]
    test_data     = [f for b in test_bags  for f in bags[b]]

    random.shuffle(training_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)

    # Create the directories for the training, validation, and test data.
    data_root = os.path.join(data_dest_dir, "data")
    if os.path.exists(data_root):
        print("Deleting and recreating 'data/' folder...")
        shutil.rmtree(data_root)
    os.makedirs(os.path.join(data_root, "train", "images"))
    os.makedirs(os.path.join(data_root, "train", "labels"))
    os.makedirs(os.path.join(data_root, "valid", "images"))
    os.makedirs(os.path.join(data_root, "valid", "labels"))
    os.makedirs(os.path.join(data_root, "test", "images"))
    os.makedirs(os.path.join(data_root, "test", "labels"))

    # Auto-generate data.yaml for single-class car detection.
    # Paths are relative to data/ (Ultralytics resolves them against the yaml location).
    data_yaml_path = os.path.join(data_root, "data.yaml")
    with open(data_yaml_path, 'w') as f:
        f.write("train: train/images\n")
        f.write("val:   valid/images\n")
        f.write("test:  test/images\n\n")
        f.write("nc: 1\n\n")
        f.write("names:\n")
        f.write("  0: car\n")
    print(f"Generated {data_yaml_path}")

    # Copy over images and labels to new directories.
    print("Copying images and labels to new directories...")
    new_image_uuid = 0
    empty_frames_kept = [0]
    weighted_frames = {}
    removed_frames = {}
    train_frames = 0
    valid_frames = 0
    test_frames = 0

    print("Copying training data:")
    for img_src, label_src, dataset_path in tqdm.tqdm(training_data):
        if (random.random() < TRAIN_PERCENTAGE):
            stem = os.path.splitext(os.path.basename(img_src))[0]
            fname = f"{dataset_path}_{stem}_{new_image_uuid}"
            img_dst   = os.path.join(data_root, "train", "images", fname + ".jpg")
            label_dst = os.path.join(data_root, "train", "labels", fname + ".txt")
            copy_data_yaml(label_src, img_src, label_dst, img_dst, empty_frames_kept, weighted_frames, removed_frames)
            new_image_uuid += 1
            train_frames += 1
    for img_src, label_src, dataset_path in tqdm.tqdm(valid_data):
        stem = os.path.splitext(os.path.basename(img_src))[0]
        fname = f"{dataset_path}_{stem}_{new_image_uuid}"
        img_dst   = os.path.join(data_root, "valid", "images", fname + ".jpg")
        label_dst = os.path.join(data_root, "valid", "labels", fname + ".txt")
        copy_data_yaml(label_src, img_src, label_dst, img_dst, empty_frames_kept, weighted_frames, removed_frames)
        new_image_uuid += 1
        valid_frames += 1
    for img_src, label_src, dataset_path in tqdm.tqdm(test_data):
        stem = os.path.splitext(os.path.basename(img_src))[0]
        fname = f"{dataset_path}_{stem}_{new_image_uuid}"
        img_dst   = os.path.join(data_root, "test", "images", fname + ".jpg")
        label_dst = os.path.join(data_root, "test", "labels", fname + ".txt")
        copy_data_yaml(label_src, img_src, label_dst, img_dst, empty_frames_kept, weighted_frames, removed_frames)
        new_image_uuid += 1
        test_frames += 1

    print("Number of empty frames kept: ", empty_frames_kept[0])
    print("Number of training frames: ", train_frames)
    print("Number of validation frames: ", valid_frames)
    print("Number of test frames: ", test_frames)
    print("Additional weighted frames created for each dataset: ", weighted_frames)
    print("Number of frames removed for each dataset: ", removed_frames)
    print("Finished creating directories for YOLOv8 training pipeline")

# ========== TRAINING YOLOv8 ========== #
def choose_model_size(model_size) -> str:
    sizes = {
        'n': ('Nano',        'yolov8n-seg.pt'),
        's': ('Small',       'yolov8s-seg.pt'),
        'm': ('Medium',      'yolov8m-seg.pt'),
        'l': ('Large',       'yolov8l-seg.pt'),
        'x': ('Extra Large', 'yolov8x-seg.pt'),
    }
    if model_size not in sizes:
        raise ValueError(f"Invalid model_size '{model_size}'. Must be one of: {list(sizes)}")
    name, weights = sizes[model_size]
    print(f"Using YOLOv8 {name} model")
    return weights

def train_model(model : YOLO, curr_data_yaml, model_size) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    start_time = time.time()
    model_name = f'yolov8{model_size}-img_size_{IMG_SIZE}_layers_frozen_{LAYER_FREEZE}_{DATE}'

    # By default, the model trains on a single GPU.
    # rect=True enables efficient landscape batching by grouping images with similar
    # aspect ratios. NOTE: Ultralytics silently disables mosaic when rect=True because
    # mosaic requires square 2x2 tiling — so mosaic and close_mosaic are not passed here.
    model.train(
        data=curr_data_yaml,
        imgsz=IMG_SIZE,
        rect=True,
        epochs=EPOCHS,
        freeze=LAYER_FREEZE,
        amp=True,
        cache="disk",
        save=True,
        save_period=5,
        name=model_name,
        seed=0,

        lr0=LR0,
        lrf=LRF,
        warmup_epochs=WARMUP_EPOCHS,

        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        bgr=BGR,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        erasing=ERASING,
        crop_fraction=CROP_FRACTION,
        workers=4,
        batch=8
    )

    end_time = time.time()
    training_time = end_time - start_time
    print("Time to train: ", training_time)

def tune_model(model : YOLO) -> None:
    # Runs a hyperparameter sweep and selects the best hyperparameters
    if HYPERPARAMETER_TUNING:
        model.tune(use_ray=USE_RAY_TUNE, iterations=TUNE_ITERS)
    else:
        print("Skipping hyperparameter tuning")


def test_model(model : YOLO, test_results_path: os.PathLike, test_images_path: os.PathLike) -> None:
    """
    Test the fine-tuned model on test images and save the results.
    """
    # Make sure the test save path exists
    if not os.path.exists(test_results_path):
        os.makedirs(test_results_path)

    # Inference fine-tuned model on test images and save results. Skip any
    # non-image entries (.DS_Store, labels.cache, etc.) so the loop doesn't crash.
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    for file in os.listdir(test_images_path):
        if not file.lower().endswith(image_exts):
            continue
        file_path = os.path.join(test_images_path, file)
        output = model.predict(file_path)
        save_path = os.path.join(test_results_path, file)
        cv2.imwrite(save_path, output[0].plot())

    print("Inference on test set complete. Results saved to: ", test_results_path)


# ========== MAIN FUNCTION ========== #
def main():
    # arg parse
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 model on SAM2 dataset.")
    parser.add_argument("dataset_dir", type=str, help="Path to SAM2 dataset directory")
    parser.add_argument("data_dest_dir", type=str, help="Path to destination directory for formatted dataset")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--model_size", type=str, default='m', choices=['n', 's', 'm', 'l', 'x'], help="Size of YOLOv8 model to use")
    parser.add_argument("--format_only", action="store_true", default=False, help="Only format the dataset without training")
    parser.add_argument("--finetune_only", action="store_true", default=False, help="Skip SAM2->label conversion and dataset formatting; train directly on existing data/")

    args = parser.parse_args()

    FORMAT_ONLY = args.format_only
    FINETUNE_ONLY = args.finetune_only

    if FORMAT_ONLY and FINETUNE_ONLY:
        print("ERROR: --format_only and --finetune_only are mutually exclusive.")
        return

    DATASETS_DIR = args.dataset_dir
    # Normalize trailing slash so path concatenation is always correct regardless
    # of whether the user passes 'path/to/dir' or 'path/to/dir/'.
    DATA_DEST_DIR = args.data_dest_dir.rstrip('/') + '/'

    data_dir = os.path.join(DATA_DEST_DIR, "data")
    curr_data_yaml = os.path.join(data_dir, "data.yaml")
    TEST_PATH = os.path.join(data_dir, "test", "images")

    RESUME_TRAINING = args.resume
    RESUME_TRAINING_PATH = args.resume_path
    MODEL_SIZE = args.model_size

    if not FINETUNE_ONLY:
        # Convert SAM2 masks to YOLO polygon labels, then build the train/val/test split.
        format_sam2_labels(DATASETS_DIR)
        format_datasets(DATASETS_DIR, DATA_DEST_DIR)
    else:
        # --finetune_only: skip conversion and dataset rebuild entirely.
        # data/ must already exist from a previous run of format_datasets.
        if not os.path.exists(data_dir):
            print(f"ERROR: --finetune_only was set but no formatted dataset found at {data_dir}. "
                  f"Run without --finetune_only first to build the dataset.")
            return
        print(f"--finetune_only: using existing dataset at {data_dir}")

    if FORMAT_ONLY:
        print("Dataset formatted. Exiting...")
        return

    print("CUDA Available: " + str(torch.cuda.is_available()))
    print("Torch CUDA Version: " + str(torch.version.cuda))
    ultralytics.utils.checks.collect_system_info()

    # Load YOLOv8 segmentation model
    if RESUME_TRAINING:
        if RESUME_TRAINING_PATH is None:
            print("ERROR: --resume requires --resume_path to be set.")
            return
        model = YOLO(RESUME_TRAINING_PATH)
    else:
        model = YOLO(choose_model_size(MODEL_SIZE))

    train_model(model, curr_data_yaml, MODEL_SIZE)
    tune_model(model)
    test_model(model, os.path.join(data_dir, "test", f"annotation_results_{EPOCHS}epochs"), TEST_PATH)

    # model.train() saves best.pt and last.pt under runs/segment/<run_name>/weights/.
    # Use best.pt from that directory as the canonical trained checkpoint.
    model.export(format='onnx', batch=ONNX_BATCH_SIZE, imgsz=IMG_SIZE, dynamic=False)

if __name__ == "__main__":
    main()