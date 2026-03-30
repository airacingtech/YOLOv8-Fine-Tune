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
# this should just be the path to the yolo-finetune repo
WORKSPACE_DIR = os.path.dirname(CURR_DIR)


# ========== YOLO PARAMS ========== #

# Change these parameters to fit your needs
EPOCHS = 50           # CHANGED: REDUCED FROM 100 FOR FINE-TUNING
NUM_TRAIN_LOOPS = 1
IMG_SIZE = 1056       # CHANGED: SET TO 1056 TO BE DIVISIBLE BY 32
LAYER_FREEZE = 10     # CHANGED: INCREASED FROM 0 TO 10 TO FREEZE THE BACKBONE

# Amount to use different data augmentations
HSV_H = 0.1           # RESTORED: KEPT YOUR ORIGINAL 0.1 VALUE
HSV_S = 0.7           # RESTORED: KEPT YOUR ORIGINAL 0.7 VALUE
HSV_V = 0.4           # RESTORED: KEPT YOUR ORIGINAL 0.4 VALUE
DEGREES = 45.0        # CHANGED: INCREASED FROM 0.4 TO ROTATE IMAGE RANDOMLY UP TO 45 DEGREES
TRANSLATE = 0.3
SCALE = 0.5
SHEAR = 0.01
PERSPECTIVE = 0.001
FLIPUD = 0.0          # CHANGED: REDUCED FROM 0.3 TO 0.0 BECAUSE CARS DON'T APPEAR UPSIDE DOWN
FLIPLR = 0.5          # CHANGED: INCREASED FROM 0.3 TO 0.5 FOR 50% CHANCE OF HORIZONTAL FLIP
BGR = 0.1             # Flips channels from RGB to BGR
MOSAIC = 0.5          
MIXUP = 0.5
COPY_PASTE = 0.4
ERASING = 0.2
CROP_FRACTION = 0.1

# CHANGED: ADDED THE FOLLOWING 4 LEARNING RATE PARAMETERS FOR FINE-TUNING
LR0 = 0.001           # CHANGED: ADDED LOWER INITIAL LEARNING RATE
LRF = 0.01            # CHANGED: ADDED FINAL LEARNING RATE FACTOR
WARMUP_EPOCHS = 3     # CHANGED: ADDED WARMUP EPOCHS
CLOSE_MOSAIC = 10     # CHANGED: ADDED TO TURN OFF MOSAIC FOR FINAL 10 EPOCHS

# Dictionary to weight dataset (randomly removed images with given probability
# or duplicate images)
DATASET_WEIGHTS = {
    'large': 0.1 # Remove extra images from dataset with no frameskipping
}

# Whether or not to use hyperparameter tuning
HYPERPARAMETER_TUNING = False
# Whether or not to use ray tune for hyperparameter sweep / tuning
USE_RAY_TUNE = False
# Number of iterations for hyperparameter sweep / tuning
TUNE_ITERS = 5

# ========== TRAINING PARAMS ========== #

# Percetnage of dataset to use for training
TRAIN_PERCENTAGE = 1.0

# Whether or not to keep empty frames (frames with no labels) in the dataset
KEEP_EMPTY_FRAMES = True
# Percentage of empty frames to keep in the dataset if KEEP_EMPTY_FRAMES is True (randomly sampled)
PERCENTAGE_EMPTY_FRAMES_TO_KEEP = 0.8

# Path to trained model weights
MODELS_PATH = WORKSPACE_DIR + '/models/'

# This line prevents the Kernel from crashing when running model.train() which calls a plotting function
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# This is the Tensorrt batch size 
ONNX_BATCH_SIZE = 1

# Todays data + Batch size + epochs
DATE = time.strftime('%Y-%m-%d-%H-%M-%S')


# ========== SAM2 DATASET PROCESSING ========== #

# function that takes a mask of 1s and 0s and converts to polygon using cv2
# NOTE: this should support multiple detections in the same frame, but hasn't been tested

def mask_to_polygon(mask):
    img_width = mask.shape[1]
    img_height = mask.shape[0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if len(contours) == 0:
        return []
    contours = [contour for contour in contours if len(contour) > 0]
    contours = [np.squeeze(contour, axis=1) for contour in contours]
    ret_contours = []
    for contour in contours:
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
                    string_to_write += "0" # CHANGED: REPLACED "2" WITH "0" SO MASKS MAP TO CLASS 0
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
    dataset_name = img_src.split("/")[-3]
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

def copy_data_yaml(label_src : os.PathLike, img_src : os.PathLike, label_dst : os.PathLike, img_dst : os.PathLike, empty_frames_kept : int, weighted_frames : dict, frames_removed : dict) -> None:
    '''
    Copies the images and labels from one directory to another. Also handles the case where the label file is empty (i.e. does not exist).
    '''
    if not os.path.exists(label_src):
        if KEEP_EMPTY_FRAMES:
            if random.random() < PERCENTAGE_EMPTY_FRAMES_TO_KEEP:
                empty_frames_kept += 1
                for i in range(choose_dataset_weight(img_src, DATASET_WEIGHTS, weighted_frames, frames_removed)):
                    img_dst_with_weight = img_dst[:-4] + "_" + str(i) + ".jpg"
                    label_dst_with_weight = label_dst[:-4] + "_" + str(i) + ".txt"
                    shutil.copy(img_src, img_dst_with_weight)
                    generate_empty_label(label_dst_with_weight)
    else:
        for i in range(choose_dataset_weight(img_src, DATASET_WEIGHTS, weighted_frames, frames_removed)):
            img_dst_with_weight = img_dst[:-4] + "_" + str(i) + ".jpg"
            label_dst_with_weight = label_dst[:-4] + "_" + str(i) + ".txt"
            shutil.copy(img_src, img_dst_with_weight)
            shutil.copy(label_src, label_dst_with_weight)

def format_datasets(datasets_path : os.PathLike, data_yaml : os.PathLike, data_dest_dir : os.PathLike) -> None:
    """
    Takes in a directory of datasets where each dataset is in the format of a COCO dataset
    and then formats the datasets into a single dataset that YOLOv8 can use for training
    placed in the 'data/' directory. This function will delete the 'data/' directory and recreate
    it. The data.yaml file must be created manually and will be copied over to the 'data/'
    directory. The function will also split the data into training, validation, and test sets.
    """
    assert os.path.exists(datasets_path), f"The dataset path, {datasets_path}, does not exist."
    assert os.path.exists(data_yaml), f"The data.yaml file, {data_yaml}, does not exist."

    print("Extracting sim images from:", datasets_path, "for training/validation data")

    # Get all images and labels from the datasets
    datasets_labels = []
    for dataset_path in os.listdir(datasets_path):
        full_path = os.path.join(datasets_path, dataset_path)
        for img_file in os.listdir(full_path + "/images/"):
            datasets_labels.append([full_path + "/images/" + img_file, full_path + "/labels/" + img_file[:-4] + ".txt", dataset_path])

    # Sort images by filename
    datasets_labels = sorted(datasets_labels, key = lambda x: x[0])

    # Shuffle images deterministically with seed
    random.seed(0)
    random.shuffle(datasets_labels)

    # Split 70% training, 20% validation, 10% test
    training_data = datasets_labels[:len(datasets_labels) * 7 // 10]
    valid_data = datasets_labels[len(datasets_labels) * 7 // 10 :len(datasets_labels) * 9 // 10]
    test_data = datasets_labels[len(datasets_labels) * 9 // 10 :]

    # Create the directories for the training, validation, and test data
    if os.path.exists(data_dest_dir + "data/"):
        print("Deleting and recreating 'data/' folder...")
        shutil.rmtree(data_dest_dir + "data/")
    os.mkdir(data_dest_dir + "data/")
    os.mkdir(data_dest_dir + "data/train/")
    os.mkdir(data_dest_dir + "data/train/images/")
    os.mkdir(data_dest_dir + "data/train/labels/")
    os.mkdir(data_dest_dir + "data/valid/")
    os.mkdir(data_dest_dir + "data/valid/images/")
    os.mkdir(data_dest_dir + "data/valid/labels/")
    os.mkdir(data_dest_dir + "data/test/")
    os.mkdir(data_dest_dir + "data/test/images/")
    os.mkdir(data_dest_dir + "data/test/labels/")

    # Copy over images and labels to new directories
    print("Copying images and labels to new directories...")
    print("Copying training data:")

    # Create a new image number to avoid overwriting images in the same chance they have the same name
    new_image_uuid = 0
    empty_frames_kept = 0
    weighted_frames = {}
    removed_frames = {}
    train_frames = 0
    valid_frames = 0
    test_frames = 0
    for img_src, label_src, dataset_path in tqdm.tqdm(training_data):
        if (random.random() < TRAIN_PERCENTAGE):
            new_image_name = img_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".jpg"
            new_label_name = label_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".txt"
            img_dst = os.path.join(data_dest_dir + "data/train/images/", dataset_path + "_" + new_image_name)
            label_dst = os.path.join(data_dest_dir + "data/train/labels/", dataset_path + "_" + new_label_name)
            copy_data_yaml(label_src, img_src, label_dst, img_dst, empty_frames_kept, weighted_frames, removed_frames)
            new_image_uuid += 1
            train_frames += 1
    for img_src, label_src, dataset_path in tqdm.tqdm(valid_data):
        new_image_name = img_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".jpg"
        new_label_name = label_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".txt"
        img_dst = os.path.join(data_dest_dir + "data/valid/images/", dataset_path + "_" + new_image_name)
        label_dst = os.path.join(data_dest_dir + "data/valid/labels/", dataset_path + "_" + new_label_name)
        copy_data_yaml(label_src, img_src, label_dst, img_dst, empty_frames_kept, weighted_frames, removed_frames)
        new_image_uuid += 1
        valid_frames += 1
    for img_src, label_src, dataset_path in tqdm.tqdm(test_data):
        new_image_name = img_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".jpg"
        new_label_name = label_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".txt"
        img_dst = os.path.join(data_dest_dir + "data/test/images/", dataset_path + "_" + new_image_name)
        label_dst = os.path.join(data_dest_dir + "data/test/labels/", dataset_path + "_" + new_label_name)
        copy_data_yaml(label_src, img_src, label_dst, img_dst, empty_frames_kept, weighted_frames, removed_frames)
        new_image_uuid += 1
        test_frames += 1

    # Copy over data.yaml file from root directory
    shutil.copy(data_yaml, data_dest_dir + "data/")
    print("Copied over 'data.yaml' file")
    print("Number of empty frames kept: ", empty_frames_kept)

    print("Number of training frames: ", train_frames)
    print("Number of validation frames: ", valid_frames)
    print("Number of test frames: ", test_frames)
    print("Additional weighted frames created for each dataset: ", weighted_frames)
    print("Number of frames removed for each dataset: ", removed_frames)
    print("Finished creating directories for YOLOv8 training pipeline")

# ========== TRAINING YOLOv8 ========== #
def choose_model_size(model_size) -> str:
    if model_size == 'n':
        print("Using YOLOv8 Nano model")
        return 'yolov8n-seg.pt'
    elif model_size == 's':
        print("Using YOLOv8 Small model")
        return 'yolov8s-seg.pt'
    elif model_size == 'm':
        print("Using YOLOv8 Medium model")
        return 'yolov8m-seg.pt'
    elif model_size == 'l':
        print("Using YOLOv8 Large model")
        return 'yolov8l-seg.pt'
    elif model_size == 'x':
        print("Using YOLOv8 Extra Large model")
        return 'yolov8x-seg.pt'

def train_model(model : YOLO, curr_data_yaml, model_size) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    start_time = time.time()
    model_name = f'yolov8{model_size}-img_size_{IMG_SIZE}_layers_frozen_{LAYER_FREEZE}_{DATE}'
    
    # By default, the model trains on a single GPU
    model.train(
        data=curr_data_yaml,
        imgsz=IMG_SIZE,
        rect=True,               # CHANGED: ADDED FOR EFFICIENT LANDSCAPE BATCHING
        epochs=EPOCHS,
        freeze=LAYER_FREEZE,
        amp=True,
        cache="disk", 
        save=True,
        save_period=5,
        name=model_name,
        
        lr0=LR0,                 # CHANGED: ADDED LOWER INITIAL LEARNING RATE
        lrf=LRF,                 # CHANGED: ADDED FINAL LEARNING RATE FACTOR
        warmup_epochs=WARMUP_EPOCHS, # CHANGED: ADDED GRADUAL WARMUP
        close_mosaic=CLOSE_MOSAIC,   # CHANGED: ADDED TO TURN OFF MOSAIC AT THE END
        
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
        mosaic=MOSAIC,
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

    # Inferencee fine-tuned model on test images and save results
    for file in os.listdir(test_images_path):
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
    parser.add_argument("data_yaml", type=str, help="Path to data.yaml file")
    parser.add_argument("data_dest_dir", type=str, help="Path to destination directory for formatted dataset")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume training from")
    
    parser.add_argument("--model_size", type=str, default='m', choices=['n', 's', 'm', 'l', 'x'], help="Size of YOLOv8 model to use") # CHANGED: DEFAULT CHANGED TO 'm' AND ADDED CHOICES
    parser.add_argument("--format_only", action="store_true", default=False, help="Only format the dataset without training")
    parser.add_argument("--finetune_only", action="store_true", default=False, help="Only run the fine-tuning without performing sam2 data conversion")

    args = parser.parse_args()

    DATASETS_DIR = args.dataset_dir
    DATA_YAML = args.data_yaml
    DATA_DEST_DIR = args.data_dest_dir

    data_dir = DATA_DEST_DIR + 'data/'
    curr_data_yaml = data_dir + 'data.yaml'
    TEST_PATH = data_dir + '/test/images/'

    RESUME_TRAINING = args.resume
    RESUME_TRAINING_PATH = args.resume_path
    MODEL_SIZE = args.model_size
    FORMAT_ONLY = args.format_only
    FINETUNE_ONLY = args.finetune_only
    if not FINETUNE_ONLY:
        # convert sam2 masks to labels
        format_sam2_labels(DATASETS_DIR)

    # format dataset for yolov8
    format_datasets(DATASETS_DIR, DATA_YAML, DATA_DEST_DIR)
    if FORMAT_ONLY:
        print("Dataset formatted. Exiting...")
        return
    
    # check system info - could just comment these out
    print("CUDA Available: " + str(torch.cuda.is_available()))
    print("Torch CUDA Version: " + str(torch.version.cuda))
    ultralytics.utils.checks.collect_system_info()

    # Load yolov8 segmentation model
    if RESUME_TRAINING:
        if RESUME_TRAINING_PATH is None:
            print("Please provide a path to the checkpoint to resume training from.")
            return
        model = YOLO(RESUME_TRAINING_PATH)
    else:
        model = YOLO(choose_model_size(MODEL_SIZE))

    # Training loop
    epochs_done = 0
    for _ in range(NUM_TRAIN_LOOPS):
        print(f"Starting training loop starting on epoch {epochs_done}")
        train_model(model, curr_data_yaml, MODEL_SIZE)
        epochs_done += EPOCHS
        tune_model(model)
        test_results_path = data_dir + '/test/annotation_results' + f'_{epochs_done}epochs'
        test_model(model, test_results_path, TEST_PATH)
        
        # can change naming convention if need be
        model_name = f"yolov8{MODEL_SIZE}_{DATE}_batch{ONNX_BATCH_SIZE}_{EPOCHS}epochs"
        model_path = MODELS_PATH + model_name + '.pt'
        model.save(model_path)
        
        model.export(format='onnx', batch=ONNX_BATCH_SIZE, imgsz=IMG_SIZE, dynamic=False) # CHANGED: ENFORCED IMGSZ AND DYNAMIC=FALSE FOR ONNX EXPORT

if __name__ == "__main__":
    main()