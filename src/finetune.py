import ultralytics
from ultralytics import YOLO
import os
import cv2
import time
import torch
import random
import shutil
import tqdm

# Set constants and parameters 
# Change these parameters to fit your needs
EPOCHS = 50
NUM_TRAIN_LOOPS = 1
IMG_SIZE = 1032 # YOLOv8 default is 640
LAYER_FREEZE = 0 # Number of layers to freeze

# Amount to use different data augmentations
HSV_H = 0.1  # Modifies hue
HSV_S = 0.7  # Modifies saturation
HSV_V = 0.4  # Modifies value (brightness)
DEGREES = 0.4  # Rotates the image randomly
TRANSLATE = 0.3
SCALE = 0.5
SHEAR = 0.01
PERSPECTIVE = 0.001
FLIPUD = 0.3
FLIPLR = 0.3
BGR = 0.1  # Flips channels from RGB to BGR
MOSAIC = 0.5
MIXUP = 0.5
COPY_PASTE = 0.4
ERASING = 0.2
CROP_FRACTION = 0.1

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

CURR_DIR = os.getcwd()
WORKSPACE_DIR = os.path.dirname(CURR_DIR)
DATASETS_DIR = WORKSPACE_DIR + '/../../../../media/shared_ssd/cvat_exported_id2/'
SSD_DIR = WORKSPACE_DIR + '/../../../../media/shared_ssd/'
DATA_YAML = WORKSPACE_DIR + '/data.yaml'
print(WORKSPACE_DIR)
print(DATASETS_DIR)
print(SSD_DIR)
# Percetnage of dataset to use for training
TRAIN_PERCENTAGE = 1.0

# Whether or not to keep empty frames (frames with no labels) in the dataset
KEEP_EMPTY_FRAMES = True
# Percentage of empty frames to keep in the dataset if KEEP_EMPTY_FRAMES is True (randomly sampled)
PERCENTAGE_EMPTY_FRAMES_TO_KEEP = 0.8

# Flag to resume training from a previous checkpoint (false if training from scratch)
RESUME_TRAINING = True
#RESUME_TRAINING_PATH = WORKSPACE_DIR + 'runs/train/exp/weights/best.pt'
RESUME_TRAINING_PATH = WORKSPACE_DIR + '/models/yolov8n_2025-02-07-09-28-51_batch4_50epochs.pt'
# Size of YOLOv8 model
MODEL_SIZE = 'n' # 'n' ,'s', 'm', 'l', 'x'

# Path to trained model weights
MODELS_PATH = WORKSPACE_DIR + '/models/'

# This line prevents the Kernel from crashing when running model.train() which calls a plotting function
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

ONNX_BATCH_SIZE = 4

# Todays data + Batch size + epochs
DATE = time.strftime('%Y-%m-%d-%H-%M-%S')

print("CUDA Available: " + str(torch.cuda.is_available()))
print("Torch CUDA Version: " + str(torch.version.cuda))
# Check to make sure CUDA is available and does not say "None"
ultralytics.utils.checks.collect_system_info()

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
        # print("Empty label file detected:", label_src)
        if KEEP_EMPTY_FRAMES:
            if random.random() < PERCENTAGE_EMPTY_FRAMES_TO_KEEP:
                empty_frames_kept += 1
                for i in range(choose_dataset_weight(img_src, DATASET_WEIGHTS, weighted_frames, frames_removed)):
                    img_dst_with_weight = img_dst[:-4] + "_" + str(i) + ".jpg"
                    label_dst_with_weight = label_dst[:-4] + "_" + str(i) + ".txt"
                    shutil.copy(img_src, img_dst_with_weight)
                    generate_empty_label(label_dst_with_weight)
                    # normalize_image(img_dst)
    else:
        for i in range(choose_dataset_weight(img_src, DATASET_WEIGHTS, weighted_frames, frames_removed)):
            img_dst_with_weight = img_dst[:-4] + "_" + str(i) + ".jpg"
            label_dst_with_weight = label_dst[:-4] + "_" + str(i) + ".txt"
            shutil.copy(img_src, img_dst_with_weight)
            shutil.copy(label_src, label_dst_with_weight)
            # normalize_image(img_dst)

def normalize_image(img_path : os.PathLike) -> None:
    '''
    Normalizes the image to the correct format for YOLOv8.
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    cv2.imwrite(img_path, img)

def format_datasets(datasets_path : os.PathLike, data_yaml : os.PathLike) -> None:
    """
    Takes in a directory of datasets where each dataset is in the format of a COCO dataset
    and then formats the datasets into a single dataset that YOLOv8 can use for training
    placed in the 'data/' directory. This function will delete the 'data/' directory and recreate
    it. The data.yaml file must be created manually and will be copied over to the 'data/'
    directory. The function will also split the data into training, validation, and test sets.
        
    Args:
        datasets_path (os.PathLike): The path to the directory containing the datasets.
        data_yaml (os.PathLike): The path to the data.yaml file specifying the dataset. This must be
            created manually.
    
    Returns:
        None
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
    # datasets_labels = sorted(datasets_labels, key = lambda x: x[0])

    # Shuffle images deterministically with seed
    random.seed(0)
    random.shuffle(datasets_labels)

    # Split 70% training, 20% validation, 10% test
    training_data = datasets_labels[:len(datasets_labels) * 7 // 10]
    valid_data = datasets_labels[len(datasets_labels) * 7 // 10 :len(datasets_labels) * 9 // 10]
    test_data = datasets_labels[len(datasets_labels) * 9 // 10 :]


    # Create the directories for the training, validation, and test data
    if os.path.exists(SSD_DIR + "data/"):
        print("Deleting and recreating 'data/' folder...")
        shutil.rmtree("data/")
    os.mkdir(SSD_DIR + "data/")
    os.mkdir(SSD_DIR + "data/train/")
    os.mkdir(SSD_DIR + "data/train/images/")
    os.mkdir(SSD_DIR + "data/train/labels/")
    os.mkdir(SSD_DIR + "data/valid/")
    os.mkdir(SSD_DIR + "data/valid/images/")
    os.mkdir(SSD_DIR + "data/valid/labels/")
    os.mkdir(SSD_DIR + "data/test/")
    os.mkdir(SSD_DIR + "data/test/images/")
    os.mkdir(SSD_DIR + "data/test/labels/")

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
            img_dst = os.path.join(SSD_DIR + "data/train/images/", dataset_path + "_" + new_image_name)
            label_dst = os.path.join(SSD_DIR + "data/train/labels/", dataset_path + "_" + new_label_name)
            copy_data_yaml(label_src, img_src, label_dst, img_dst, empty_frames_kept, weighted_frames, removed_frames)
            new_image_uuid += 1
            train_frames += 1
    for img_src, label_src, dataset_path in tqdm.tqdm(valid_data):
        new_image_name = img_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".jpg"
        new_label_name = label_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".txt"
        img_dst = os.path.join(SSD_DIR + "data/valid/images/", dataset_path + "_" + new_image_name)
        label_dst = os.path.join(SSD_DIR + "data/valid/labels/", dataset_path + "_" + new_label_name)
        copy_data_yaml(label_src, img_src, label_dst, img_dst, empty_frames_kept, weighted_frames, removed_frames)
        new_image_uuid += 1
        valid_frames += 1
    for img_src, label_src, dataset_path in tqdm.tqdm(test_data):
        new_image_name = img_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".jpg"
        new_label_name = label_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".txt"
        img_dst = os.path.join(SSD_DIR + "data/test/images/", dataset_path + "_" + new_image_name)
        label_dst = os.path.join(SSD_DIR + "data/test/labels/", dataset_path + "_" + new_label_name)
        copy_data_yaml(label_src, img_src, label_dst, img_dst, empty_frames_kept, weighted_frames, removed_frames)
        new_image_uuid += 1
        test_frames += 1

    # Copy over data.yaml file from root directory
    shutil.copy(data_yaml, SSD_DIR + "data/")
    print("Copied over 'data.yaml' file")
    print("Number of empty frames kept: ", empty_frames_kept)

    print("Number of training frames: ", train_frames)
    print("Number of validation frames: ", valid_frames)
    print("Number of test frames: ", test_frames)
    print("Additional weighted frames created for each dataset: ", weighted_frames)
    print("Number of frames removed for each dataset: ", removed_frames)
    print("Finished creating directories for YOLOv8 training pipeline")

def choose_model_size() -> str:
    """
    Takes the global variable MODEL_SIZE and returns the corresponding string
    to pass to the YOLO class and print the model parameter size.

    Returns:
        str: The model string to pass to the YOLO class.
    """
    if MODEL_SIZE == 'n':
        print("Using YOLOv8 Nano model")
        return 'yolov8n-seg.pt'
    elif MODEL_SIZE == 's':
        print("Using YOLOv8 Small model")
        return 'yolov8s-seg.pt'
    elif MODEL_SIZE == 'm':
        print("Using YOLOv8 Medium model")
        return 'yolov8m-seg.pt'
    elif MODEL_SIZE == 'l':
        print("Using YOLOv8 Large model")
        return 'yolov8l-seg.pt'
    elif MODEL_SIZE == 'x':
        print("Using YOLOv8 Extra Large model")
        return 'yolov8x-seg.pt'

def train_model(model : YOLO) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    start_time = time.time()
    model_name = f'yolov8{MODEL_SIZE}-img_size_{IMG_SIZE}_layers_frozen_{LAYER_FREEZE}_{DATE}'
    # By default, the model trains on a single GPU
    model.train(
        data=curr_data_yaml,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        freeze=LAYER_FREEZE,
        amp=True,
        cache="disk", # If the size of your dataset is larger than your available memory, cache to disk instead with cache="disk", else use cache=True to keep the dataset in memory
        save=True,
        save_period=10,
        name=model_name,
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
        # bgr=BGR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        erasing=ERASING,
        crop_fraction=CROP_FRACTION
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

def test_model(model : YOLO, test_results_path: os.PathLike) -> None:
    """
    Test the fine-tuned model on test images and save the results.

    Parameters:
        model (YOLO): The fine-tuned YOLO model.
        test_results_path (os.PathLike): The path to save the test results.

    Returns:
        None

    """
    # Make sure the test save path exists
    if not os.path.exists(test_results_path):
        os.makedirs(test_results_path)

    # Inferencee fine-tuned model on test images and save results
    for file in os.listdir(TEST_PATH):
        file_path = os.path.join(TEST_PATH, file)
        output = model.predict(file_path)
        save_path = os.path.join(test_results_path, file)
        cv2.imwrite(save_path, output[0].plot())

    print("Inference on test set complete. Results saved to: ", test_results_path)
    

if __name__ == "__main__":
    # create dataset
    # format_datasets(DATASETS_DIR, DATA_YAML)

    # Load yolov8 nano segmentation model
    if RESUME_TRAINING:
        model = YOLO(RESUME_TRAINING_PATH)
    # Load yolov8 nano segmentation model
    else:
        model = YOLO(choose_model_size())

    # Find dataset images
    data_dir = SSD_DIR + 'data/'
    curr_data_yaml = data_dir + 'data.yaml'
    TEST_PATH = data_dir + '/test/images/'

    epochs_done = 50
    for _ in range(NUM_TRAIN_LOOPS):
        print(f"Starting training loop starting on epoch {epochs_done}")
        train_model(model)
        epochs_done += EPOCHS
        tune_model(model)
        test_results_path = data_dir + '/test/annotation_results' + f'_{epochs_done}epochs'
        test_model(model, test_results_path)
        model_name = f"yolov8{MODEL_SIZE}_{DATE}_batch{ONNX_BATCH_SIZE}_{EPOCHS}epochs"
        model_path = MODELS_PATH + model_name + '.pt'
        model.save(model_path)
        # Export the model as an .onnx file
        model.export(format='onnx', batch=ONNX_BATCH_SIZE)

    # Export the model as an .onnx file
    model.export(format='onnx', batch=ONNX_BATCH_SIZE)