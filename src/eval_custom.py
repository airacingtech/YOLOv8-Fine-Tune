import ultralytics
from ultralytics import YOLO
import os
import cv2
import time
import torch
import random
import shutil
import tqdm
import argparse


CURR_DIR = os.getcwd()
# WORKSPACE_DIR = os.path.dirname(CURR_DIR)
# DATASETS_DIR = WORKSPACE_DIR + '/datasets/holdout/'
# SSD_DIR = WORKSPACE_DIR + '/../'
# DATA_YAML = WORKSPACE_DIR + '/testing_data.yaml'
# print(DATA_YAML)
# print(WORKSPACE_DIR)
# print(DATASETS_DIR)
# print(SSD_DIR)

# format dataset to be used for evaluation
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
    placed in the 'eval/' directory. This function will delete the 'eval/' directory and recreate
    it. The data.yaml file must be created manually and will be copied over to the 'eval/'
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

    print("Extracting sim images from:", datasets_path, "for evaluation data")

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

    eval_data = datasets_labels

    # Create the directories for the evaluation data
    if os.path.exists(SSD_DIR + "eval_data/"):
        print("Deleting and recreating 'eval/' folder...")
        shutil.rmtree(SSD_DIR + "eval_data/")
    os.mkdir(SSD_DIR + "eval_data/")
    os.mkdir(SSD_DIR + "eval_data/eval/")
    os.mkdir(SSD_DIR + "eval_data/eval/images/")
    os.mkdir(SSD_DIR + "eval_data/eval/labels/")

    # Copy over images and labels to new directories
    print("Copying images and labels to new directories...")
    print("Copying evaluation data:")

    new_image_uuid = 0
    empty_frames_kept = 0
    weighted_frames = {}
    removed_frames = {}
    eval_frames = 0

    for img_src, label_src, dataset_path in tqdm.tqdm(eval_data):
        new_image_name = img_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".jpg"
        new_label_name = label_src.split("/")[-1][:-4] + "_" + str(new_image_uuid) + ".txt"
        img_dst = os.path.join(SSD_DIR + "eval_data/eval/images/", dataset_path + "_" + new_image_name)
        label_dst = os.path.join(SSD_DIR + "eval_data/eval/labels/", dataset_path + "_" + new_label_name)
        copy_data_yaml(label_src, img_src, label_dst, img_dst, empty_frames_kept, weighted_frames, removed_frames)
        new_image_uuid += 1
        eval_frames += 1

    # Copy over data.yaml file from root directory
    shutil.copy(data_yaml, SSD_DIR + "eval_data/")
    print("Copied over 'data.yaml' file")
    print("Number of empty frames kept: ", empty_frames_kept)
    print("Number of evaluation frames: ", eval_frames)
    print("Additional weighted frames created for each dataset: ", weighted_frames)
    print("Number of frames removed for each dataset: ", removed_frames)
    print("Finished creating directories for YOLOv8 evaluation pipeline")

def main():
    parser = argparse.ArgumentParser(description="Format datasets for YOLOv8 evaluation.")
    parser.add_argument("--workspace_dir", type=str, required=True, help="Path to the workspace directory")
    parser.add_argument("--datasets_dir", type=str, required=True, help="Path to the datasets directory")
    parser.add_argument("--data_yaml", type=str, required=True, help="Path to the data.yaml file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLOv8 model file")
    args = parser.parse_args()


    WORKSPACE_DIR = args.workspace_dir
    DATASETS_DIR = args.datasets_dir
    DATA_YAML = args.data_yaml
    CURR_MODEL = args.model_path

    # CHANGE THESE VARIABLES AS NEEDED
    # Whether or not to keep empty frames (frames with no labels) in the dataset
    KEEP_EMPTY_FRAMES = True
    # Percentage of empty frames to keep in the dataset if KEEP_EMPTY_FRAMES is True (randomly sampled)
    PERCENTAGE_EMPTY_FRAMES_TO_KEEP = 1.0

    DATASET_WEIGHTS = {
        'large': 0.1 # Remove extra images from dataset with no frameskipping
    }
    
    format_datasets(DATASETS_DIR, DATA_YAML)

    # Load model and run validation
    CURR_MODEL_PATH = CURR_DIR + "/" + CURR_MODEL
    model = YOLO(CURR_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = model.val(data=DATA_YAML)
    print(results.results_dict)

if __name__ == "__main__":
    main()



