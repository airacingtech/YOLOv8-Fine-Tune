import cv2 
import numpy as np
import onnxruntime as ort
import os 
import argparse
from ultralytics import YOLO
from tqdm import tqdm


def run_segmentation(video_path, model_path, output_path, img_size=1056):
    model = YOLO(model_path, task='segment')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


    frame_count = 0
    with tqdm(total = total_frames, desc="Processing Video", unit="frame", ncols=100) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=img_size, verbose=False)
            annotated_frame = results[0].plot()
            out_vid.write(annotated_frame)

            pbar.update(1)

    cap.release()
    out_vid.release()
    print(f"Saved segmented video to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 segmentation on a video.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--model_path', type=str, required=True, help='Path to YOLOv8 segmentation .pt model')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save annotated video')
    parser.add_argument('--imgsz', type=int, default=1056, help='Image size for YOLOv8 input')

    args = parser.parse_args()

    run_segmentation(
        video_path=args.video_path,
        model_path=args.model_path,
        output_path=args.output_path,
        img_size=args.imgsz
    )

if __name__ == '__main__':
    main()