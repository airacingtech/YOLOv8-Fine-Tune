"""Run a fine-tuned YOLOv8 segmentation model over a video and write an
annotated copy. Accepts both .pt and .onnx checkpoints via the ultralytics
YOLO wrapper, which handles backend selection automatically.

Example:
    python onnx_inference.py \\
        --video_path  ../videos/ims_run1_front.mp4 \\
        --model_path  ../models/best.onnx \\
        --output_path ../videos/ims_run1_front_annotated.mp4 \\
        --imgsz 1056
"""
import argparse

import cv2
from tqdm import tqdm
from ultralytics import YOLO


def run_segmentation(video_path: str, model_path: str, output_path: str, img_size: int = 1056) -> None:
    model = YOLO(model_path, task='segment')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Error opening video file: {video_path}')

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        with tqdm(total=total_frames, desc='Processing video', unit='frame', ncols=100) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, imgsz=img_size, verbose=False)
                out_vid.write(results[0].plot())
                pbar.update(1)
    finally:
        cap.release()
        out_vid.release()

    print(f'Saved annotated video to: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Run YOLOv8 segmentation on a video.')
    parser.add_argument('--video_path',  type=str, required=True, help='Path to input video')
    parser.add_argument('--model_path',  type=str, required=True, help='Path to YOLOv8-seg weights (.pt or .onnx)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save annotated video')
    parser.add_argument('--imgsz',       type=int, default=1056,  help='Inference image size (must match ONNX export size)')
    args = parser.parse_args()

    run_segmentation(
        video_path=args.video_path,
        model_path=args.model_path,
        output_path=args.output_path,
        img_size=args.imgsz,
    )


if __name__ == '__main__':
    main()
