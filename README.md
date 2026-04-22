# YOLOv8-Fine-Tune — AI Racing Tech

Fine-tuning pipeline for YOLOv8 instance segmentation on AI Racing Tech (ART)
footage. The model is trained as a **single-class `car` detector** that runs
on both prime-lens and fisheye-lens cameras from multiple car views.

Labels come from SAM2 segmentation masks which are converted to YOLO polygon
format, flattened to one class (`0: car`), and split into train/val/test.

## Repo layout

```
YOLOv8-Fine-Tune/
├── environment.yml         # conda env for training + eval (pinned, recommended)
├── requirements.txt        # minimal pip alternative for machines with CUDA 12.1 + PyTorch pre-installed
├── testing_data.yaml       # single-class data.yaml used for held-out eval
├── src/
│   ├── sam2_finetune_med.py  # end-to-end: SAM2 masks -> YOLO labels -> finetune -> ONNX export
│   ├── eval.ipynb            # evaluate a trained .pt / .onnx on held-out bags
│   └── onnx_inference.py     # run a trained model over a video and save annotated output
```

Everything under `data/`, `eval_data/`, `sam2_labeled_data/`, `src/runs/`, and
all `*.pt` / `*.onnx` weights are gitignored.

## Install

**Option A — conda (recommended, exact pinned environment):**
```bash
conda env create -f environment.yml
conda activate yolo-env
```
> `environment.yml` pins the exact environment used for ART training. It
> includes both CUDA 11.8 and 12.1 packages due to how pip wheels were
> installed alongside the conda runtime. If `conda env create` fails with
> libcublas conflicts on a fresh machine, use Option B instead.

**Option B — pip on a machine with CUDA 12.1 and PyTorch already installed:**
```bash
pip install -r requirements.txt
```

## Expected data layout

SAM2-labeled rosbags live **outside** the repo:

```
<workspace>/
├── YOLOv8-Fine-Tune/            # this repo
└── sam2_labeled_data/
    ├── bag1/
    │   ├── images/    # *.jpg / *.png
    │   └── masks/     # *.json — see format below
    ├── bag2/
    └── ...
```

Each mask JSON is a dict keyed by SAM2 object id, with a 2D binary matrix
matching the image dimensions:

```json
{"1": [[0, 0, ..., 0],
       [0, 0, ..., 0],
                ...
       [1, 1, ..., 0]]}
```

The conversion step in `sam2_finetune_med.py` turns every mask into a YOLO
segmentation polygon with class id `0` (car). All objects in all bags collapse
to the single `car` class regardless of the SAM2 object id.

## `data.yaml`

`sam2_finetune_med.py` auto-generates `data/data.yaml` inside the destination
directory on every run, so you do not need to create or maintain one manually.
The generated file always contains:

```yaml
train: train/images
val:   valid/images
test:  test/images

nc: 1

names:
  0: car
```

`testing_data.yaml` at the repo root is a separate file used by `eval.ipynb`
for held-out evaluation (points at `eval_data/eval/images`).

## Fine-tuning — `sam2_finetune_med.py`

Run from `src/`:

```bash
cd src
python sam2_finetune_med.py <dataset_dir> <data_dest_dir> [flags]
```

The script will:

1. Convert every `masks/*.json` in each bag under `<dataset_dir>` to a
   YOLO polygon label file under `<bag>/labels/` (class id `0`).
2. Split bags (not individual frames) 70/20/10 into train/val/test to prevent
   temporal leakage — adjacent frames in a ROS bag are near-identical, so a
   per-frame shuffle inflates val mAP without reflecting real generalization.
3. Delete and rebuild `<data_dest_dir>/data/{train,valid,test}/{images,labels}/`.
4. Auto-generate `data/data.yaml` with `nc: 1` and `names: {0: car}`.
5. Fine-tune `yolov8{size}-seg.pt` for `EPOCHS` epochs (see constants at the
   top of the script — `IMG_SIZE=1056`, `LAYER_FREEZE=10`, augmentation knobs,
   etc.).
6. Run inference on the held-out test split and save annotated frames.
7. Export a batch-1 `.onnx` at `IMG_SIZE=1056`. Both the `.pt` checkpoint and
   the `.onnx` export are saved by Ultralytics to
   `runs/segment/<run_name>/weights/` (relative to `src/`).

### Arguments

| Argument          | Type  | Required | Description |
|-------------------|-------|----------|-------------|
| `dataset_dir`     | str   | yes      | Directory holding SAM2-labeled bags (see layout above). |
| `data_dest_dir`   | str   | yes      | Destination directory; the script creates `<data_dest_dir>/data/` inside it. |
| `--model_size`    | str   | no       | One of `n`, `s`, `m`, `l`, `x`. Default `m`. |
| `--format_only`   | flag  | no       | Only build the YOLO dataset, skip training. |
| `--finetune_only` | flag  | no       | Skip SAM2 → YOLO conversion and dataset rebuild entirely; train on existing `data/`. |
| `--resume`        | flag  | no       | Resume training from `--resume_path`. |
| `--resume_path`   | str   | no       | Checkpoint `.pt` to resume from. |

### Examples

```bash
# Standard end-to-end run (medium model)
python sam2_finetune_med.py ../../sam2_labeled_data/ ../../

# Extra-large model
python sam2_finetune_med.py ../../sam2_labeled_data/ ../../ --model_size x

# Resume from a previous checkpoint
python sam2_finetune_med.py ../../sam2_labeled_data/ ../../ \
    --resume --resume_path runs/segment/yolov8m-img_size_1056_layers_frozen_10_2025-04-07-01-33-24/weights/best.pt

# Skip label conversion and dataset rebuild (data/ already exists from a previous run)
python sam2_finetune_med.py ../../sam2_labeled_data/ ../../ --finetune_only
```

### Generated dataset shape

```
data/
├── data.yaml          ← auto-generated (nc: 1, names: {0: car})
├── train/ {images, labels}
├── valid/ {images, labels}
└── test/  {images, labels}
```

Each `labels/*.txt` line is one polygon: `0 x1 y1 x2 y2 ...` with normalized
coordinates. `0` is the single `car` class.

## Evaluation — `src/eval.ipynb`

Evaluates a trained `.pt` or `.onnx` checkpoint against a directory of
held-out bags that already have YOLO-format labels. Edit the config cell at
the top of the notebook to point at:

- `DATASETS_DIR` — held-out bags (same `bag/{images,labels}/` layout).
- `DATA_YAML` — `testing_data.yaml`.
- `MODEL_PATH` — trained weights.
- `IMG_SIZE` — must match training / ONNX export size (`1056` by default).

The notebook flattens any legacy class ids in label files to `0`, builds a
flat `eval_data/eval/{images,labels}/` directory, and runs `model.val()` to
report box + mask precision / recall / mAP.

If your held-out bags only have SAM2 masks, run the label-generation step
first:

```bash
python sam2_finetune_med.py <holdout_bags_dir> /tmp/unused/ --format_only
```

(or call `format_sam2_labels(<holdout_bags_dir>)` directly from a Python
shell).

## Video inference — `src/onnx_inference.py`

Runs a trained model over a video and writes an annotated `.mp4`:

```bash
# Weights are saved to runs/segment/<run_name>/weights/ after training.
python onnx_inference.py \
    --video_path  ../videos/ims_run1_front.mp4 \
    --model_path  runs/segment/<run_name>/weights/best.onnx \
    --output_path ../videos/ims_run1_front_annotated.mp4 \
    --imgsz 1056
```

`--imgsz` must match the size the ONNX model was exported with (1056 by
default). `.pt` checkpoints also work.
