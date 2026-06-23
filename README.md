# Farmland - Agricultural Segmentation & Classification

Deep learning models for farmland segmentation and classification using satellite/aerial imagery.

## Quick Start

### 1. Install

```bash
python3 -m venv farmland_env
source farmland_env/bin/activate
pip install -r requirements.txt
```

### 2. Dataset

Download datasets and organize as:
```
data/datasets/
├── train/images/  + _annotations.coco.json
├── valid/images/  + _annotations.coco.json
└── test/images/   + _annotations.coco.json
```

**Public Datasets**: xView, Agriculture-Vision, COCO

Update `DATASET_PATH` in each model file before running.

### 3. Run Models

#### Segmentation
```bash
cd segmentation/deeplabv3_resnet50/
python deeplabv3+resnet50.py          # Train
python predict.py                      # Infer

cd ../unet_resnet50/
python unetresnet50.py                 # Train
python Area.py                         # Calculate area

cd ../hr-net/
python Hrnet.py                        # Train/Infer

cd ../samgeo/
python sammm.py                        # SAM GEO
```

#### Classification
```bash
cd classification/
python amunet.py                       # Train/Infer
jupyter notebook AM-Unet.ipynb         # Notebook
```

#### Utilities
```bash
python utils/Area.py                   # Area calculation
python utils/combine.py                # Combine predictions
python utils/georeferencer.py          # Geospatial processing
python utils/rmse.py                   # Evaluate metrics
```

## Models

| Model | Framework | Type |
|-------|-----------|------|
| DeepLabV3+ResNet50 | PyTorch | Semantic Segmentation |
| U-Net+ResNet50 | TensorFlow | Semantic Segmentation |
| HR-Net | PyTorch | Semantic Segmentation |
| Mask R-CNN | Detectron2 | Instance Segmentation |
| HED | PyTorch | Edge Detection |
| SAM GEO | Geospatial | Segment Anything |
| AM-UNet | PyTorch | Classification |

## Directory Structure

```
Farmland/
├── segmentation/           # Segmentation models
│   ├── deeplabv3_resnet50/
│   ├── unet_resnet50/
│   ├── hr-net/
│   ├── hed/
│   ├── mask-rcnn/
│   └── samgeo/
├── classification/         # Classification models
├── utils/                  # Utilities (area, combine, geo, rmse, etc.)
├── data/datasets/          # Place datasets here
└── test-results/           # Output results
```

## Configuration

Edit model files to update:
- `DATASET_PATH`: Path to your dataset
- `IMG_SIZE`: Image size (typically 256)
- `BATCH_SIZE`: Batch size
- `EPOCHS`: Number of training epochs
- `DEVICE`: GPU or CPU