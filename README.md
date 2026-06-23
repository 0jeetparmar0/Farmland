# Farmland - Agricultural Segmentation & Classification

Deep learning models for farmland segmentation and classification using satellite/aerial imagery.

## Quick Start

### 1. Install

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.4.0-0-Linux-x86_64.sh
bash ./Miniconda3-py311_24.4.0-0-Linux-x86_64.sh
conda create -n farmland python=3.12 
conda activate farmland

pip install -r requirements.txt
```

### 2. Dataset

https://universe.roboflow.com/jeet-dfono/farm-boundarys
Download datasets and organize as:
```
data/datasets/
├── train/images/  + _annotations.coco.json
├── valid/images/  + _annotations.coco.json
└── test/images/   + _annotations.coco.json
```

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
  
### Metrics Calculated

- **Accuracy**: Pixel-wise accuracy
- **Precision/Recall**: For each class
- **F1-Score**: Balanced metric
- **IoU (Intersection over Union)**: Segmentation quality
- **RMSE**: Against ground truth (if available)

## 🎓 Tips & Best Practices

1. **Data Preprocessing**:
   - Normalize images (typically divide by 255 for 0-1 range)
   - Ensure consistent image sizes
   - Check COCO JSON annotations are valid

2. **Training**:
   - Start with a pretrained model (most models use ImageNet weights)
   - Use data augmentation (Albumentations included)
   - Monitor GPU memory with `nvidia-smi`

3. **Inference**:
   - Use batch processing for speed
   - Image chunking for large images (see `utils/chunks.py`)
   - Save intermediate results for debugging

4. **Evaluation**:
   - Always validate on unseen test set
   - Calculate multiple metrics (not just accuracy)
   - Visualize false positives/negatives

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce BATCH_SIZE or IMG_SIZE |
| Model not loading | Check file paths in configuration |
| Low accuracy | Try different model, augment data, increase epochs |
| Slow inference | Enable GPU, use batch processing |
| COCO annotation errors | Validate JSON format, check image paths |

---

## 📝 License

This project is provided as-is for research and educational purposes.

---

## 📧 Contributing

For improvements or bug reports, please submit issues or pull requests.

---

**Last Updated**: 2026
