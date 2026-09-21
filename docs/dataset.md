# Dataset documentation 

## 1. Objective

## 2. Source Dataset
- Roboflow 
[dataset_link](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/13)

## 3. Dataset Overview
The original dataset has already been done a data augmentations 
### Augmentations
- Outputs per training example: 14
- Flip: Horizontal
- Crop: 0% Minimum Zoom, 15% Maximum Zoom
- Rotation: Between -10° and +10°
- Shear: ±2° Horizontal, ±2° Vertical
- Grayscale: Apply to 10% of images
- Hue: Between -15° and +15°
- Saturation: Between -15% and +15%
- Brightness: Between -15% and +15%
- Exposure: Between -15% and +15%
- Blur: Up to 0.5px
- Cutout: 5 boxes with 2% size each

## 4. Dataset Selection
### Data Audit 
Before selecting any images, do data audit with src/data/audit.py 
1. Check metadata 
2. Find and flag, bboxes that are less than 0.05% of image area, bec it is probably unlearnable
3. Find and flag, plates(anno bbox) covering 90% of the frame which is suspicious

### Issues 
Found 2278 warning issues, check with issues-checker.py






## 5. Data Cleaning

## 6. Dataset Split

## 7. Data Distribution

## 8. Final Dataset

## 9. Limitations


