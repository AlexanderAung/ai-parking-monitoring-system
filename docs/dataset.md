# Dataset

## 1. Dataset Source

[Official_roboflow_dataset_link](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/13)

## 2. Dataset Structure
train/
valid/
test/

## 3. Dataset Statistics
| Split | Images | Annotations |
|------|-------:|------------:|
| Train | 98798 | 102844|
| Validation | 2048| ... |
| Test | 1020| ... |

## 4. Classes
- license_plate


## 5. Data Augmentations 
The following augmentations techniques are applied to the dataset and the output per training example is 14
1. Outputs per training example: 14
2. Flip: Horizontal
3. Crop: 0% Minimum Zoom, 15% Maximum Zoom
4. Rotation: Between -10° and +10°
5. Shear: ±2° Horizontal, ±2° Vertical
6. Grayscale: Apply to 10% of images
7. Hue: Between -15° and +15°
8. Saturation: Between -15% and +15%
9. Brightness: Between -15% and +15%
10. Exposure: Between -15% and +15%
11. Blur: Up to 0.5px
12. Cutout: 5 boxes with 2% size each

## 6. Data Cleaning
To effectively use the data, we need to analyze the dataset and take out relevant data for the project. 
In our case, we should start off by calculating the bounding box area relative to the pic of the actual footage, CCTV that we will be using. 



## 7. Final Dataset
...

## 8. Limitations
...
