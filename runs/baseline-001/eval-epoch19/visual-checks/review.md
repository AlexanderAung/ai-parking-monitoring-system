## Qualitative Review of Validation Annotations

I individually did visual inspection to qualify annotation-quality issues and differences between the validation data and the intended use case. Below are my findings. 


## Large plates

| Example Image | Observation |
|---|---|
| `00b6f96c840767fa_jpg.rf.633b5af0bbe4e4cb5abb5cf5236c3140.jpg` | Bounding box is excessively loose and unsuitable as a reliable evaluation reference. |
| `Saudi_Arabia_license_plate_KSA_jpeg_jpg.rf.7431b4b5ad08347bdc12bb4fc6c515aa.jpg` | Bounding box is excessively loose and incorrectly sized. |

#### `IMG_` image group

This group contains images whose filenames start with `IMG_`,  pattern: `IMG_XXXX-conv`  
- example:
`IMG_6987-conv_jpeg_jpg.rf.7986b7ca6b375d78bff5ad8fed4baf16.jpg`.
- The annotations in the reviewed group members were all judged unacceptable. The specific defects require further categorization.


#### `pic_` image group

This group contains images whose filenames start with `pic_`, pattern: `pic_XXX` 
- Example: `pic_004_jpg.rf.7118b24ff92da96367d523c643555559.jpg`.
- The annotations in the reviewed group members were all judged unacceptable. The specific defects require further categorization.


## Medium plates

| Example Image | Observation |
|---|---|
| `CarLongPlate8_jpg.rf.6e8c01c850c41861aa2366a73d331e05.jpg` | Among the highest-quality annotations reviewed. The image is also relevant to the intended parking-entry scenario. |

#### `xemay` image group

This group contains images whose filenames start with `xemay`, pattern: `xemay*`

- Example: `xemay13_jpg.rf.e5165c7fc425b4f01ef45ef819b4e766.jpg`.
- Poor annotation quality was observed across the reviewed group members. All reviewed members include motorcycle plates, which differ from the project’s intended focus on car license plates at parking entrances.
- Motorcycle plates remain valid instances of the license-plate class. Their relevance to the target application is a separate concern from bounding-box correctness.

## Small plates

#### `quandoi` image group

This group contains images whose filenames start with `quandoi`, pattern: `quandoi*`

- Example: `quandoi23_jpg.rf.e09ebc1a704ff51e0d6f48e99fe35435.jpg`.
- Annotations in the reviewed group members appeared acceptable during visual inspection.


### Visual inspection side by side with GT and predictions

Large-plate predictions generally appeared well aligned with the visible plates. In some cases, predictions fitted the plate more accurately than the supplied ground-truth boxes, highlighting annotation-quality limitations.

Predicted boxes around medium and small plates were often somewhat loose. Small absolute boundary errors represent a larger proportion of a small plate’s dimensions and can therefore substantially reduce IoU, especially at stricter evaluation thresholds.

No systematic box shifts or stretching were observed in the reviewed comparisons. This supports correct mapping of predictions back to original-image coordinates, although the review does not establish correctness for every validation image.

This is consistent with small-plate AP falling from 0.7363 at IoU 0.50 to 0.4311 at IoU 0.75. The scores alone don’t establish the cause;  visual observations provide supporting evidence.

### Implications for the baseline

The reproduced metrics describe agreement with the supplied annotations. Loose or incorrectly sized ground-truth boxes can penalize otherwise accurate predictions, particularly at stricter IoU thresholds. Oversized annotations can also place plates into an inappropriate COCO size group, affecting interpretation of size-specific AP.

These findings justify a structured annotation audit before attributing performance gaps entirely to the model. They also motivate a separate review of the training annotations, but do not establish that the training set has the same defects.

The current baseline should therefore be retained as a reproducible reference, with annotation quality documented as a limitation.
