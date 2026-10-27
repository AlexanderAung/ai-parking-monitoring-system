# Baseline Error Analysis 

## Objective
Understand what types of examples cause the baseline model to make detection errors.

## Experiment 
- Model: RT-DETRv2
- Checkpoint: Best checkpoint, zero-based epoch index 19 (20th completed epoch)
- Dataset: License plate dataset v1
- Validation set: 2048

## Baseline Metrics
- mAP@[0.50:0.95]: 0.6565
- AP50: 0.9323
- AP75: 0.7714
- Small AP: 0.4189
- Medium AP: 0.7143
- Large AP: 0.6686

Source: the `test_coco_eval_bbox` values at epoch index 19 in [log.txt](log.txt).
Despite the field name, these are validation metrics. Prediction-level error
analysis has not yet been recorded. See the [baseline report](report.md).

## Observations



## Error Patterns



## Hypothesis
What do I think is causing the dominant error?

## Next Experiment
What experiment will test the hypothesis?

