# Spatio-Temporal Explainability for 3D CNNs (UCF101 Case Study)

## Overview

This project investigates explainability methods for 3D Convolutional Neural Networks applied to video classification.

We fine-tune a pretrained 3D ResNet-18 (R3D-18) on a binary subset of UCF101 (Basketball vs BasketballDunk) and evaluate temporal explanation quality using quantitative faithfulness metrics and counterfactual analysis.

The goal is not only high accuracy, but validated interpretability.

---

## Model

- Architecture: 3D ResNet-18 (R3D-18)
- Input: (3, 16, 112, 112) video clips
- Pretrained on Kinetics
- Fine-tuned last layers
- Test Accuracy: 100% (binary subset)

---

## Explainability Methods

We implement two spatio-temporal attribution methods:

1. **Grad-CAM (3D)**
   - Applied to final convolutional block
   - Produces spatial heatmaps per frame
   - Temporal importance derived from frame-wise CAM averages

2. **Integrated Gradients (Video)**
   - Path-integrated attribution from baseline
   - Temporal importance computed from mean absolute attribution per frame

---

## Faithfulness Evaluation

We evaluate explanation quality using deletion-based metrics:

- Remove top-k important frames
- Compare probability drop vs random frame removal
- Compute dataset-level deletion AUC

### Dataset-Level Results (20 balanced test videos)

| Method | Mean AUC ↓ | Std |
|--------|------------|-----|
| Grad-CAM | 0.8838 | 0.1232 |
| Integrated Gradients | **0.7567** | 0.2251 |

Lower AUC indicates faster confidence collapse → stronger faithfulness.

Integrated Gradients outperformed Grad-CAM in 90% of samples.

---

## Sanity Check (Model Randomization)

We perform weight randomization testing:

- Randomized model explanations differ significantly
- Indicates explanations depend on learned weights
- Passes sanity criteria from saliency literature

---

## Counterfactual Temporal Analysis

We generate counterfactual videos:

- Temporal frame shuffling
- Fast resampling
- Slow resampling

Findings:

- Model is sensitive to frame selection density
- Less sensitive to strict temporal order
- Explanation shifts under temporal perturbations

---

## Key Findings

- Integrated Gradients shows stronger faithfulness under deletion metrics.
- Grad-CAM produces smoother temporal importance curves.
- Model relies more on motion density than strict ordering.
- Quantitative validation is critical for video explainability.

---

## Reproducibility

1. Install requirements
2. Download UCF101 dataset
3. Run training script
4. Run explanation and evaluation modules

---

## Future Work

- Multi-class extension (full UCF101)
- Alternative video architectures (I3D, SlowFast)
- Additional attribution methods
- Robustness benchmarking across datasets
