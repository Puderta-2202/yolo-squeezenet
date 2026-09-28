# YOLO12-SqueezeNet

Implementation of a lightweight YOLO12 object detection model with a SqueezeNet-inspired backbone for coffee leaf disease detection.

This repository contains the modified Ultralytics YOLO12 implementation and experimental notebooks used in the development and evaluation of the YOLO12-SqueezeNet model.

---

## Overview

Coffee leaf diseases can negatively affect coffee production and require early and accurate identification. Deep learning-based object detection can be used to automatically detect disease symptoms from coffee leaf images.

This project explores the integration of a lightweight **SqueezeNet-inspired architecture** into **YOLO12** to reduce model complexity while maintaining effective object detection performance.

The main objective is to develop a lightweight object detection model that can be used for coffee leaf disease detection and has potential for deployment on resource-constrained environments.

---

## Research Objective

The main objectives of this project are:

- Implement YOLO12 with a lightweight SqueezeNet-inspired backbone.
- Reduce computational complexity and model size.
- Maintain effective detection performance.
- Evaluate the proposed architecture through controlled experiments.
- Analyze the effect of data augmentation on detection performance.
- Provide a reproducible implementation for research purposes.

---

## Model Architecture

The proposed architecture combines:

### YOLO12

YOLO12 is used as the main object detection framework. The YOLO architecture follows a one-stage object detection approach, allowing object localization and classification to be performed in a single detection pipeline.

### SqueezeNet

SqueezeNet is used as the inspiration for the lightweight backbone design.

SqueezeNet introduces the **Fire module**, consisting of:

- Squeeze convolution using `1×1` filters.
- Expand layers using `1×1` and `3×3` filters.
- Channel-wise concatenation of the expanded features.

This design aims to reduce the number of parameters and computational requirements while maintaining useful feature representations.

---

## Proposed Architecture

The modified YOLO12 implementation introduces SqueezeNet-inspired components into the model architecture.

The repository contains configuration files for several YOLO12 model variants:

- YOLO12n-SqueezeNet
- YOLO12s-SqueezeNet
- YOLO12m-SqueezeNet
- YOLO12l-SqueezeNet
- YOLO12x-SqueezeNet

The corresponding model configuration files are located inside:

```text
ultralytics/
└── cfg/
    └── models/
        └── 12/
            ├── yolo12n-squeezenet.yaml
            ├── yolo12s-squeezenet.yaml
            ├── yolo12m-squeezenet.yaml
            ├── yolo12l-squeezenet.yaml
            └── yolo12x-squeezenet.yaml
