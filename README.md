# NYCU Computer Vision 2025 Spring HW1


Author: 黃皓君  
StudentID: 111550034

## Introduction

The task is to implement a deep-learning-based approach for image classification. The core idea is to improve model robustness and accuracy by leveraging ensemble learning with bagging, combined with test-time augmentation (TTA). In my approach, I use the ResNeXt101_32x8d architecture as our backbone due to its strong performance on ImageNet.
## How to Install
> **Example**:
> 1. Clone this repository:
>    ```bash
>    git clone https://github.com/yourusername/yourrepo.git
>    ```
> 2. Run the main training script:
>    ```bash
>    python bagging.py
>    ```
> 3. Use essemble script to combine multiple model:
>    ```bash
>    python essemble.py
>    ```
## Performance Snapshot

![snapshot](./image/snapshot.png)  
