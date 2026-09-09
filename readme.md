
# CRISP-YOLO

**Joint Restoration and Cross-layer Feature Reconstruction for Object Detection in Foggy UAV Imagery**

CRISP-YOLO is a lightweight end-to-end framework for object detection in foggy UAV imagery. It jointly optimizes image restoration and object detection to improve the representation of small and weak targets under adverse weather conditions while maintaining high inference efficiency.

## Overview

UAV images often contain small targets with weak visual features. Fog further reduces image contrast and blurs structural details, making object detection particularly challenging. CRISP-YOLO addresses these challenges through a unified restoration-detection architecture built upon YOLO26s.

The framework incorporates three major components:

* **Converse2D-based Image Restoration (IR)**
  Provides restoration supervision during training to improve feature learning under foggy conditions. The restoration branch is removed during inference to avoid additional deployment cost.
* **SCC3k2**
  Replaces selected standard convolutions with SCConv to suppress spatial and channel redundancy and improve target discriminability.
* **Cross-Layer Connection (CLC) + DySample**
  CLC enhances shallow-deep feature fusion, while DySample provides content-aware upsampling for better spatial alignment and preservation of fine structures.

## Key Results

CRISP-YOLO is evaluated on the synthetic **HazyDet** dataset and the real-world **RDDTS** dataset.

| Dataset |    mAP |
| ------- | --------------: |
| HazyDet | **77.5%** |
| RDDTS   | **54.3%** |

Compared with the retrained YOLO26s* baseline:

* **+2.8 mAP** on HazyDet
* **+10.9 mAP** on RDDTS
* **123.5 FPS** at $768\times768$ on a single NVIDIA RTX 4090
* **9.95M parameters** and **22.9G FLOPs** during inference

The restoration branch is used only during training, resulting in a lightweight inference model.

## Datasets

### HazyDet

HazyDet is a large-scale benchmark for UAV object detection under hazy weather.

* 11,000 synthetic hazy images
* Approximately 365,000 object instances
* Three object classes:

  * Car
  * Truck
  * Bus
* Training / validation / test split: **8:1:2**

### RDDTS

RDDTS is an independent real-world subset containing UAV images captured under genuine fog.

* 600 real-world UAV images
* Captured in urban, rural, and littoral environments
* Diverse flight altitudes and viewing angles
* Training / test split: **2:1**
* No fine-tuning is used for the zero-shot evaluation reported in the paper

For dataset access and licensing information, please refer to the original HazyDet release.

## Environment

The experiments were conducted with:

* Python 3.9
* PyTorch 2.1.0
* CUDA 11.8
* Ultralytics 8.4.40
* NVIDIA RTX 4090 (24 GB)
* Input resolution: $768\times768$

Install the required dependencies with:

```bash
git clone https://github.com/[YOUR_USERNAME]/CRISP-YOLO.git
cd CRISP-YOLO

conda create -n crisp-yolo python=3.9
conda activate crisp-yolo

pip install -r requirements.txt
```

## Training

The complete CRISP-YOLO model is trained with the restoration branch enabled.

Main training settings:

* Epochs: **150**
* Batch size: **10**
* Image size: **768**
* Precision: **FP32**
* Optimizer: **SGD**
* Initial learning rate: **0.01**
* Momentum: **0.937**
* Weight decay: **0.0005**
* Restoration loss weight: **0.8**
* Detection loss weight: **0.2**

Example:

```bash
python train.py
```

Please modify the dataset path and configuration file according to your local environment.

## Efficiency

Inference speed is measured on a single NVIDIA RTX 4090 at $768\times768$.

* Precision: FP32
* FPS is calculated from per-image latency
* Preprocessing is included
* Inference is included
* Postprocessing is included
* No inference warm-up is used

CRISP-YOLO achieves **123.5 FPS** with **9.95M parameters** and **22.9G FLOPs** during inference.

During training, when the restoration branch is enabled:

* Parameters: **10.27M**
* Computational cost: **31.9 GFLOPs**

The restoration branch is discarded at inference to reduce deployment overhead.

## Citation

If you find this work useful, please cite:

```bibtex
@article{CRISPYOLO,
  title   = {Joint Restoration and Cross-layer Feature Reconstruction for Object Detection in Foggy UAV Imagery},
  author = {Meiling Xie and Yijie Ke and Kexin Zhu and Chen Feng and Zhide Chen}
},
  year    = {2026}
}
```

The final BibTeX entry will be updated after publication.

## Acknowledgements

This project is built upon the YOLO framework and incorporates the following related components:

* YOLO26
* SCConv
* Converse2D
* DySample

We thank the authors of these works for making their methods publicly available.
