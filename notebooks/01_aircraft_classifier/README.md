# Aircraft Classifier

This folder contains the complete development pipeline for the ATAS aircraft image classification system used in Stage 2 of the project.

---

## Notebook overview

| Notebook                          | Purpose                                                                                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_aircraft_classifier_v1.ipynb` | Initial end to end classifier development including preprocessing validation, baseline CNN experimentation, augmentation testing, and early transfer learning |
| `02_aircraft_classifier_v2.ipynb` | Final EfficientNetV2 L production pipeline with fine tuning, TTA inference, and held out evaluation                                                           |

---

## Dataset

| Property      | Value                               |
| ------------- | ----------------------------------- |
| Dataset       | Military Aircraft Detection Dataset |
| Source        | Kaggle                              |
| Total images  | 41,441                              |
| Classes       | 101 aircraft types                  |
| Test set size | 2,769                               |
| Label format  | Folder per class                    |

Dataset link:

[Military Aircraft Detection Dataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset?utm_source=chatgpt.com)

The dataset is a fine grained aircraft classification problem with:

* High inter class similarity
* Similar aircraft silhouettes
* Variable viewing angles
* Lighting variation
* Background clutter
* Class imbalance

Random chance baseline accuracy is approximately 0.99%, making the classification task intentionally difficult.

---

## Model evolution

The classifier pipeline began with smaller baseline experiments before moving toward larger transfer learning architectures.

The first notebook focused on validating the full image pipeline:

* Dataset loading
* Label mapping
* Augmentation behavior
* GPU training stability
* Baseline CNN experimentation
* Early transfer learning tests

Once the pipeline stabilized, the project transitioned to EfficientNetV2 L for final production training.

EfficientNetV2 L was selected because classification accuracy was prioritized over lightweight inference speed. Smaller architectures trained faster, but struggled to consistently separate visually similar aircraft classes across difficult viewing conditions.

Compared to earlier baselines, EfficientNetV2 L provided:

* Stronger feature extraction on fine aircraft details
* Better confidence separation between similar aircraft
* Higher Top 1 accuracy after fine tuning
* More stable predictions during TTA inference

The final production pipeline combines:

* ImageNet pretrained EfficientNetV2 L
* Fine tuning
* Aggressive augmentation
* TTA inference at N = 15

to maximize classification reliability on ambiguous aircraft silhouettes.

---

## Model architecture

| Component            | Configuration                         |
| -------------------- | ------------------------------------- |
| Base model           | EfficientNetV2 L                      |
| Pretraining          | ImageNet                              |
| Framework            | TensorFlow                            |
| Pooling layer        | GlobalAveragePooling2D                |
| Classification head  | Dense → Dropout → Dense(101, softmax) |
| Serialization format | `.keras`                              |

The notebook sets:

```python id="39g9hy"
TF_USE_LEGACY_KERAS = "1"
```

before TensorFlow imports to maintain serialization compatibility for `.keras` model export.

---

## Training pipeline

1. Data loading and directory setup

2. Image preprocessing using EfficientNetV2 preprocessing utilities

3. Training augmentation:

   * Horizontal flip
   * Vertical flip
   * Random brightness
   * Random contrast

4. Baseline training with frozen EfficientNetV2 L backbone

5. Fine tuning with partial backbone unfreezing and lower learning rate

6. Test Time Augmentation inference with N = 15 passes per image

7. Final evaluation on the held out test set

---

## Augmentation and TTA

### Training augmentation

| Augmentation      | Configuration     |
| ----------------- | ----------------- |
| Horizontal flip   | Enabled           |
| Vertical flip     | Enabled           |
| Random brightness | `max_delta = 0.1` |
| Random contrast   | `0.9 → 1.1`       |

### Test Time Augmentation

| Parameter            | Value                                  |
| -------------------- | -------------------------------------- |
| TTA passes           | 15                                     |
| Aggregation method   | Mean softmax probability               |
| Inference transforms | Same augmentation pipeline as training |

TTA significantly improves prediction stability on ambiguous aircraft silhouettes and difficult viewing angles.

N = 15 is intentionally fixed and treated as part of the final inference pipeline rather than a tunable runtime parameter.

---

## Final metrics

| Metric             | Score                      |
| ------------------ | -------------------------- |
| Top 1 Accuracy     | 78.08%                     |
| Top 5 Accuracy     | 92.02%                     |
| Evaluation dataset | 2,769 held out test images |
| Number of classes  | 101                        |

Top 5 accuracy above 92% means the correct aircraft class is almost always present within the model’s highest confidence predictions.

---

## Saved model

```text id="5kr7xj"
release_models/
└── aircraft_classifier/
    └── atas_final_fine_tuned_aircraft_classifier_model.keras
```

| Property         | Value             |
| ---------------- | ----------------- |
| Format           | `.keras`          |
| Size             | 898 MB            |
| Storage location | Hugging Face only |

The final model is intentionally stored in the Hugging Face repository instead of GitHub because the serialized model exceeds practical GitHub LFS limits.

Model backup:

[ATAS Models Repository](https://huggingface.co/Eakempreet/ATAS-models?utm_source=chatgpt.com)

This routing decision reflects a real deployment and repository management constraint encountered during the project.

---

## Environment notes

| Environment                   | Purpose                          |
| ----------------------------- | -------------------------------- |
| WSL2 + VS Code + RTX 3050 4GB | Local development and debugging  |
| Kaggle P100 GPU               | Full dataset training            |
| Lightning AI T4 GPU           | Additional cloud experimentation |

Full dataset training was performed on cloud GPUs due to local VRAM limitations with EfficientNetV2 L and large batch pipelines.

---

## How to run

### Open locally with Jupyter

```bash id="4lg5rt"
jupyter notebook
```

Open either notebook:

```text id="bn3i1m"
01_aircraft_classifier_v1.ipynb
```

or

```text id="95br5h"
02_aircraft_classifier_v2.ipynb
```

### Required packages

```bash id="9m2yvw"
pip install tensorflow pandas numpy matplotlib scikit-learn
```

### GPU recommendation

EfficientNetV2 L training is GPU intensive.

Cloud GPU environments are strongly recommended for full dataset training.
