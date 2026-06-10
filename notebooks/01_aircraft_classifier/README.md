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

I started with smaller CNN baselines and early transfer learning experiments to validate the pipeline before moving to larger architectures.

EfficientNetV2 L ended up giving the best balance of feature quality and prediction stability on difficult aircraft classes, especially during TTA inference where smaller models were less consistent.

The final pipeline combines ImageNet pretrained EfficientNetV2 L, fine tuning, aggressive augmentation, and fixed N = 15 TTA inference to improve classification reliability on ambiguous aircraft silhouettes.

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

```python
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

```text
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

```bash
jupyter notebook
```

Open either notebook:

```text
01_aircraft_classifier_v1.ipynb
```

or

```text
02_aircraft_classifier_v2.ipynb
```

### Required packages

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

### GPU recommendation

EfficientNetV2 L training is GPU intensive.

Cloud GPU environments are strongly recommended for full dataset training.
