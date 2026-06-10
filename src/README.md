# Source Modules

This folder contains the core inference pipeline behind ATAS including aircraft classification, metadata lookup, physics feature generation, ML inference, and tactical decision logic.

---

## File overview

| File                                   | Purpose                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------- |
| `classifier.py`                        | Loads the EfficientNetV2 L aircraft classifier and performs TTA inference  |
| `metadata.py`                          | Retrieves aircraft specifications from the metadata CSV                    |
| `physics_generator.py`                 | Generates synthetic engagement data and assembles inference feature arrays |
| `models.py`                            | Loads the ETA regressor and hit classifier and performs inference          |
| `decision.py`                          | Converts model outputs into tactical recommendations                       |
| `schemas.py`                           | Shared constants for feature order, model paths, and thresholds            |
| `analysis_for_physics_generator.ipynb` | Exploratory validation notebook for the synthetic engagement dataset       |

---

## classifier.py

Handles aircraft image classification using the trained EfficientNetV2 L model.

Main function:

```python
predict_aircraft(image_path)
```

Key responsibilities:

* Image preprocessing
* TTA generation
* Batched inference
* Aircraft class prediction

### Important design decisions

TTA is fixed at `N = 15` and mirrors the exact augmentation pipeline used during training:

* Horizontal flip
* Vertical flip
* Brightness jitter
* Contrast jitter

TensorFlow memory growth is also enabled so VRAM is allocated dynamically instead of reserving all GPU memory at startup.

---

## metadata.py

Acts as the lookup layer between the classifier and the physics pipeline.

Once an aircraft is identified, this module retrieves:

* Missile speed
* Missile range
* Aircraft generation
* Maneuverability
* Air to air capability

Main function:

```python
get_aircraft_metadata(aircraft_name)
```

The lookup is case insensitive and returns a dictionary from `aircraft_metadata.csv`.

---

## physics_generator.py

Core simulation layer used for both training and inference.

Main responsibilities:

* Monte Carlo engagement generation
* Physics derived labels
* Closure rate calculation
* Missile phase derivation
* Evasion time generation
* Hit outcome generation
* Inference feature assembly

### Key functions

```python
generate_dataset()
```

Creates 1,000,000 synthetic engagement rows.

```python
build_feature_array()
```

Builds the exact 14 feature dictionary required during inference.

### Important design decisions

Only valid threat scenarios are included:

```python
closure_rate > 0
```

Invalid engagements are filtered out during dataset generation.

The module also separates:

* Physics solvable calculations
* Learned behavioral outcomes

which became the main architectural idea behind ATAS.

---

## analysis_for_physics_generator.ipynb

Small validation notebook used to inspect the synthetic engagement dataset before model training.

Used for:

* Distribution checks
* Feature sanity validation
* Label inspection
* Relationship analysis between generated variables

This notebook exists mainly to verify that the generator produces realistic engagement patterns before training begins.

---

## models.py

Handles inference for:

* ETA regressor
* Hit classifier

The models are loaded once into memory during module import to avoid repeated disk loading during requests.

Main function:

```python
make_predictions(feature_dict)
```

Returns:

* `eta_seconds`
* `hit_probability`

### Important design decisions

The feature array always follows the exact order defined in:

```python
FEATURE_COLUMNS
```

ETA predictions are clipped at zero during every inference call:

```python
np.maximum(model.predict(features), 0)
```

because negative evasion time has no physical meaning.

---

## decision.py

Converts model outputs into pilot facing tactical recommendations.

Inputs:

* ETA prediction
* Hit probability
* Platform metadata

Output:

* Human readable tactical recommendation

Main function:

```python
get_recommendation()
```

### Important design decisions

The decision layer is intentionally rule based instead of learned.

This keeps the tactical logic:

* Interpretable
* Debuggable
* Easy to modify
* Easy to validate

---

## schemas.py

Shared configuration layer for the entire pipeline.

Without this file, multiple modules would hardcode:

* Feature order
* Model paths
* Threshold values

### Shared constants

| Constant              | Purpose                         |
| --------------------- | ------------------------------- |
| `FEATURE_COLUMNS`     | Exact 14 feature training order |
| `MODEL_PATHS`         | Paths to saved models           |
| `DECISION_THRESHOLDS` | Tactical decision cutoffs       |
| `BASE_DIR`            | Root project path               |

---

## Data flow

Single `/analyze` request flow:

```text
Image Upload
      ↓
classifier.py
      ↓
Predicted Aircraft Name
      ↓
metadata.py
      ↓
Aircraft Specifications
      ↓
physics_generator.py
      ↓
14 Feature Array
      ↓
models.py
      ↓
ETA + Hit Probability
      ↓
decision.py
      ↓
Tactical Recommendation
```

---

## Key constraints

These rules are intentionally hard coded into the pipeline and should not be changed casually.

### Legacy Keras compatibility

```python
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

must be set before all TensorFlow imports.

---

### ETA clipping rule

```python
np.maximum(model.predict(features), 0)
```

is always applied during inference because negative evasion time is physically invalid.

---

### TTA locked at N = 15

The classifier inference pipeline always uses:

```python
n_augments = 15
```

This value is intentionally fixed to match the final evaluated inference configuration.

---

### Shared feature ordering

All modules import feature order from:

```python
src.schemas
```

The 14 feature array cannot be reordered.

---

### Import structure

All internal imports use the `src.` prefix:

```python
from src.schemas import FEATURE_COLUMNS
```

This prevents path inconsistencies between local execution, Docker, and Hugging Face deployment.

---

### Uvicorn startup rule

The FastAPI server is started with:

```bash
uvicorn app.main:app
```

The `--reload` flag is intentionally avoided in deployment environments.
